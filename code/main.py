import gc
import json
import os
import torch

from time import sleep
from models.florence import generate_response
from models.ppe import ppe_pred

# Debug flags
DEBUG = False
METRIC = True

# PPE classes
PPE_CLASSES = [
    "Gloves",
    "Goggles",
    "Hardhat",
    "Mask",
    "Safety Vest"
]

PPE_MISSING_CLASSES = [
    "NO-Gloves",
    "NO-Goggles",
    "NO-Hardhat",
    "NO-Mask",
    "NO-Safety Vest"
]

ENV_CLASSES = [
    "Fall-Detected",
    "Ladder",
    "Safety Cone"
]

CLASS_NAMES = [
    'Fall-Detected', 'Gloves', 'Goggles', 'Hardhat', 'Ladder',
    'Mask', 'NO-Gloves', 'NO-Goggles', 'NO-Hardhat',
    'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest'
]

TEST_IMAGE_PATH = "data/test/images"
TEST_LABEL_PATH = "data/test/labels"

MODEL_PATH = "code/models/best.pt"

"""
Due to the limited GPU resources and large number of test images
The responses were done in batches of 100
With a total of 45 test response files
"""
SPLIT_LENGTH = 100
TEST_Files = 45

def generate_evidence(images: list) -> dict:
    """Generate evidence for each image with PPE detection results.

    Args:
        images (list): List of image paths.

    Returns:
        dict: A dictionary mapping image paths to their detection results.
    """
    results = ppe_pred(MODEL_PATH, images)

    gc.collect()
    torch.cuda.empty_cache()

    evidences = {}

    for img, result in results.items():
        violations = result["violations"]
        detected_ppe = result["compliant_items"]
        missing_ppe = result["missing_ppe"]
        fall_detected = result["fall_detected"]

        evidence = {
            "required_ppe":    ["Hardhat", "Gloves", "Goggles", "Mask", "Safety Vest"],
            "detected_ppe":    detected_ppe,
            "missing_ppe":     missing_ppe,
            "violations":      violations,
            "fall_detected":   fall_detected,
        }

        evidences[img] = evidence

    return evidences

def load_gt(label_path: list) -> dict:
    """Load the Ground Truth from the labels

    Args:
        label_path (list): path to the label files

    Returns:
        dict: A dictionary mapping image paths to their ground truth labels.
    """
    gt = {}

    for path in label_path:
        labels = []
        with open(path, 'r') as f:
            for line in f:
                cls_id = int(line.split()[0])
                label = CLASS_NAMES[cls_id]
                labels.append(label)

            missing = set()
            for label in labels:
                if label.startswith("NO-"):
                    missing.add(label.replace("NO-", "").lower())

            gt[path] = {"detected": labels,
                        "missing": sorted(list(missing))}

    return gt

def parse_florence_pred(response: dict) -> dict:
    """Parses the Florence response to convert it to the needed format

    Args:
        response (dict): A dictionary containing the Florence response for a single image.

    Returns:
        dict: A dictionary containing the parsed results.
    """
    missing_raw = response.get("missing_ppe", "")
    
    missing_items = []
    if missing_raw and missing_raw.lower() != "none":
        missing_items = [
            item.strip().lower()
            for item in missing_raw.split(",")
            if item.strip()
        ]

    is_compliant = response.get("verdict", "").lower() == "compliant"

    return {
        "compliance":    "compliant" if is_compliant else "non-compliant",
        "missing_items": list(set(missing_items))
    }

def single_metric(pred: dict, gt: dict) -> dict:
    """The metrics for a single response

    Args:
        pred (dict): The predicted compliance and missing items from Florence response.
        gt (dict): The ground truth compliance and missing items from the label.

    Returns:
        dict: A dictionary containing the calculated metrics.
    """
    accuracy = 1 if gt["compliance"] == pred["compliance"] else 0

    gt_set   = set(gt["missing_items"])
    pred_set = set(pred["missing_items"])

    tp = len(gt_set & pred_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0)

    hallucination = fp  # items predicted present but not in ground truth

    return {
        "accuracy":      accuracy,
        "precision":     precision,
        "recall":        recall,
        "f1":            f1,
        "hallucination": hallucination,
        "tp":            tp,
        "fp":            fp,
        "fn":            fn,
    }

def parse_gt(gt_result: dict) -> dict:
    """Parses the ground truth is fill in the missing items and compliance status

    Args:
        gt_result (dict): A dictionary containing the ground truth detected and missing items.

    Returns:
        dict: A dictionary containing the parsed ground truth with compliance status and missing items.
    """
    missing_items = gt_result.get("missing", [])
    
    return {
        "compliance":    "non-compliant" if missing_items else "compliant",
        "missing_items": [item.lower() for item in missing_items]
    }

def metrics(pred: dict, gt: dict) -> dict:
    """The full metrics function that finds the average of the metrics

    Args:
        pred (dict): A dictionary containing the predicted compliance and missing items from Florence responses.
        gt (dict): A dictionary containing the ground truth compliance and missing items from the labels.

    Returns:
        dict: A dictionary containing the average metrics.
    """
    all_metrics   = []
    unmatched     = []

    for img_path, florence_result in pred.items():

        img_basename = os.path.basename(img_path)
        label_name   = os.path.splitext(img_basename)[0] + ".txt"

        gt_key = None
        for key in gt:
            if os.path.basename(key) == label_name:
                gt_key = key
                break

        if gt_key is None:
            unmatched.append(img_path)
            continue

        pred       = parse_florence_pred(florence_result)
        gt_parsed  = parse_gt(gt[gt_key])
        m          = single_metric(gt_parsed, pred)
        all_metrics.append(m)

    if not all_metrics:
        print("WARNING: No matched images found between responses and ground truth.")
        return {}

    n = len(all_metrics)

    avg = {
        "total_images": n,
        "matched_images": len(unmatched),
        "accuracy":      sum(m["accuracy"] for m in all_metrics) / n,
        "precision":     sum(m["precision"] for m in all_metrics) / n,
        "recall":        sum(m["recall"] for m in all_metrics) / n,
        "f1":            sum(m["f1"] for m in all_metrics) / n,
        "hallucination": sum(m["hallucination"] for m in all_metrics) / n,
    }

    return avg

def main():
    if not METRIC:
        gc.collect()
        torch.cuda.empty_cache()

        print("Loading ground truth for test images...")
        label_paths = [os.path.join(TEST_LABEL_PATH, img.replace(".jpg", ".txt")) for img in os.listdir(TEST_IMAGE_PATH)]
        gt = load_gt(label_paths)
            
        with open("resources/test_res/gt.json", "w") as f:
                    json.dump(gt, f, indent=4)

        print("Generating evidence and Florence responses for test images...")

        img_paths = [os.path.join(TEST_IMAGE_PATH, img) for img in os.listdir(TEST_IMAGE_PATH)]

        total_img = len(img_paths)

        for i in range(0, total_img, SPLIT_LENGTH):

            print(f"Processing images {i} to {min(i+SPLIT_LENGTH, total_img)}...")

            split_img_paths = img_paths[i:i+SPLIT_LENGTH]
            

            print(f"Generating evidence for images {i} to {min(i+SPLIT_LENGTH, total_img)}...")
            evidence = generate_evidence(split_img_paths)

            print(f"Generating Florence responses for images {i} to {min(i+SPLIT_LENGTH, total_img)}...")
            gc.collect()
            torch.cuda.empty_cache()

            florence_responses = {}
            for img, ev in evidence.items():
                florence_response = generate_response(img, ev)
                florence_responses[img] = florence_response
                
            with open(f"resources/test_res/florence_responses-{i}.json", "w") as f:
                json.dump(florence_responses, f, indent=4)

            print(f"Completed processing images {i} to {min(i+SPLIT_LENGTH, total_img)}.")

            gc.collect()
            torch.cuda.empty_cache()
            
            sleep(5)  # Sleep for a short time to avoid overwhelming the system
    
    gt = load_gt([os.path.join(TEST_LABEL_PATH, img.replace(".jpg", ".txt")) for img in os.listdir(TEST_IMAGE_PATH)])
    

    with open("resources/test_res/total_responses.json", "r") as f:
        total_responeses = json.load(f)

    print(f"""
        Total images processed: {len(total_responeses)}
        Total ground truth entries: {len(gt)}

        calculating metrics...

        """)

    metric = metrics(total_responeses, gt)

    with open("resources/test_res/metrics.json", "w") as f:
        json.dump(metric, f, indent=4)

    

if __name__ == "__main__":

    if DEBUG:
        image_path = [f"{TEST_IMAGE_PATH}/-1x-1_jpg.rf.282bd04459c7ab3088358b5ac6ebeea7.jpg"]

        label_path = [f"{TEST_LABEL_PATH}/-1x-1_jpg.rf.282bd04459c7ab3088358b5ac6ebeea7.txt"]

        evidence = generate_evidence(image_path)

        for img, ev in evidence.items():
            
            florence_response = generate_response(img, ev)

        gt = load_gt(label_path)

        print(gt)

        with open("resources/test_res/test.json", "w") as f:
            json.dump(florence_response, f, indent=4)
    
    else:
        main()
