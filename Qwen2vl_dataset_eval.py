import os
import csv
import time
from pathlib import Path

import torch
from ultralytics import YOLO
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


# -----------------------
# CONFIG
# -----------------------
IMAGE_DIR = "/mnt/datassd3/thisali/ppe_dataset/test/images"
LABEL_DIR = "/mnt/datassd3/thisali/ppe_dataset/test/labels"
YOLO_WEIGHTS = "/mnt/datassd3/thisali/ppe_dataset/runs/detect/train3/weights/best.pt"
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
OUTPUT_CSV = "yolo_qwen_eval_results.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = [
    "Fall-Detected", "Gloves", "Goggles", "Hardhat", "Ladder",
    "Mask", "NO-Gloves", "NO-Goggles", "NO-Hardhat",
    "NO-Mask", "NO-Safety Vest", "Person", "Safety Cone", "Safety Vest"
]

VALID_MISSING_ITEMS = {"gloves", "goggles", "hardhat", "mask", "safety vest"}

CONFLICT_PAIRS = [
    ("Gloves", "NO-Gloves"),
    ("Goggles", "NO-Goggles"),
    ("Hardhat", "NO-Hardhat"),
    ("Mask", "NO-Mask"),
    ("Safety Vest", "NO-Safety Vest"),
]


# -----------------------
# HELPERS
# -----------------------
def get_label_path(image_path: Path) -> Path:
    return Path(str(image_path).replace("/images/", "/labels/")).with_suffix(".txt")


def load_gt_labels(label_path: Path) -> list[str]:
    labels = []
    if not label_path.exists():
        return labels

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            cls_id = int(parts[0])
            if 0 <= cls_id < len(CLASS_NAMES):
                labels.append(CLASS_NAMES[cls_id])
    return labels


def generate_gt(labels: list[str]) -> dict:
    missing = set()

    for label in labels:
        if label == "NO-Gloves":
            missing.add("gloves")
        elif label == "NO-Goggles":
            missing.add("goggles")
        elif label == "NO-Hardhat":
            missing.add("hardhat")
        elif label == "NO-Mask":
            missing.add("mask")
        elif label == "NO-Safety Vest":
            missing.add("safety vest")

    compliance = "non-compliant" if missing else "compliant"
    return {
        "compliance": compliance,
        "missing_items": sorted(list(missing))
    }


def resolve_conflicts(detections: list[str]) -> tuple[list[str], bool]:
    det_set = set(detections)
    clean = set(det_set)
    ambiguous = False

    for pos, neg in CONFLICT_PAIRS:
        if pos in det_set and neg in det_set:
            clean.discard(pos)
            clean.discard(neg)
            ambiguous = True

    return sorted(list(clean)), ambiguous


def build_qwen_messages(image_path: str, detections_text: str):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {
                    "type": "text",
                    "text": f"""You are a PPE safety inspector.

Detected objects:
{detections_text}

Use only the detected objects above.

Answer using exactly these two lines and nothing else:
Compliance: compliant OR non-compliant
Missing: none OR comma-separated items"""
                },
            ],
        }
    ]


def parse_model_output(text: str) -> dict:
    compliance = "unknown"
    missing_items = []

    lines = [line.strip() for line in text.splitlines() if line.strip()]

    comp_line = None
    miss_line = None

    for line in lines:
        low = line.lower()
        if low.startswith("compliance:"):
            comp_line = line
        elif low.startswith("missing:"):
            miss_line = line

    if comp_line is not None:
        comp_value = comp_line.split(":", 1)[1].strip().lower()
        if "non-compliant" in comp_value:
            compliance = "non-compliant"
        elif "compliant" in comp_value:
            compliance = "compliant"

    if miss_line is not None:
        miss_value = miss_line.split(":", 1)[1].strip().lower()
        if miss_value != "none":
            parts = [p.strip() for p in miss_value.split(",")]
            for p in parts:
                if p in VALID_MISSING_ITEMS:
                    missing_items.append(p)

    return {
        "compliance": compliance,
        "missing_items": sorted(list(set(missing_items)))
    }


def run_qwen(image_path: str, detections_text: str, processor, model) -> str:
    messages = build_qwen_messages(image_path, detections_text)

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]

    response = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    return response.strip()


# -----------------------
# LOAD MODELS
# -----------------------
print("Loading YOLO...")
yolo = YOLO(YOLO_WEIGHTS)

print("Loading Qwen2-VL...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto" if DEVICE == "cuda" else None
)

if DEVICE == "cpu":
    model = model.to("cpu")

print("Models ready")


# -----------------------
# DATASET
# -----------------------
image_paths = sorted(
    [p for p in Path(IMAGE_DIR).glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
)

# Optional: test on subset first
# image_paths = image_paths[:100]

print(f"Found {len(image_paths)} images")


# -----------------------
# METRIC ACCUMULATORS
# -----------------------
total_images = 0
used_images = 0
skipped_ambiguous = 0
skipped_unparsed = 0

compliance_correct = 0

tp_total = 0
fp_total = 0
fn_total = 0

hallucinated_items_total = 0
predicted_missing_total = 0

rows = []


# -----------------------
# MAIN LOOP
# -----------------------
start_all = time.time()

for idx, image_path in enumerate(image_paths, start=1):
    total_images += 1
    label_path = get_label_path(image_path)

    gt_labels = load_gt_labels(label_path)
    gt = generate_gt(gt_labels)

    results = yolo(str(image_path), verbose=False)

    detections = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls)
            detections.append(yolo.names[cls])

    detections, ambiguous = resolve_conflicts(detections)

    if ambiguous:
        skipped_ambiguous += 1
        rows.append({
            "image": image_path.name,
            "status": "skipped_ambiguous_yolo",
            "yolo_detections": ", ".join(detections),
            "gt_compliance": gt["compliance"],
            "gt_missing": ", ".join(gt["missing_items"]),
            "pred_compliance": "",
            "pred_missing": "",
            "compliance_correct": "",
            "tp": "",
            "fp": "",
            "fn": "",
            "hallucinated_items": ""
        })
        continue

    detections_text = ", ".join(sorted(set(detections))) if detections else "none"

    response = run_qwen(str(image_path), detections_text, processor, model)
    pred = parse_model_output(response)

    if pred["compliance"] == "unknown":
        skipped_unparsed += 1
        rows.append({
            "image": image_path.name,
            "status": "skipped_unparsed_qwen",
            "yolo_detections": detections_text,
            "gt_compliance": gt["compliance"],
            "gt_missing": ", ".join(gt["missing_items"]),
            "pred_compliance": pred["compliance"],
            "pred_missing": ", ".join(pred["missing_items"]),
            "compliance_correct": "",
            "tp": "",
            "fp": "",
            "fn": "",
            "hallucinated_items": ""
        })
        continue

    used_images += 1

    comp_ok = int(pred["compliance"] == gt["compliance"])
    compliance_correct += comp_ok

    gt_set = set(gt["missing_items"])
    pred_set = set(pred["missing_items"])

    tp = len(gt_set & pred_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)

    hallucinated_items = len(pred_set - gt_set)

    tp_total += tp
    fp_total += fp
    fn_total += fn

    hallucinated_items_total += hallucinated_items
    predicted_missing_total += len(pred_set)

    rows.append({
        "image": image_path.name,
        "status": "used",
        "yolo_detections": detections_text,
        "gt_compliance": gt["compliance"],
        "gt_missing": ", ".join(gt["missing_items"]),
        "pred_compliance": pred["compliance"],
        "pred_missing": ", ".join(pred["missing_items"]),
        "compliance_correct": comp_ok,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "hallucinated_items": hallucinated_items
    })

    if idx % 25 == 0 or idx == len(image_paths):
        print(f"Processed {idx}/{len(image_paths)} images")


# -----------------------
# FINAL METRICS
# -----------------------
compliance_accuracy = compliance_correct / used_images if used_images > 0 else 0.0
precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
hallucination_rate = hallucinated_items_total / predicted_missing_total if predicted_missing_total > 0 else 0.0

elapsed = time.time() - start_all

print("\n===== DATASET RESULTS =====")
print(f"Total images: {total_images}")
print(f"Used images: {used_images}")
print(f"Skipped ambiguous YOLO: {skipped_ambiguous}")
print(f"Skipped unparsable Qwen: {skipped_unparsed}")
print(f"Compliance Accuracy: {compliance_accuracy:.4f}")
print(f"Missing PPE Precision: {precision:.4f}")
print(f"Missing PPE Recall: {recall:.4f}")
print(f"Missing PPE F1: {f1:.4f}")
print(f"Total Hallucinated Items: {hallucinated_items_total}")
print(f"Hallucination Rate: {hallucination_rate:.4f}")
print(f"Total time: {elapsed:.2f} sec")
print("===========================")


# -----------------------
# SAVE CSV
# -----------------------
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "image", "status", "yolo_detections",
            "gt_compliance", "gt_missing",
            "pred_compliance", "pred_missing",
            "compliance_correct", "tp", "fp", "fn",
            "hallucinated_items"
        ]
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved per-image results to: {OUTPUT_CSV}")