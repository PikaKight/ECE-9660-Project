#!/usr/bin/env python
import os

from ultralytics import YOLO, settings

CLASS_NAMES = [
    'Fall-Detected',    # 0
    'Gloves',           # 1
    'Goggles',          # 2
    'Hardhat',          # 3
    'Ladder',           # 4
    'Mask',             # 5
    'NO-Gloves',        # 6
    'NO-Goggles',       # 7
    'NO-Hardhat',       # 8
    'NO-Mask',          # 9
    'NO-Safety Vest',   # 10
    'Person',           # 11
    'Safety Cone',      # 12
    'Safety Vest'       # 13
]

VIOLATION_CLASSES  = {'Fall-Detected', 'NO-Gloves', 'NO-Goggles',
                      'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest'}
COMPLIANT_CLASSES  = {'Gloves', 'Goggles', 'Hardhat', 'Mask', 'Safety Vest'}
REQUIRED_PPE       = {'Hardhat', 'Safety Vest', 'Mask', 'Gloves', 'Goggles'}
NEUTRAL_CLASSES    = {'Person', 'Ladder', 'Safety Cone'}


def setup():
    cwd = os.getcwd()

    data_dir = settings.get("datasets_dir")

    if cwd is data_dir:
        print("same")
        return

    settings.update({"datasets_dir": cwd})

def ppe_model(yaml_path: str, save_path: str):

    model = YOLO("yolo11n.pt", task="detect")

    model.train(
        data=yaml_path,
        epochs=50,
        imgsz=640,
        workers=0,
        batch=8,
        patience=15,
        optimizer="AdamW",
        name="ppe",
        )

    model.save(save_path)

def ppe_metrics(model_path, yaml_path):
    model = YOLO(model_path)
    
    metrics = model.val(data=yaml_path, batch=16, conf=0.2, iou=0.5)

    avg_precision = metrics.box.mp

    avg_recall = metrics.box.mr

    f1 = 2 * ((avg_precision * avg_recall) / (avg_precision + avg_recall))

    print(f"""
    
        Avg Precision: {avg_precision}

        Avg Recall: {avg_recall}

        Avg F1: {f1}
          """)


def ppe_pred(model_path: str, tests: list) -> dict:
    """_summary_

    Args:
        model_path (str): Path to the trained YOLO model.
        tests (list): A list of paths to the test images.

    Returns:
        dict: A dictionary containing the prediction results for each test image.
    """
    model = YOLO(model_path)

    tests = [os.path.abspath(t) for t in tests]

    pred = model(tests, conf=0.25)

    results = {}
    
    for img, org_path in zip(pred, tests):

        img_name = os.path.basename(org_path).split(".")[0]
        img.save_txt(f"resources/files/{img_name}.txt")

        detections = []

        for box in img.boxes:
            res = box.xyxy[0].tolist()
            cls_id = int(box.cls[0].tolist())
            cls_name = CLASS_NAMES[cls_id]
            confidence = round(box.conf[0].tolist(), 3)

            if cls_name in VIOLATION_CLASSES:
                role = "violation"
            elif cls_name in COMPLIANT_CLASSES:
                role = "compliant_item"
            else:
                role = "neutral"

            detections.append({
                "class":      cls_name,
                "role":       role,
                "confidence": confidence,
                "bbox":       res,
            })

        detected_classes = {d["class"] for d in detections}
        violations = [d for d in detections if d["role"] == "violation"]
        compliant_items = [d["class"] for d in detections if d["role"] == "compliant_item"]

        # Determine which required PPE is unaccounted for entirely
        accounted = set()
        for cls in detected_classes:
            base = cls.replace("NO-", "")
            accounted.add(base)
        missing_ppe = list(REQUIRED_PPE - accounted)

        # Overall YOLO-level compliance verdict
        yolo_compliant = len(violations) == 0 and len(missing_ppe) == 0

        results[org_path] = {
            "detections":      detections,
            "violations":      [v["class"] for v in violations],
            "violation_detail": violations,          # includes bbox + confidence
            "compliant_items": compliant_items,
            "missing_ppe":     missing_ppe,
            "person_detected": "Person" in detected_classes,
            "fall_detected":   "Fall-Detected" in detected_classes,
            "yolo_compliant":  yolo_compliant,
            "n_violations":    len(violations),
            "n_compliant":     len(compliant_items),
            "n_missing":       len(missing_ppe),
        }

    return results

    
if __name__ == "__main__":

    import json

    # path = "data/data.yaml"
    
    # setup()

    model_path = "code/models/best.pt"

    # if not os.path.exists(model_path):
    #     ppe_model(path, model_path)

    #     ppe_metrics(model_path, path)

    test = "data/test/images/"

    tests = [os.path.join(test, f) for f in os.listdir(test)]

    print(tests[:5])

    res = ppe_pred(model_path, tests[:5])

    with open("resources/test_res/test.json", 'w') as f:
        json.dump(res, f, indent=4)
