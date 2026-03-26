from ultralytics import YOLO
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import torch
import time
import os
import re

# -----------------------
# CONFIG
# -----------------------
image_path = '/mnt/datassd3/thisali/ppe_dataset/test/images/-88-_png_jpg.rf.e94d9b00557acaca46256628591e2090.jpg'

# corresponding label file (YOLO format)
label_path = image_path.replace("/images/", "/labels/").replace(".jpg", ".txt")

CLASS_NAMES = [
    'Fall-Detected', 'Gloves', 'Goggles', 'Hardhat', 'Ladder',
    'Mask', 'NO-Gloves', 'NO-Goggles', 'NO-Hardhat',
    'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest'
]

# -----------------------
# Load YOLO
# -----------------------
print("Loading YOLO...")
yolo = YOLO('/mnt/datassd3/thisali/ppe_dataset/runs/detect/train3/weights/best.pt')

# -----------------------
# Load BLIP
# -----------------------
print("Loading BLIP-2...")
model_id = "Qwen/Qwen2-VL-2B-Instruct"

processor = AutoProcessor.from_pretrained(model_id)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16
).to("cuda")

print("BLIP ready")

# -----------------------
# Load image
# -----------------------
image = Image.open(image_path).convert("RGB")

# -----------------------
# YOLO detection
# -----------------------
results = yolo(image_path)

detections = []

for r in results:
    for box in r.boxes:
        cls = int(box.cls)
        label = yolo.names[cls]
        detections.append(label)

detections_text = ", ".join(set(detections))
print("YOLO detections:", detections_text)

# -----------------------
# PROMPT (IMPROVED)
# -----------------------
prompt = f"""
USER: <image>
You are a strict PPE safety inspector.

Detected objects:
{detections_text}

Rules:
- Only use detected objects
- Do NOT assume anything not listed
- If unsure, say "unknown"

Answer EXACTLY in this format:

Compliance: <compliant/non-compliant>
Missing: <comma-separated items or none>
Explanation: <short reason>

ASSISTANT:
"""

# -----------------------
# Run Qwen2-VL
# -----------------------
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {
                "type": "text",
                "text": f"""
You are a strict PPE safety inspector.

Detected objects:
{detections_text}

Rules:
- Only use detected objects
- Do NOT assume anything not listed
- If unsure, say "unknown"

Answer EXACTLY in this format:
Compliance: <compliant/non-compliant>
Missing: <comma-separated items or none>
Explanation: <short reason>
"""
            },
        ],
    }
]

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

inputs = {k: v.to("cuda") for k, v in inputs.items()}

print("Running Qwen2-VL...")
start = time.time()

with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False
    )

torch.cuda.synchronize()
print("Time:", time.time() - start)

generated_ids_trimmed = [
    out_ids[len(in_ids):]
    for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
]

response = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)[0]

print("\n===== QWEN RESPONSE =====")
print(response)
print("=========================")

# -----------------------
# PARSE BLIP OUTPUT
# -----------------------
def parse_BLIP_output(text):
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

    valid_items = {"gloves", "goggles", "hardhat", "mask", "safety vest"}

    if miss_line is not None:
        miss_value = miss_line.split(":", 1)[1].strip().lower()
        if miss_value != "none":
            parts = [p.strip() for p in miss_value.split(",")]
            for p in parts:
                if p in valid_items:
                    missing_items.append(p)

    return {
        "compliance": compliance,
        "missing_items": list(set(missing_items))
    }

pred = parse_BLIP_output(response)

print("\nParsed Prediction:", pred)

# -----------------------
# LOAD GROUND TRUTH
# -----------------------
def load_gt_labels(label_path):
    labels = []

    if not os.path.exists(label_path):
        print("WARNING: Label file not found!")
        return labels
    print("\n===== RAW LABEL FILE =====")
    with open(label_path, 'r') as f:
        for line in f.readlines():
            print(line.strip())
            cls_id = int(line.split()[0])
            labels.append(CLASS_NAMES[cls_id])
    print("==========================")
    return labels

def generate_gt(labels):
    missing = set()   # ← use set instead of list

    for l in labels:
        if l.startswith("NO-"):
            missing.add(l.replace("NO-", "").lower())

    compliance = "non-compliant" if missing else "compliant"

    return {
        "compliance": compliance,
        "missing_items": list(missing)
    }

gt_labels = load_gt_labels(label_path)
gt = generate_gt(gt_labels)

print("Ground Truth:", gt)

# -----------------------
# METRICS
# -----------------------

# Accuracy
accuracy = 1 if gt["compliance"] == pred["compliance"] else 0

# Missing PPE metrics
gt_set = set(gt["missing_items"])
pred_set = set(pred["missing_items"])

tp = len(gt_set & pred_set)
fp = len(pred_set - gt_set)
fn = len(gt_set - pred_set)

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

# Hallucination
hallucination = len(pred_set - gt_set)

# -----------------------
# PRINT RESULTS
# -----------------------
print("\n===== METRICS =====")
print("Compliance Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Hallucinated Items:", hallucination)
print("===================")