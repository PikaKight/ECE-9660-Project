import os
import csv
import torch
from PIL import Image
from ultralytics import YOLO
from transformers import LlavaProcessor, LlavaForConditionalGeneration

# =========================
# CONFIG
# =========================
YOLO_MODEL_PATH = "/mnt/datassd3/thisali/ppe_dataset/runs/detect/train3/weights/best.pt"
IMAGE_FOLDER = "/mnt/datassd3/thisali/ppe_dataset/test/images"
OUTPUT_CSV = "/mnt/datassd3/thisali/ppe_dataset/ppe_vlm_results.csv"

LLAVA_MODEL_ID = "llava-hf/llava-1.5-7b-hf"
CONF_THRESHOLD = 0.25
MAX_NEW_TOKENS = 120
USE_HALF = torch.cuda.is_available()

# =========================
# CLASS GROUPS
# =========================
PPE_PRESENT_CLASSES = {
    "Gloves",
    "Goggles",
    "Hardhat",
    "Mask",
    "Safety Vest",
}

PPE_MISSING_MAP = {
    "NO-Gloves": "Gloves",
    "NO-Goggles": "Goggles",
    "NO-Hardhat": "Hardhat",
    "NO-Mask": "Mask",
    "NO-Safety Vest": "Safety Vest",
}

ENVIRONMENT_CLASSES = {
    "Fall-Detected": "Fall-Detected",
    "Ladder": "Ladder",
    "Safety Cone": "Safety Cone",
}

# =========================
# HELPERS
# =========================
def unique_keep_order(items):
    seen = set()
    output = []
    for item in items:
        if item not in seen:
            seen.add(item)
            output.append(item)
    return output

def extract_yolo_evidence(results, model_names):
    detections = []

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls.item()) if hasattr(box.cls, "item") else int(box.cls)
            conf = float(box.conf.item()) if hasattr(box.conf, "item") else float(box.conf)
            label = model_names[cls_id]
            detections.append((label, conf))

    detected_present = []
    detected_missing = []
    environment_present = []

    for label, _ in detections:
        if label in PPE_PRESENT_CLASSES:
            detected_present.append(label)

        if label in PPE_MISSING_MAP:
            detected_missing.append(PPE_MISSING_MAP[label])

        if label in ENVIRONMENT_CLASSES:
            environment_present.append(ENVIRONMENT_CLASSES[label])

    detected_present = unique_keep_order(detected_present)
    detected_missing = unique_keep_order(detected_missing)
    environment_present = unique_keep_order(environment_present)

    return detections, detected_present, detected_missing, environment_present

def build_prompt(detected_present, detected_missing, environment_present):
    detected_text = ", ".join(detected_present) if detected_present else "None"
    missing_text = ", ".join(detected_missing) if detected_missing else "None"
    environment_text = ", ".join(environment_present) if environment_present else "None"

    prompt = f"""<image>
You are a workplace safety inspector.

Use ONLY the information provided below.
Do NOT assume any extra missing PPE beyond the listed evidence.

Detected PPE:
{detected_text}

Missing PPE:
{missing_text}

Environment indicators:
{environment_text}

Workers should wear:
Hardhat, Gloves, Goggles, Mask, and Safety Vest.

Answer these questions briefly:
1. Is the worker compliant with PPE safety rules?
2. What PPE is missing?
3. Are Fall-Detected, Ladder, or Safety Cone present?

Base your answer only on the listed evidence.
"""
    return prompt

def run_llava(processor, vlm_model, device, image_path, prompt):
    image = Image.open(image_path).convert("RGB")

    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt"
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = vlm_model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False
        )

    answer = processor.decode(output[0], skip_special_tokens=True)
    return answer

# =========================
# LOAD MODELS
# =========================
print("Loading YOLO...")
yolo_model = YOLO(YOLO_MODEL_PATH)

print("Loading LLaVA...")
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if USE_HALF else torch.float32

processor = LlavaProcessor.from_pretrained(LLAVA_MODEL_ID, use_fast=False)
vlm_model = LlavaForConditionalGeneration.from_pretrained(
    LLAVA_MODEL_ID,
    torch_dtype=dtype
).to(device)
vlm_model.eval()

# =========================
# PROCESS ALL IMAGES
# =========================
image_files = sorted(
    [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))]
)

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "image_name",
        "all_yolo_detections",
        "detected_ppe",
        "missing_ppe",
        "environment_indicators",
        "prompt",
        "vlm_response"
    ])

    for idx, image_name in enumerate(image_files, start=1):
        #image_path = os.path.join(IMAGE_FOLDER, image_name)
        image_path="/mnt/datassd3/thisali/ppe_dataset/test/images/-1x-1_jpg.rf.282bd04459c7ab3088358b5ac6ebeea7.jpg"
        print(f"[{idx}/{len(image_files)}] Processing: {image_name}")

        # Step 1: YOLO inference
        results = yolo_model(image_path, conf=CONF_THRESHOLD, verbose=False)

        # Step 2: Convert YOLO output to evidence
        detections, detected_present, detected_missing, environment_present = extract_yolo_evidence(
            results, yolo_model.names
        )

        # Step 3: Build dynamic prompt from YOLO results
        prompt = build_prompt(detected_present, detected_missing, environment_present)

        # Step 4: VLM reasoning
        try:
            vlm_response = run_llava(processor, vlm_model, device, image_path, prompt)
        except Exception as e:
            vlm_response = f"ERROR: {str(e)}"

        # Save raw detections as label(conf)
        detections_text = "; ".join([f"{label}({conf:.3f})" for label, conf in detections]) if detections else "None"

        writer.writerow([
            image_name,
            detections_text,
            ", ".join(detected_present) if detected_present else "None",
            ", ".join(detected_missing) if detected_missing else "None",
            ", ".join(environment_present) if environment_present else "None",
            prompt.replace("\n", " ").strip(),
            vlm_response.replace("\n", " ").strip()
        ])

print(f"\nDone. Results saved to:\n{OUTPUT_CSV}")
