from ultralytics import YOLO
from transformers import LlavaForConditionalGeneration, AutoProcessor
from PIL import Image
import torch
import time

# -----------------------
# Load YOLO
# -----------------------
print("Loading YOLO...")
yolo = YOLO('/mnt/datassd3/thisali/ppe_dataset/runs/detect/train3/weights/best.pt')   # your PPE model

# -----------------------
# Load LLaVA
# -----------------------
print("Loading LLaVA...")
model_id = "llava-hf/llava-1.5-7b-hf"

model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16
).to("cuda")

processor = AutoProcessor.from_pretrained(model_id)

print("Models ready")

# -----------------------
# Load image
# -----------------------
image_path = '/mnt/datassd3/thisali/ppe_dataset/train/images/YouTube-FreeStockFootage-PersonThinkingDeeply-h-HC-hj-Zo-720p_mp4-25_jpg.rf.aea03c53f339861749419bf2d4c8106a.jpg'
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
# Chatbot prompt
# -----------------------
prompt = f"""
USER: <image>
You are a workplace safety inspector chatbot.

YOLO detected the following objects:
{detections_text}

Answer the user's question conversationally.

Question: Workers should wear:
Hardhat, Gloves, Goggles, Mask, and Safety Vest.

Answer these questions briefly:
1. Is the worker compliant with PPE safety rules and why?
2. What PPE is missing?
3. Are Fall-Detected, Ladder, or Safety Cone present?

Base your answer only on the listed evidence.

ASSISTANT:
"""

# -----------------------
# Prepare inputs
# -----------------------
inputs = processor(
    text=prompt,
    images=image,
    return_tensors="pt"
)

inputs = {k: v.to("cuda") for k, v in inputs.items()}

# -----------------------
# Inference
# -----------------------
print("Running LLaVA chatbot...")

start = time.time()

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False
    )

torch.cuda.synchronize()

print("Time:", time.time() - start)

response = processor.decode(output[0], skip_special_tokens=True)

print("\n===== CHATBOT RESPONSE =====")
print(response)
print("============================")
