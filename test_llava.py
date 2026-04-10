from transformers import LlavaForConditionalGeneration, AutoProcessor
from PIL import Image
import torch
import time

# -------------------------------
# Model
# -------------------------------
model_id = "llava-hf/llava-1.5-7b-hf"

print("Loading model...")

model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
).to("cuda")

print("Model loaded")

# -------------------------------
# Processor
# -------------------------------
processor = AutoProcessor.from_pretrained(model_id)

print("Processor loaded")

# -------------------------------
# Load Image
# -------------------------------
image_path = "/mnt/datassd3/thisali/ppe_dataset/runs/detect/predict/-1x-1_jpg.rf.282bd04459c7ab3088358b5ac6ebeea7.jpg"

image = Image.open(image_path).convert("RGB")

print("Image loaded")

# -------------------------------
# Prompt
# -------------------------------
prompt = """USER: <image>
You are a workplace safety inspector.

Answer ONLY based on what you see in the image.

Questions:
1. What PPE items are visible?
2. What PPE items are missing?
3. Is the worker compliant with PPE safety rules?

ASSISTANT:"""

# -------------------------------
# Prepare Inputs
# -------------------------------
inputs = processor(
    text=prompt,
    images=image,
    return_tensors="pt"
)

inputs = {k: v.to("cuda") for k, v in inputs.items()}

print("Inputs prepared")

# -------------------------------
# Inference
# -------------------------------
print("Starting inference...")

start_time = time.time()

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=120,
        do_sample=False
    )

torch.cuda.synchronize()

end_time = time.time()

print("Generation finished")
print(f"Inference time: {end_time - start_time:.2f} seconds")

# -------------------------------
# Decode Output
# -------------------------------
response = processor.decode(output[0], skip_special_tokens=True)

print("\n===== LLaVA RESPONSE =====")
print(response)
print("==========================")
