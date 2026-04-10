from transformers import LlavaProcessor, LlavaForConditionalGeneration
from PIL import Image
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model_id = "llava-hf/llava-1.5-7b-hf"

processor = LlavaProcessor.from_pretrained(model_id)

vlm_model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16
).to(device)

image_path = "/mnt/datassd3/thisali/ppe_dataset/test/images/-1x-1_jpg.rf.282bd04459c7ab3088358b5ac6ebeea7.jpg"
image = Image.open(image_path).convert("RGB")

prompt = """
<image>
You are a workplace safety inspector.

Detected PPE: Hardhat
Missing PPE: Gloves
Environment indicators: Ladder

Workers must wear Hardhat, Gloves, Goggles, Mask, and Safety Vest.

Is the worker compliant with PPE safety rules?
Explain your reasoning.
"""


inputs = processor(
    text=prompt,
    images=image,
    return_tensors="pt"
).to(device)

output = vlm_model.generate(
    **inputs,
    max_new_tokens=120
)

answer = processor.decode(output[0], skip_special_tokens=True)

print("VLM Response:")
print(answer)
