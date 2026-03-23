import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM 

model_id = "microsoft/Florence-2-large"
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if device == "cuda" else torch.float32

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch_dtype,
    trust_remote_code=True
    ).to(device)

processor = AutoProcessor.from_pretrained(
    model_id,
    trust_remote_code=True
    )

def generate_response(image_path, question):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(
        text=question,
        images=image,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=512)
    
    response = processor.decode(outputs[0], skip_special_tokens=True)
    return response



