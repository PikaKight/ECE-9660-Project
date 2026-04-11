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

def generate_response(image_path, prompt):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=512)
    
    response = processor.decode(outputs[0], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    image_path = "resources/test/images/-1x-1_jpg.rf.282bd04459c7ab3088358b5ac6ebeea7.jpg"
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
    
    response = generate_response(image_path, prompt)
    print("Florence Response:")
    print(response)



