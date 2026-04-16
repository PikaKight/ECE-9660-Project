import gc

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM 

PPE_VISUAL_TERMS = {
    "Hardhat":      "hard hat",
    "Safety Vest":  "high visibility vest",
    "Gloves":       "work gloves",
    "Goggles":      "safety goggles",
    "Mask":         "face mask",
}

model_id = "microsoft/Florence-2-large"
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if device == "cuda" else torch.float32

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch_dtype,
    trust_remote_code=True,
    ).to(device)
model.eval()

processor = AutoProcessor.from_pretrained(
    model_id,
    trust_remote_code=True
    )

def run_task(image: Image, prompt: str, input: str="") -> str: 
    
    full_prompt = prompt if not input else f"{prompt}{input}"

    inputs = processor(
        text=full_prompt,
        images=image,
        return_tensors="pt"
    )

    inputs = {
        k: v.to(device, torch_dtype) if v.dtype == torch.float32 else v.to(device)
        for k, v in inputs.items()
    }

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            num_beams=1,
            early_stopping=True
            )
    
    response = processor.decode(outputs[0], skip_special_tokens=True)

    if response.startswith(prompt):
        response = response[len(prompt):].strip()

    del inputs, outputs
    gc.collect()
    torch.cuda.empty_cache()

    return response

def generate_response(image_path, evidence: dict) -> dict:
    
    image = Image.open(image_path).convert("RGB")
    
    visual_description = run_task(image, "<MORE_DETAILED_CAPTION>")

    detection = run_task(image, "<CAPTION>")

    items_to_check = {"person"}
    for item in evidence["detected_ppe"] + evidence["missing_ppe"]:
        if item in PPE_VISUAL_TERMS:
            items_to_check.add(PPE_VISUAL_TERMS[item])

    ppe_search_str = ", ".join(items_to_check)
    grounding_results = run_task(
        image, 
        "<OPEN_VOCABULARY_DETECTION>", 
        ppe_search_str
    )

    tue_missing = set(evidence["required_ppe"]) - set(evidence["detected_ppe"])

    worn = ", ".join(evidence["detected_ppe"]) or "none"
    violations = ", ".join(evidence["missing_ppe"]) or "none" 
    missing = ", ".join(tue_missing) or "none"
   
    is_compliant = len(evidence["missing_ppe"]) == 0 and len(tue_missing) == 0

    verdict = "Compliant" if is_compliant else "Non-Compliant"

    explanation = f""" 
        Visual Description: {visual_description}
        Detected PPE: {worn}
        Missing PPE: {violations}
        PPE that should be worn but is not detected: {missing}
        Verdict: {verdict}
    """

    return {
        "image" : image_path,
        "visual_description": visual_description,
        "florence_detection": detection,
        "grounding_results": grounding_results,
        "detected_ppe": worn,
        "missing_ppe": violations,
        "undetected_ppe": missing,
        "verdict": verdict,
        "explanation": explanation
    }

if __name__ == "__main__":
    image_path = "data/test/images/-1x-1_jpg.rf.282bd04459c7ab3088358b5ac6ebeea7.jpg"

    evidence = {
        "required_ppe":    ["Hardhat", "Gloves", "Goggles", "Mask", "Safety Vest"],
        "detected_ppe":    ["Hardhat"],
        "missing_ppe":     ["Goggles", "Mask", "Safety Vest"],  # truly undetected PPE
        "violations":      ["NO-Gloves"],                        # explicit NO-* classes
        "fall_detected":   False,
    }
    
    print(f"Looking at image: {image_path}")

    response = generate_response(image_path, evidence)
    print(f"Florence Response: {response}")



