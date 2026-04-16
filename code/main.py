import json

from PIL import Image
from models.florence import generate_response
from models.ppe import ppe_pred

# PPE classes
PPE_CLASSES = [
    "Gloves",
    "Goggles",
    "Hardhat",
    "Mask",
    "Safety Vest"
]

PPE_MISSING_CLASSES = [
    "NO-Gloves",
    "NO-Goggles",
    "NO-Hardhat",
    "NO-Mask",
    "NO-Safety Vest"
]

ENV_CLASSES = [
    "Fall-Detected",
    "Ladder",
    "Safety Cone"
]

TEST_IMAGE_PATH = "data/test/images"

MODEL_PATH = "code/models/best.pt"

def generate_prompt(images: list):
    """_summary_

    Args:
        images (list): _description_
    
    example prompt:
        evidence = {
            "required_ppe":    ["Hardhat", "Gloves", "Goggles", "Mask", "Safety Vest"],
            "detected_ppe":    ["Hardhat"],
            "missing_ppe":     ["Goggles", "Mask", "Safety Vest"],  # truly undetected PPE
            "violations":      ["NO-Gloves"],                        # explicit NO-* classes
            "fall_detected":   False,
        }
    
    """
    
    results = ppe_pred(MODEL_PATH, images)

    return results


    
if __name__ == "__main__":

    image_path = [f"{TEST_IMAGE_PATH}/-1x-1_jpg.rf.282bd04459c7ab3088358b5ac6ebeea7.jpg"]

    prompt = generate_prompt(image_path)

    with open("resources/test_res/test.json", 'w') as f:
        json.dump(prompt, f, indent=4)