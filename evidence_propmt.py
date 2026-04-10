from ultralytics import YOLO
import os

# Load YOLO model
model = YOLO("/mnt/datassd3/thisali/ppe_dataset/runs/detect/train3/weights/best.pt")

# Test images folder
image_folder = "/mnt/datassd3/thisali/ppe_dataset/test/images"

# PPE classes
ppe_present_classes = [
    "Gloves",
    "Goggles",
    "Hardhat",
    "Mask",
    "Safety Vest"
]

ppe_missing_classes = [
    "NO-Gloves",
    "NO-Goggles",
    "NO-Hardhat",
    "NO-Mask",
    "NO-Safety Vest"
]

environment_classes = [
    "Fall-Detected",
    "Ladder",
    "Safety Cone"
]

# Loop through images
for image_name in os.listdir(image_folder):

    image_path = os.path.join(image_folder, image_name)

    results = model(image_path, conf=0.25)

    #################################
    # STEP 3: Extract detected classes
    #################################

    detections = []

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls)
            label = model.names[cls_id]
            detections.append(label)

    #################################
    # STEP 4: Separate class groups
    #################################

    detected_present = []
    detected_missing = []
    environment_present = []

    for label in detections:

        if label in ppe_present_classes:
            detected_present.append(label)

        if label in ppe_missing_classes:
            detected_missing.append(label.replace("NO-", ""))

        if label in environment_classes:
            environment_present.append(label)

    #################################
    # STEP 5: Build evidence block
    #################################

    evidence = {
        "Detected PPE": detected_present,
        "Missing PPE": detected_missing,
        "Environment": environment_present
    }

    #################################
    # STEP 6: Create prompt
    #################################

    prompt = f"""
You are a workplace safety inspector.

Detected PPE:
{', '.join(detected_present)}

Missing PPE:
{', '.join(detected_missing)}

Environment indicators:
{', '.join(environment_present)}

Workers must wear:
Hardhat, Gloves, Goggles, Mask, Safety Vest.

Questions:
1. Is the worker compliant with PPE safety rules?
2. What PPE is missing?
3. Are any hazards present (fall detected, ladder, safety cone)?

Explain your reasoning.
"""

    print("IMAGE:", image_name)
    print(prompt)
    print("--------------------------------------------------")
