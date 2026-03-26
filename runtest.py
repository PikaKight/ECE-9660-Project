from ultralytics import YOLO

model = YOLO("/mnt/datassd3/thisali/ppe_dataset/runs/detect/train3/weights/best.pt")

results = model.predict(
    source="/mnt/datassd3/thisali/ppe_dataset/test/images",
    conf=0.25,
    save=True
)
