from ultralytics import YOLO


model = YOLO("yolov8n.pt")
results = model.train(
    data="/Users/nysa/doodlecode/pytorch-ultralytics/dataset/data.yaml",
    imgsz=640,
    epochs=10,
    batch=8,
    name="yolov8n_custom",
)
