from ultralytics import YOLO

def train_ultralytics(
        dataset,
        model="yolov8s.pt",
        epochs=10,
        imgsz=640,
        save_dir="histories/ultralytics",
        ):
    model = YOLO("yolov8s.pt")
    # Run for M1/M2, MPS for M1 neural engine
    # model.train(
    #     device="mps",
    #     data=dataset,
    #     epochs=epochs,
    #     batch=-1,
    #     imgsz=imgsz,
    #     save_dir=save_dir,
    # )
    model.train(
        data=dataset,
        epochs=epochs,
        batch=2,
        imgsz=imgsz,
        save_dir=save_dir,
    )


def predict_ultralytics(
    img, model_path="runs/detect/train8/weights/best.pt", conf=0.5, iou=0.7
):
    model = YOLO(model_path)
    results = model.predict(img,
                            save=True,
                            conf=conf,
                            iou=iou)
    for r in results:
        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0]
            print(f"Box: {x1}, {y1}, {x2}, {y2}")
    return results
