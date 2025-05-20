if __name__ == "__main__":
    from ultralytics import YOLO
    model = YOLO('yolo11n.pt')
    model.train(data='datasets/data.yaml', epochs=20, imgsz=440, batch=8, amp=True)
