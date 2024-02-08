from ultralytics import YOLO

model = YOLO("yolov8m.pt")

results = model.predict(source=0, show=True)