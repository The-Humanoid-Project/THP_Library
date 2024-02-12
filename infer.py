from ultralytics import YOLO

model = YOLO("yolov8m-pose.pt")

results = model.predict(source=0, show=True)