from ultralytics import YOLO

model = YOLO("Freeze_Best.pt")

model.export(format="onnx", opset=13, dynamic=True, simplify=True)