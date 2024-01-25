from ultralytics import YOLO

model = YOLO("FullyRetrain_Best.pt")

model.export(format="onnx", opset=15)