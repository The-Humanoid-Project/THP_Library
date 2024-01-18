from ultralytics import YOLO

model_FR = YOLO("D:\\Code\\THP_Library\\runs\\detect\\train24\\weights\\best.pt")
model_freeze = YOLO("D:\\Code\\THP_Library\\runs\\detect\\train46\\weights\\best.pt")

image = "D:\\Code\\THP_Library\\dataset\\test\\images\\591_jpg.rf.b039610d44b4b7b3f33173da94ae8123.jpg"

model_FR.predict(image)
model_freeze.predict(image)