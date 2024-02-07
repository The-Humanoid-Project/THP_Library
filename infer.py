import cv2
import os
from ultralytics import YOLO

def capture_image(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            return None
        cv2.imshow("Press Enter to capture", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 13:
            print("Image captured successfully.")
            directory = "inferencing/captured_images"
            if not os.path.exists(directory):
                os.makedirs(directory)
            save_path = os.path.join(directory, "captured_image.jpg")
            cv2.imwrite(save_path, frame)
            cap.release()
            cv2.destroyAllWindows()
            return save_path

saved_image_path = capture_image()
if saved_image_path is not None:
    print("Image saved successfully at:", saved_image_path)
else:
    print("No image captured or saved.")

model = YOLO("Models\\train_scratch_v2x.pt")

results = model.predict(source="inferencing\captured_images\captured_image.jpg", classes=0, show=True, save=True, save_crop=True)