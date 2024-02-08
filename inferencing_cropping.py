import os
import cv2
from ultralytics import YOLO

# Function to find the latest image index in a directory
def find_latest_image_index(directory):
    try:
        if not os.path.exists(directory):
            return 0
        image_files = [f for f in os.listdir(directory) if f.endswith('.jpg')]
        if not image_files:
            return 0
        latest_index = max([int(f.split('_')[-1].split('.')[0]) for f in image_files])
        return latest_index
    except Exception as e:
        print(f"Error finding latest image index: {e}")
        return 0

# Global variables to track image count
captured_image_count = find_latest_image_index("inferencing/captured_images")
book_count = find_latest_image_index("inferencing/Books")
Model = "Models\\train_scratch_v2x.pt"

# Function to Capture and Save Images
def capture_image(camera_index=0, target_resolution=(1920, 1080)):
    global captured_image_count
    try:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2592)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1944)
        while True:
            # Read a frame from the camera
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                return None
            resized_frame = cv2.resize(frame, target_resolution)
            cv2.imshow("Press Enter to capture", resized_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 13:
                print("Image captured successfully.")
                directory = "inferencing/captured_images"
                if not os.path.exists(directory):
                    os.makedirs(directory)
                captured_image_count += 1
                save_path = os.path.join(directory, f"captured_image_{captured_image_count}.jpg")
                cv2.imwrite(save_path, resized_frame)
                print("Image captured and saved successfully at:", save_path)
                cap.release()
                cv2.destroyAllWindows()
                return save_path
    except Exception as e:
        print(f"Error capturing image: {e}")
        return None
    
# Function to Detect and Crop Labels
def get_label_crops(book_crop_path, book_count, model):
    try:
        book_image = cv2.imread(book_crop_path)
        model = YOLO(model)
        label_results = model.predict(source=book_image, classes=1)
        for label_detection in label_results:
            data = label_detection.boxes.data.clone().detach()
            label_count = 1
            for label_subimage_parameters in data:
                x1, y1, x2, y2, score, label = label_subimage_parameters
                x1 = int(x1.item())
                y1 = int(y1.item())
                x2 = int(x2.item())
                y2 = int(y2.item())
                label_subimage = book_image[y1:y2, x1:x2]
                directory = f'inferencing/labels/book_{book_count}'
                if not os.path.exists(directory):
                    os.makedirs(directory)
                labelcrops_save_path = f'{directory}/label_{label_count}.jpg'
                cv2.imwrite(labelcrops_save_path, label_subimage)
                print(f"Label Crop {label_count} saved successfully at:", labelcrops_save_path, f" Conf: {score} ", f" Label: {label}", f" for Book {book_count}")
                label_count += 1
                
    except Exception as e:
        print(f"Error Saving Label Crops: {e}")
        return None

# Function to Detect and Crop Books
def get_book_crops(image_path, model):
    global book_count
    try:
        image = cv2.imread(image_path)
        yolomodel = YOLO(model)
        book_results = yolomodel.predict(source=image, classes=0)
        for book_detection in book_results:
            data = book_detection.boxes.data.clone().detach()
            for book_subimage_parameters in data:
                x1, y1, x2, y2, score, label = book_subimage_parameters
                x1 = int(x1.item())
                y1 = int(y1.item())
                x2 = int(x2.item())
                y2 = int(y2.item())
                book_subimage = image[y1:y2, x1:x2]
                book_count += 1
                directory = "inferencing/Books"
                bookcrops_save_path = f'{directory}/book_{book_count}.jpg'
                if not os.path.exists(directory):
                    os.makedirs(directory)
                cv2.imwrite(bookcrops_save_path, book_subimage)
                print(f"Book Crop {book_count} saved successfully at:", bookcrops_save_path, f" Conf: {score} ", f" Label: {label}")
            
                get_label_crops(bookcrops_save_path, book_count, model)
        
    except Exception as e:
        print(f"Error Saving Book Crops: {e}")
        return None
    
captured_image_path = capture_image()

get_book_crops(captured_image_path, Model)