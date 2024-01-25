import os
import time
import csv
from ultralytics import YOLO

models = ["Freeze_Best.onnx", "Freeze_Best.pt", "FullyRetrain_Best.onnx", "FullyRetrain_Best.pt"]
dataset_folder = "Dataset_RAW"
total_times_csv_file_path = "total_times.csv"
with open(total_times_csv_file_path, mode='w', newline='') as total_times_csv_file:
    total_times_csv_writer = csv.writer(total_times_csv_file)
    total_times_csv_writer.writerow(['Model', 'Total Time (seconds)'])
    for model_name in models:
        model = YOLO(model_name)
        total_time = 0
        individual_times_csv_file_path = f"individual_times_{model_name}.csv"
        with open(individual_times_csv_file_path, mode='w', newline='') as individual_times_csv_file:
            individual_times_csv_writer = csv.writer(individual_times_csv_file)
            individual_times_csv_writer.writerow(['Image', 'Time (seconds)'])
            for root, dirs, files in os.walk(dataset_folder):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(root, file)
                        start_time = time.time()
                        results = model.predict(img_path, save=False)
                        end_time = time.time()
                        total_time += (end_time - start_time)
                        individual_times_csv_writer.writerow([file, end_time - start_time])
        total_times_csv_writer.writerow([model_name, total_time])
print(f"Total times have been exported to {total_times_csv_file_path}")
print(f"Individual times for each model have been exported to CSV files")