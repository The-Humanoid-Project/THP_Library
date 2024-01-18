import subprocess
import multiprocessing
import torch
from ultralytics import YOLO

def main():
    model = "yolov8s.pt"
    dataset = 'dataset\data.yaml'
    epochs_list = [50, 100, 150]
    learning_rate_list = [0.1, 0.01, 0.001]
    weight_decay_list = [0.1, 0.01, 0.001]

    for epochs in epochs_list:
        for learning_rate in learning_rate_list:
            for weight_decay in weight_decay_list:
                command = f"yolo train data={dataset} model={model} freeze={10} epochs={epochs} lr0={learning_rate} weight_decay={weight_decay} batch=4 imgsz=640 verbose=True"
                subprocess.run(command, shell=True)

if __name__ == '__main__':
    # Add freeze_support() for Windows multiprocessing support
    #multiprocessing.freeze_support()

    # Execute main function
    main()