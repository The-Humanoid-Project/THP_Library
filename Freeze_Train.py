import subprocess
import multiprocessing
from ultralytics import YOLO

def get_layers(model):
    data = model.info()
    return data[0]

def main():
    model = "yolov8s.pt"
    total_layers = get_layers(YOLO(model))
    dataset = 'dataset\data.yaml'
    epochs_list = [50, 100, 150]
    learning_rate_list = [0.1, 0.01, 0.001]
    weight_decay_list = [0.1, 0.01, 0.001]

    for epochs in epochs_list:
        for learning_rate in learning_rate_list:
            for weight_decay in weight_decay_list:
                command = f"yolo train data={dataset} model={model} freeze={total_layers-2} epochs={epochs} lr0={learning_rate} weight_decay={weight_decay}"
                subprocess.run(command, shell=True)

if __name__ == '__main__':
    # Add freeze_support() for Windows multiprocessing support
    multiprocessing.freeze_support()

    # Execute main function
    main()