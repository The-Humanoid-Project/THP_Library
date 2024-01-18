import subprocess
import multiprocessing
from ultralytics import YOLO

def main():
    model = "yolov8s.pt"
    dataset = 'dataset\data.yaml'
    epochs_list = [50, 100, 150]
    learning_rate_list = [0.1, 0.01, 0.001]
    weight_decay_list = [0.1, 0.01, 0.001]
    
    #################################
    # Define the last completed run #
    #################################
    last_completed_run = 5

    # Iterate through all possible combinations
    for epochs in epochs_list:
        for learning_rate in learning_rate_list:
            for weight_decay in weight_decay_list:
                run_number = epochs_list.index(epochs) * 9 + learning_rate_list.index(learning_rate) * 3 + weight_decay_list.index(weight_decay) + 1

                # Check if the run has already been completed
                if run_number <= last_completed_run:
                    print(f"Skipping Run {run_number}: Already completed.")
                    continue

                # Construct the command
                command = f"yolo train data={dataset} model={model} freeze={10} epochs={epochs} lr0={learning_rate} weight_decay={weight_decay} batch=4 imgsz=640 verbose=True"

                # Run the command
                print(f"Running {command}")
                subprocess.run(command, shell=True)

if __name__ == '__main__':
    # Add freeze_support() for Windows multiprocessing support
    #multiprocessing.freeze_support()

    # Execute main function
    main()