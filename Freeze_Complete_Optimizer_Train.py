from ultralytics import YOLO
import multiprocessing

def main():
    # Load a model
    model = YOLO('D:\\Code\\THP_Library\\runs\\detect\\train32\\weights\\last.pt')  # load a partially trained model

    # Resume training
    results = model.train(resume=True)

if __name__ == '__main__':
    # Add freeze_support() for Windows multiprocessing support
    multiprocessing.freeze_support()

    # Execute main function
    main()
