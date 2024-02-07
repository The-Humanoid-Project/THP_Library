def main():
    from ultralytics import YOLO
    model = YOLO('runs\\detect\\train56\\weights\\last.pt')
    results = model.train(resume=True, batch=4)
    
if __name__ == '__main__':
    main()