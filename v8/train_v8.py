from ultralytics import YOLO
import os

os.environ["WANDB_MODE"] = "offline"


if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/v8/yolov8m-seg.yaml')

    model.load('yolov8m-seg.pt')
    result = model.train(data='./v8/data.yaml',
                         epochs=100, imgsz=640, device=0, batch=4, workers=8, save=True,
               scale=0.5)
    metrics = model.val()