#coding: utf-8
from ultralytics import YOLO
import matplotlib
matplotlib.use("TkAgg")

if __name__ == '__main__':
    #加载训练好的模型
    model = YOLO('D:/User/Desktop/runs/detect/train4/weights/best.pt')
    # 对验证集进行评估
    metrics = model.val(data='D:/User/Desktop/ultralytics/v8/data.yaml')
