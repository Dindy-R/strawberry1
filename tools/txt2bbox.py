import cv2
import os

# 定义绘制边界框的颜色
color = (255, 0, 0)  # BGR格式，例如红色


def draw_boxes(image_path, labels_path):
    # 读取图像
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    # 读取标签文件
    with open(labels_path, 'r') as file:
        lines = file.readlines()

    # 遍历每一行标签数据
    for line in lines:
        data = line.strip().split()
        if len(data) < 5:  # 检查是否有足够的数据
            continue

        # 解析类别和归一化的边界框坐标
        class_id = int(data[0])
        x_center, y_center, box_w, box_h = map(float, data[1:])

        # 将归一化的坐标转换为像素坐标
        x = int((x_center - box_w / 2) * width)
        y = int((y_center - box_h / 2) * height)
        w = int(box_w * width)
        h = int(box_h * height)

        # 绘制边界框
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

        # 添加类别标签
        label = f"Class {class_id}"
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 显示图像
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Image', 800, 600)  # 设置窗口大小
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 调用函数
image_path = 'D:/User/Desktop/ultralytics/data/test_caomei/train/images/0009.jpg'
labels_path = 'D:/User/Desktop/ultralytics/data/test_caomei/train/labels/0009.txt'
draw_boxes(image_path, labels_path)
