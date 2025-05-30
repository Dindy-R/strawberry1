from ultralytics import YOLO
import cv2
import numpy as np

# 加载模型
model = YOLO("yolov8n-seg.pt")

# 进行预测
results = model("D:/User/Desktop/ultralytics/v8/man30.jpg")

# 读取原始图像
image_path = "D:/User/Desktop/ultralytics/v8/man30.jpg"
original_image = cv2.imread(image_path)

# 假设行人对应的类别ID为0
person_class_id = 0

# 创建一个新的空白图像用于绘制分割结果
segmentation_image = np.zeros_like(original_image)

# 解析预测结果
for result in results:
    masks = result.masks.cpu().numpy()  # 获取分割掩码
    boxes = result.boxes.cpu().numpy()  # 获取边界框
    classes = result.boxes.cls.cpu().numpy()  # 获取类别

    print(f"Masks shape: {masks.shape}")
    print(f"Boxes shape: {boxes.shape}")
    print(f"Classes shape: {classes.shape}")

    for mask, box, cls in zip(masks.data, boxes, classes):
        if int(cls) == person_class_id:
            print(f"Box: {box}")
            print(f"Mask shape: {mask.shape}")

            # 确保 box 有四个元素
            if len(box) < 4:
                print("Box does not have enough elements.")
                continue

            # 将掩码转换为二值图像
            mask = mask.astype(np.uint8)
            mask = cv2.resize(mask, (int(box[2] - box[0]), int(box[3] - box[1])))

            # 计算掩码的位置
            x1, y1, x2, y2 = map(int, box[:4])

            # 在新的图像上绘制掩码
            roi = segmentation_image[y1:y2, x1:x2]
            roi[mask > 0] = [0, 255, 0]  # 使用绿色高亮行人

# 合并原始图像和分割图像
combined_image = cv2.addWeighted(original_image, 0.7, segmentation_image, 0.3, 0)

# 保存结果图像
output_path = "D:/User/Desktop/ultralytics/v8/man50_segmented.jpg"
cv2.imwrite(output_path, combined_image)

print(f"分割结果已保存至: {output_path}")
