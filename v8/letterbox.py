import os
import cv2
from ultralytics.data.augment import LetterBox

# 设置输入和输出文件夹
input_folder = 'D:/User/Desktop/ultralytics/data/split_data_seg/test/images'
output_folder = 'D:/User/Desktop/ultralytics/data/test/test384'

# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)

# 初始化 LetterBox 对象
letterbox = LetterBox(new_shape=(384, 384), auto=False, scaleFill=False, scaleup=True, center=True, stride=32)

# 获取所有图像文件
image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png'))]

# 遍历每个图像文件
for image_file in image_files:
    # 读取图像
    image_path = os.path.join(input_folder, image_file)
    image = cv2.imread(image_path)

    # 调用 LetterBox 进行缩放和填充
    resized_image = letterbox(image=image)

    # 保存缩放后的图像
    output_path = os.path.join(output_folder, image_file)
    cv2.imwrite(output_path, resized_image)

print(f"Resized images saved to {output_folder}")
