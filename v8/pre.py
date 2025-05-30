import logging
import os
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
import cv2

model = YOLO('D:/User/Desktop/runs/segment/train14/weights/best.pt')


def predict_on_folder(folder_path, output_folder):
    """
    :param folder_path: 包含待预测图像的文件夹路径
    :param output_folder: 预测结果的保存路径.
    """
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取文件夹内所有图像文件
    images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    # 对每张图片进行预测
    for image_name in images:
        result = model.predict(os.path.join(folder_path, image_name))
        # 保存结果
        result[0].save(os.path.join(output_folder, image_name))


def predict_resize(folder_path, output_folder):
    """
    :param folder_path: 包含待预测图像的文件夹路径
    :param output_folder: 预测结果的保存路径.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    letterbox = LetterBox(new_shape=(640, 640), auto=False, scaleFill=False, scaleup=True, center=True, stride=32)

    # 对每张图片进行预测
    for image_name in images:
        img_path = os.path.join(folder_path, image_name)
        img = cv2.imread(img_path)
        if img is None:
            logging.error(f"Failed to read image {img_path}")
            continue

        # 使用 LetterBox 进行 resize
        img_resize = letterbox(image=img)

        try:
            result = model.predict(img_resize)
            result[0].save(os.path.join(output_folder, image_name))
            logging.info(f"Processed and saved {image_name}")
        except Exception as e:
            logging.error(f"Error processing {image_name}: {e}")


if __name__ == '__main__':
    input_folder = 'D:/User/Desktop/ultralytics/data/1/strawberry_crop'  # 替换为你的输入文件夹路径
    output_folder = 'D:/User/Desktop/ultralytics/data/1/pre'  # 替换为你的输出文件夹路径
    predict_resize(input_folder, output_folder)
