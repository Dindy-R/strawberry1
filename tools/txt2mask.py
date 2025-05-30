import cv2
import numpy as np
import os


def read_mask_coordinates(file_path, width, height):
    """
    从文本文件中读取掩码坐标，并返回符合条件的坐标列表。

    参数:
    - file_path: 文本文件路径
    - width: 图像宽度
    - height: 图像高度

    返回:
    - 掩码坐标列表 [np.array([[x1, y1], [x2, y2], ...]), ...]
    - 类别标签列表 [label1, label2, ...]
    """
    masks = []
    labels = []

    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            label = int(parts[0])
            coords = list(map(float, parts[1:]))

            if len(coords) % 2 != 0:
                print(f"Ignoring invalid mask data: {coords}")
                continue

            # 将归一化的坐标转换为像素坐标
            pixel_coords = [(x * width, y * height) for x, y in zip(coords[::2], coords[1::2])]
            masks.append(np.array(pixel_coords).reshape(-1, 2))
            labels.append(label)

    return masks, labels


def draw_masks(image, masks, labels):
    """
    在图像上绘制掩码及其类别标签。

    参数:
    - image: 输入图像
    - masks: 掩码坐标列表 [np.array([[x1, y1], [x2, y2], ...]), ...]
    - labels: 类别标签列表 [label1, label2, ...]

    返回:
    - 绘制后的图像
    """
    color = (0, 255, 0)  # 绿色
    thickness = 2  # 边框厚度
    alpha = 0.5  # 透明度
    font = cv2.FONT_HERSHEY_SIMPLEX  # 字体
    font_scale = 1  # 字体大小
    font_thickness = 2  # 字体厚度
    label_color = (255, 255, 255)  # 白色文字

    for i, mask in enumerate(masks):
        cv2.polylines(image, [mask.astype(int)], isClosed=True, color=color, thickness=thickness)
        # 添加透明填充效果
        overlay = image.copy()
        cv2.fillPoly(overlay, [mask.astype(int)], color)
        image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

        # 在掩码附近添加类别标签
        (text_width, text_height), _ = cv2.getTextSize(str(labels[i]), font, font_scale, font_thickness)
        text_offset_x, text_offset_y = mask[0][0], mask[0][1] - 10  # 标签位置
        box_coords = ((int(text_offset_x), int(text_offset_y)),
                      (int(text_offset_x + text_width + 2), int(text_offset_y - text_height - 2)))
        cv2.rectangle(image, box_coords[0], box_coords[1], color, cv2.FILLED)
        cv2.putText(image, str(labels[i]), (int(text_offset_x), int(text_offset_y)), font, font_scale, label_color,
                    font_thickness,
                    cv2.LINE_AA)

    return image


def process_images_in_folder(image_folder, mask_folder, output_folder=None):
    """
    处理文件夹中的图像和掩码文件，并绘制掩码。

    参数:
    - image_folder: 图像文件夹路径
    - mask_folder: 掩码文件夹路径
    - output_folder: 输出文件夹路径（可选）
    """
    image_files = sorted(os.listdir(image_folder))
    mask_files = sorted(os.listdir(mask_folder))

    for image_file, mask_file in zip(image_files, mask_files):
        image_path = os.path.join(image_folder, image_file)
        mask_file_path = os.path.join(mask_folder, mask_file)

        # 读取图像
        image = cv2.imread(image_path)

        # 检查图像是否成功读取
        if image is None:
            print(f"Failed to load image from {image_path}")
            continue

        # 获取图像的宽度和高度
        height, width = image.shape[:2]

        # 读取掩码信息
        masks, labels = read_mask_coordinates(mask_file_path, width, height)

        # 绘制掩码
        image_with_masks = draw_masks(image, masks, labels)

        # 创建窗口并设置大小
        window_name = f'Masked Image - {image_file}'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 600)

        # 显示结果
        cv2.imshow(window_name, image_with_masks)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 如果需要保存结果
        if output_folder:
            output_path = os.path.join(output_folder, image_file)
            cv2.imwrite(output_path, image_with_masks)


# 使用示例
image_folder = 'D:/User/Desktop/ultralytics/data/1/images'
mask_folder = 'D:/User/Desktop/ultralytics/data/1/labels'

# image_folder = 'D:/User/Desktop/ultralytics/data/train/output/test/images'
# mask_folder = 'D:/User/Desktop/ultralytics/data/train/output/test/labels'

output_folder = 'D:/User/Desktop/ultralytics/data/1/out'

process_images_in_folder(image_folder, mask_folder, None)
