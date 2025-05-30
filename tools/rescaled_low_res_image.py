import os
from PIL import Image
import numpy as np


def resize_images_and_labels(image_dir, label_dir, output_image_dir, output_label_dir, target_size=(384, 384)):
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)
    if not os.path.exists(output_label_dir):
        os.makedirs(output_label_dir)

    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # 处理图像
            image_path = os.path.join(image_dir, filename)
            image = Image.open(image_path)
            original_size = image.size

            # 调整图像尺寸
            image = image.resize(target_size, Image.Resampling.LANCZOS)
            image.save(os.path.join(output_image_dir, filename))

            # 处理标签
            label_path = os.path.join(label_dir, filename.replace('.jpg', '.txt').replace('.png', '.txt'))
            if os.path.exists(label_path):
                with open(label_path, 'r') as file:
                    lines = file.readlines()

                new_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) < 2 or (len(parts) - 1) % 2 != 0:
                        print(f"Skipping line with incorrect format: {line}")
                        continue

                    category = parts[0]
                    coords = np.array(parts[1:], dtype=float).reshape(-1, 2)

                    # 转换为原始像素坐标
                    coords[:, 0] = coords[:, 0] * original_size[0]
                    coords[:, 1] = coords[:, 1] * original_size[1]

                    # 调整坐标到新尺寸
                    coords[:, 0] = coords[:, 0] * target_size[0] / original_size[0]
                    coords[:, 1] = coords[:, 1] * target_size[1] / original_size[1]

                    # 重新归一化坐标
                    coords[:, 0] = coords[:, 0] / target_size[0]
                    coords[:, 1] = coords[:, 1] / target_size[1]

                    # 生成新的标签行
                    new_line = [category] + coords.flatten().tolist()
                    new_line = ' '.join(map(str, new_line))
                    new_lines.append(new_line)

                # 保存新标签
                with open(os.path.join(output_label_dir, filename.replace('.jpg', '.txt').replace('.png', '.txt')),
                          'w') as file:
                    file.write('\n'.join(new_lines))


resize_images_and_labels(
    'D:/User/Desktop/ultralytics/data/filtered/images',
    'D:/User/Desktop/ultralytics/data/filtered/labels',
    'D:/User/Desktop/ultralytics/data/low_data/images',
    'D:/User/Desktop/ultralytics/data/low_data/labels'
)
