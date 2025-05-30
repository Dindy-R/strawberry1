import os
import random
import shutil
from PIL import Image

# 设置数据集划分比例
train_ratio = 0.6
val_ratio = 0.2
test_ratio = 0.2

# 确保比例之和为1
assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum up to 1"

# 数据集根目录
dataset_dir = 'D:/User/Desktop/ultralytics/data/low_res_image_straw'
images_dir = os.path.join(dataset_dir, 'images')
labels_dir = os.path.join(dataset_dir, 'labels')

# 创建输出目录
output_dir = 'D:/User/Desktop/ultralytics/data/low_data'
os.makedirs(output_dir, exist_ok=True)

# 创建子目录
sub_dirs = ['train', 'val', 'test']
for sub_dir in sub_dirs:
    os.makedirs(os.path.join(output_dir, sub_dir), exist_ok=True)
    os.makedirs(os.path.join(output_dir, sub_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, sub_dir, 'labels'), exist_ok=True)

# 获取所有图像文件
image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))]

# 打乱文件列表
random.shuffle(image_files)

# 计算分割索引
total_images = len(image_files)
train_split = int(total_images * train_ratio)
val_split = train_split + int(total_images * val_ratio)

# 分割数据集
train_files = image_files[:train_split]
val_files = image_files[train_split:val_split]
test_files = image_files[val_split:]

# 归一化坐标函数
def normalize_mask_coordinates(label_lines, image_width, image_height):
    """Normalize integer mask coordinates to range [0, 1], leave float coordinates unchanged."""
    normalized_lines = []
    for line in label_lines:
        parts = line.strip().split()
        class_id = int(parts[0]) if parts[0].isdigit() else parts[0]  # 检查类别值是否为整数
        coordinates = [float(coord) if '.' in coord else int(coord) for coord in parts[1:]]

        # 归一化整数坐标，浮点数坐标保持不变
        normalized_coordinates = [
            max(0, min(1, coord / image_width)) if isinstance(coord, int) and i % 2 == 0 else
            max(0, min(1, coord / image_height)) if isinstance(coord, int) else coord
            for i, coord in enumerate(coordinates)
        ]

        # 重新组合成字符串
        normalized_line = f"{class_id} {' '.join([str(coord) for coord in normalized_coordinates])}\n"
        normalized_lines.append(normalized_line)
    return normalized_lines

# 复制文件到各自的目录
def copy_files(file_list, split):
    for file in file_list:
        image_path = os.path.join(images_dir, file)
        # 获取文件名（不含扩展名）
        base_name = os.path.splitext(file)[0]
        label_path = os.path.join(labels_dir, base_name + '.txt')

        # 读取图像尺寸
        image = Image.open(image_path)
        image_width, image_height = image.size

        # 读取标签文件内容
        with open(label_path, 'r') as f:
            label_lines = f.readlines()

        # 归一化坐标
        normalized_lines = normalize_mask_coordinates(label_lines, image_width, image_height)

        # 复制图像文件
        shutil.copy(image_path, os.path.join(output_dir, split, 'images', file))

        # 写入归一化后的标签文件
        output_label_path = os.path.join(output_dir, split, 'labels', base_name + '.txt')
        with open(output_label_path, 'w') as f:
            f.writelines(normalized_lines)

# 复制文件到各自的目录
copy_files(train_files, 'train')
copy_files(val_files, 'val')
copy_files(test_files, 'test')

# 输出统计信息
print(f"Total images: {total_images}")
print(f"Train images: {len(train_files)}")
print(f"Validation images: {len(val_files)}")
print(f"Test images: {len(test_files)}")
