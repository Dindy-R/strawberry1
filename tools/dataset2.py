"""
扩充，裁剪，绘制
"""


import cv2
import numpy as np
import os


def read_coordinates_from_txt(txt_path):
    coordinates = []
    class_labels = []
    with open(txt_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 3:
                continue  # 忽略无效行
            class_label = int(parts[0])
            x_coords = [float(parts[i]) for i in range(1, len(parts), 2)]
            y_coords = [float(parts[i]) for i in range(2, len(parts), 2)]
            coords = list(zip(x_coords, y_coords))
            coordinates.extend(coords)
            class_labels.extend([class_label] * len(coords))
    return np.array(coordinates), np.array(class_labels)


def expand_and_merge_rectangles(coordinates, class_labels, expansion_factor=1.5):
    # 分离类别1和类别2的坐标
    class_1_coords = coordinates[class_labels == 1]
    class_2_coords = coordinates[class_labels == 2]

    # 获取类别1的外接矩形
    x_min_1, y_min_1 = np.min(class_1_coords, axis=0)
    x_max_1, y_max_1 = np.max(class_1_coords, axis=0)

    # 获取类别2的外接矩形
    x_min_2, y_min_2 = np.min(class_2_coords, axis=0)
    x_max_2, y_max_2 = np.max(class_2_coords, axis=0)

    # 检查两个扩大矩形是否相交
    intersect_x_min = max(x_min_1, x_min_2)
    intersect_y_min = max(y_min_1, y_min_2)
    intersect_x_max = min(x_max_1, x_max_2)
    intersect_y_max = min(y_max_1, y_max_2)

    if intersect_x_max > intersect_x_min and intersect_y_max > intersect_y_min:
        # 合并两个类别的坐标
        merged_coords = np.concatenate((class_1_coords, class_2_coords))
        x_min, y_min = np.min(merged_coords, axis=0)
        x_max, y_max = np.max(merged_coords, axis=0)

        # 计算合并后矩形的宽高
        merged_width = x_max - x_min
        merged_height = y_max - y_min

        # 根据给定的比例扩大合并后的矩形
        expanded_x_min = max(0, x_min - merged_width * (expansion_factor - 1) / 2)
        expanded_y_min = max(0, y_min - merged_height * (expansion_factor - 1) / 2)
        expanded_x_max = min(x_max + merged_width * (expansion_factor - 1) / 2, coordinates[:, 0].max())
        expanded_y_max = min(y_max + merged_height * (expansion_factor - 1) / 2, coordinates[:, 1].max())

        return (x_min, y_min, x_max, y_max), (expanded_x_min, expanded_y_min, expanded_x_max, expanded_y_max), True

    return (x_min_1, y_min_1, x_max_1, y_max_1), (x_min_1, y_min_1, x_max_1, y_max_1), False


def crop_bounding_box(image, coordinates):
    x_min, y_min, x_max, y_max = coordinates
    cropped_image = image[int(y_min):int(y_max), int(x_min):int(x_max)]
    return cropped_image, x_min, y_min, x_max, y_max


def update_coordinates(coordinates, x_min, y_min, x_max, y_max):
    updated_coordinates = coordinates - [x_min, y_min]
    normalized_coordinates = updated_coordinates / [x_max - x_min, y_max - y_min]
    return normalized_coordinates


def draw_rectangle(image, coordinates, color=(0, 255, 0), thickness=2):
    x_min, y_min, x_max, y_max = coordinates
    cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, thickness)
    return image


def process_images_and_texts(image_folder, label_folder, output_image_folder, output_label_folder):
    os.makedirs(output_image_folder, exist_ok=True)
    os.makedirs(output_label_folder, exist_ok=True)

    image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
    label_files = [f for f in os.listdir(label_folder) if f.endswith('.txt')]

    for image_file in image_files:
        base_name = os.path.splitext(image_file)[0] + '.txt'
        label_file = os.path.join(label_folder, base_name)
        if base_name not in label_files:
            print(f"Warning: Label file {base_name} not found for image {image_file}.")
            continue

        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)

        coordinates, class_labels = read_coordinates_from_txt(label_file)
        coordinates = coordinates.astype(np.float32)

        if len(coordinates) < 3:
            print(f"Error: Coordinates must have at least 3 points for image {image_file}.")
            continue

        (x_min, y_min, x_max, y_max), (
        expanded_x_min, expanded_y_min, expanded_x_max, expanded_y_max), merged = expand_and_merge_rectangles(
            coordinates, class_labels)

        # 绘制原始的合并矩形
        image_with_rectangle = draw_rectangle(image.copy(), (x_min, y_min, x_max, y_max), color=(0, 255, 0),
                                              thickness=2)

        # 绘制扩大后的合并矩形
        image_with_expanded_rectangle = draw_rectangle(image_with_rectangle.copy(),
                                                       (expanded_x_min, expanded_y_min, expanded_x_max, expanded_y_max),
                                                       color=(0, 0, 255), thickness=2)

        # 保存带有矩形框的图像
        output_image_path = os.path.join(output_image_folder, f'drawn_{image_file}')
        cv2.imwrite(output_image_path, image_with_expanded_rectangle)

        # 使用扩大后的矩形裁剪图像
        cropped_image, _, _, _, _ = crop_bounding_box(image,
                                                      (expanded_x_min, expanded_y_min, expanded_x_max, expanded_y_max))

        # 保存裁剪后的图像
        output_cropped_image_path = os.path.join(output_image_folder, f'cropped_{image_file}')
        cv2.imwrite(output_cropped_image_path, cropped_image)

        # 更新裁剪后的坐标
        updated_coordinates = update_coordinates(coordinates, expanded_x_min, expanded_y_min, expanded_x_max,
                                                 expanded_y_max)

        # 保存更新后的标签文件
        output_label_path = os.path.join(output_label_folder, f'cropped_{os.path.splitext(base_name)[0]}.txt')
        with open(output_label_path, 'w') as file:
            current_class = None
            current_line = ""
            for i, coord in enumerate(updated_coordinates):
                if class_labels[i] != current_class:
                    if current_line:
                        file.write(current_line + '\n')
                        current_line = ""
                    current_class = class_labels[i]
                    current_line = f"{class_labels[i]}"
                current_line += f" {coord[0]} {coord[1]}"
            if current_line:
                file.write(current_line + '\n')

    print("Processing complete.")


# 设置文件夹路径
# image_folder = 'D:/User/Desktop/ultralytics/data/filtered/images'
# label_folder = 'D:/User/Desktop/ultralytics/data/filtered/labels'
# output_image_folder = 'D:/User/Desktop/ultralytics/data/team-seg/images'
# output_label_folder = 'D:/User/Desktop/ultralytics/data/team-seg/labels'

image_folder = 'D:/User/Desktop/ultralytics/data/test/images'
label_folder = 'D:/User/Desktop/ultralytics/data/test/labels'
output_image_folder = 'D:/User/Desktop/ultralytics/data/test/cropped_images'
output_label_folder = 'D:/User/Desktop/ultralytics/data/test/cropped_labels'

# 处理图像和文本文件
process_images_and_texts(image_folder, label_folder, output_image_folder, output_label_folder)
