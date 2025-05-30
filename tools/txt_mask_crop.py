import os
import cv2
import numpy as np

def load_labels(label_path):
    with open(label_path, 'r') as file:
        labels = [list(map(float, line.strip().split())) for line in file]
    return labels

def denormalize_coordinates(labels, width, height):
    denormalized = []
    for label in labels:
        class_id = int(label[0])
        normalized_coords = label[1:]
        denormalized_coords = [int(coord * width) if i % 2 == 0 else int(coord * height) for i, coord in enumerate(normalized_coords)]
        denormalized.append([class_id] + denormalized_coords)
    print(f"Denormalized Coordinates: {denormalized}")
    return denormalized

def find_bounding_box(coordinates):
    if not coordinates:
        return 0, 0, 0, 0  # 返回默认边界框

    x_coords = [coord[i] for coord in coordinates for i in range(1, len(coord), 2)]
    y_coords = [coord[i] for coord in coordinates for i in range(2, len(coord), 2)]

    x_min = min(x_coords)
    y_min = min(y_coords)
    x_max = max(x_coords)
    y_max = max(y_coords)

    print(f"Bounding Box: ({x_min}, {y_min}, {x_max}, {y_max})")
    return x_min, y_min, x_max, y_max

def map_mask_coordinates_to_cropped_image(mask_coordinates, bounding_box, cropped_image, window_size=(800, 600)):
    x_min, y_min, x_max, y_max = bounding_box
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("Invalid bounding box coordinates")

    width = x_max - x_min
    height = y_max - y_min
    mapped_coordinates = []

    print(f"Bounding Box: {bounding_box}")
    print(f"Crop Size: {width}x{height}")

    for mask in mask_coordinates:
        class_id = mask[0]
        # 将掩码坐标相对于边界框进行偏移调整
        mapped_coords = [(coord - x_min) if i % 2 == 0 else (coord - y_min) for i, coord in enumerate(mask[1:])]

        print(f"Original Coords: {mask[1:]} -> Mapped Coords: {mapped_coords}")

        # 确保坐标值在合理范围内
        mapped_coords = [
            max(0, min(coord, width)) if i % 2 == 0 else max(0, min(coord, height))
            for i, coord in enumerate(mapped_coords)
        ]

        print(f"Clipped Coords: {mapped_coords}\n")

        # 将类别ID和调整后的坐标组合成一个列表，并添加到结果列表中
        mapped_coordinates.append([class_id] + mapped_coords)

        # 绘制多边形
        points = np.array([[mapped_coords[i], mapped_coords[i+1]] for i in range(0, len(mapped_coords), 2)], dtype=np.int32)
        points = points.reshape((-1, 1, 2))
        cv2.polylines(cropped_image, [points], isClosed=True, color=(0, 255, 0), thickness=2)

    # 显示图像
    resized_image = cv2.resize(cropped_image, window_size)
    cv2.imshow('Cropped Image', resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return mapped_coordinates

def crop_and_normalize(image, labels, bounding_box, new_images_folder, new_labels_folder):
    x_min, y_min, x_max, y_max = bounding_box
    cropped_image = image[int(y_min):int(y_max), int(x_min):int(x_max)]
    cropped_width, cropped_height, _ = cropped_image.shape
    mapped_labels = map_mask_coordinates_to_cropped_image(labels, bounding_box, cropped_image)

    # 保存裁剪后的图像
    cv2.imwrite(os.path.join(new_images_folder, f'{os.path.basename(image_path)}'), cropped_image)

    return mapped_labels

# 主程序入口
if __name__ == "__main__":
    images_folder = 'D:/User/Desktop/ultralytics/data/train/images'
    labels_folder = 'D:/User/Desktop/ultralytics/data/train/labels'
    new_images_folder = 'D:/User/Desktop/ultralytics/data/train/output/test/output/images'
    new_labels_folder = 'D:/User/Desktop/ultralytics/data/train/output/test/output/labels'

    os.makedirs(new_images_folder, exist_ok=True)
    os.makedirs(new_labels_folder, exist_ok=True)

    for filename in os.listdir(images_folder):
        if filename.endswith('.jpg'):
            image_path = os.path.join(images_folder, filename)
            label_path = os.path.join(labels_folder, filename.replace('.jpg', '.txt'))

            image = cv2.imread(image_path)
            height, width, _ = image.shape

            labels = load_labels(label_path)
            if not labels:
                continue  # 如果标签文件为空，跳过当前图像

            denormalized_labels = denormalize_coordinates(labels, width, height)
            bounding_box = find_bounding_box(denormalized_labels)
            mapped_labels = crop_and_normalize(image, denormalized_labels, bounding_box, new_images_folder, new_labels_folder)

            with open(os.path.join(new_labels_folder, filename.replace('.jpg', '.txt')), 'w') as file:
                for label in mapped_labels:
                    file.write(f"{int(label[0])} {' '.join(str(coord) for coord in label[1:])}\n")
