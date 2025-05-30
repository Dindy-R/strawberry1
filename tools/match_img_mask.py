import os

def delete_labels_without_images(images_dir, labels_dir):
    # 获取images文件夹下所有图片文件名
    image_filenames = set([f.split('.')[0] for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    # 遍历labels文件夹
    for label_file in os.listdir(labels_dir):
        if label_file.endswith('.txt'):
            # 获取对应图片文件的基础名（去掉扩展名）
            base_name = os.path.splitext(label_file)[0]
            # 检查图片文件是否存在
            if base_name not in image_filenames:
                # 删除不存在对应图片的标签文件
                label_path = os.path.join(labels_dir, label_file)
                os.remove(label_path)
                print(f"Deleted {label_path} because corresponding image does not exist.")

# 使用示例
images_directory = 'D:/User/Desktop/ultralytics/data/caomei/images'  # 替换为实际路径
labels_directory = 'D:/User/Desktop/ultralytics/data/caomei/labels'  # 替换为实际路径

delete_labels_without_images(images_directory, labels_directory)
