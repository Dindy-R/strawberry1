import os
import shutil


def move_images(txt_folder, img_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取 txt 文件夹中的所有 .txt 文件名
    txt_files = [f for f in os.listdir(txt_folder) if f.endswith('.txt')]

    # 遍历每个 .txt 文件
    for txt_file in txt_files:
        # 获取不带扩展名的文件名
        file_name = os.path.splitext(txt_file)[0]

        # 构建对应的图片文件路径
        img_path = os.path.join(img_folder, file_name + '.jpg')

        # 检查图片文件是否存在
        if os.path.exists(img_path):
            # 构建目标路径
            target_path = os.path.join(output_folder, file_name + '.jpg')

            # 移动图片文件
            shutil.move(img_path, target_path)
            print(f"Moved {file_name}.jpg to {output_folder}")


# 使用示例
txt_folder = 'D:/User/下载/text_label/text_label/2'
img_folder = 'D:/User/下载/images/images/test'
output_folder = 'D:/User/下载/labeling'

move_images(txt_folder, img_folder, output_folder)
