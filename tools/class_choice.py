import os


def remove_class_1_from_txt_files(folder_path):
    # 遍历文件夹中的所有 .txt 文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)

            # 读取文件内容
            with open(file_path, 'r') as file:
                lines = file.readlines()

            # 删除类别为 1 的行
            filtered_lines = [line for line in lines if not (line.startswith('1'))]

            # 写入新内容
            with open(file_path, 'w') as file:
                file.writelines(filtered_lines)


# 调用函数
folder_path = 'D:/User/Desktop/ultralytics/data/caomei/labels'  # 替换为实际文件夹路径
remove_class_1_from_txt_files(folder_path)
