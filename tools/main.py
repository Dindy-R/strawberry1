import os


def rename_files(start_num, folder_path):
    """
    重命名指定文件夹内的文件，从指定数字开始递增命名。

    :param start_num: 起始编号
    :param folder_path: 文件夹路径
    """
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    files.sort()  # 确保文件按名称排序

    for i, filename in enumerate(files, start=start_num):
        base_name, ext = os.path.splitext(filename)
        new_name = f"{i}{ext}"
        old_file_path = os.path.join(folder_path, filename)
        new_file_path = os.path.join(folder_path, new_name)

        # 重命名文件
        os.rename(old_file_path, new_file_path)
        print(f"Renamed {old_file_path} to {new_file_path}")


# 指定起始编号
start_number = 1

# 遍历两个子文件夹并重命名其中的文件
image_folder_path = 'D:/User/Desktop/ultralytics/data/low_data/images'
label_folder_path = 'D:/User/Desktop/ultralytics/data/low_data/labels'

rename_files(start_number, image_folder_path)
rename_files(start_number, label_folder_path)
