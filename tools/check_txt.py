import os


def count_categories_in_file(file_path):
    """
    从给定的txt文件中读取并统计不同类别的数量。
    """
    categories = set()
    with open(file_path, 'r') as file:
        for line in file:
            category = int(line.split()[0])  # 假设类别是每行的第一个数字
            categories.add(category)
    return len(categories)


def check_and_delete_inconsistent_files(txt_directory, img_directory):
    """
    检查指定目录下所有txt文件中类别的数量是否一致，
    如果发现不一致，删除这些txt文件及其对应图片文件夹中的同名.jpg文件。
    """
    category_counts = []
    for filename in os.listdir(txt_directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(txt_directory, filename)
            count = count_categories_in_file(file_path)
            category_counts.append((filename, count))

    # 找出类别数量不一致的文件
    baseline_count = None
    inconsistent_files = []
    for filename, count in category_counts:
        if baseline_count is None:
            baseline_count = count
        elif count != baseline_count:
            inconsistent_files.append(filename)

    if inconsistent_files:
        print("以下文件的类别数量与其他文件不一致:")
        for file in inconsistent_files:
            print(file)

            # 删除txt文件
            txt_file_path = os.path.join(txt_directory, file)
            os.remove(txt_file_path)
            print(f"已删除 {txt_file_path}")

            # 删除对应的jpg文件
            img_file_name = file.replace('.txt', '.jpg')
            img_file_path = os.path.join(img_directory, img_file_name)
            if os.path.exists(img_file_path):
                os.remove(img_file_path)
                print(f"已删除 {img_file_path}")
            else:
                print(f"{img_file_path} 不存在")
    else:
        print("所有文件的类别数量一致。")


# 使用示例
txt_directory = 'D:/User/Desktop/ultralytics/data/split_data_det/test/labels'  # 替换为你的txt文件夹路径
img_directory = 'D:/User/Desktop/ultralytics/data/split_data_det/test/labels'  # 替换为你的图片文件夹路径
check_and_delete_inconsistent_files(txt_directory, img_directory)
