import os


def count_classes_in_folder(folder_path):
    # 初始化类别计数器
    class_counts = {}
    file_class_counts = {}

    # 遍历文件夹内的所有 .txt 文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)

            # 读取标签文件
            with open(file_path, 'r') as file:
                lines = file.readlines()

                # 遍历每一行标签数据
                current_file_classes = set()  # 当前文件中的类别集合
                for line in lines:
                    data = line.strip().split()
                    if len(data) < 5:  # 检查是否有足够的数据
                        continue

                    # 解析类别
                    class_id = int(data[0])

                    # 更新类别计数器
                    if class_id in class_counts:
                        class_counts[class_id] += 1
                    else:
                        class_counts[class_id] = 1

                    # 记录当前文件中的类别
                    current_file_classes.add(class_id)

                # 存储当前文件中的类别
                file_class_counts[filename] = current_file_classes

    # 输出统计结果
    print("类别统计结果：")
    for class_id, count in sorted(class_counts.items()):
        print(f"类别 {class_id}: {count}")

    # 输出每个文件中包含的类别
    print("\n每个文件中包含的类别：")
    for filename, classes in file_class_counts.items():
        print(f"{filename}: {sorted(classes)}")


# 调用函数
folder_path = 'D:/BaiduNetdiskDownload/VOC2020/labels'
count_classes_in_folder(folder_path)
