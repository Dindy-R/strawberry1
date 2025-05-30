import os

def modify_to_label(dolder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder_path, filename)

            with open(filepath, 'r', encoding='utf-8') as file:
                lines = file.readlines()

            modified_lines = []
            for line in lines:
                parts = line.strip().split()
                if parts:
                    label = parts[0]
                    if label == '0':
                        parts[0] = '1'
                    modified_line = ''.join(parts) + '\n'
                    modified_lines.apped(modified_line)
            with open(filepath, 'w', encoding='utf-8') as file:
                file.writelines(modified_lines)

def modify_txt_files(folder_path):
    # 遍历指定文件夹
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):  # 只处理.txt文件
            file_path = os.path.join(folder_path, filename)

            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()

            # 替换每行的第一个数字
            modified_lines = []
            for line in lines:
                parts = line.strip().split()
                if parts:
                    label = parts[0]
                    if label == '1':
                        parts[0] = '0'
                    elif label == '2':
                        parts[0] = '1'
                    modified_line = ' '.join(parts) + '\n'
                    modified_lines.append(modified_line)

            # 写回文件
            with open(file_path, 'w', encoding='utf-8') as file:
                file.writelines(modified_lines)
if __name__ == '__main__':
    folder_path = 'D:/User/Desktop/ultralytics/data/low_data/labels'
    modify_txt_files(folder_path)