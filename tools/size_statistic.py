import os
from PIL import Image

def get_image_size(file_path):
    """ 获取图片的宽度和高度 """
    with Image.open(file_path) as img:
        return img.size

def count_image_sizes(directory):
    """ 统计目录下所有图片的尺寸及数量 """
    size_count = {}
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            file_path = os.path.join(directory, filename)
            try:
                size = get_image_size(file_path)
                size_str = f"{size[0]}x{size[1]}"
                if size_str in size_count:
                    size_count[size_str] += 1
                else:
                    size_count[size_str] = 1
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    return size_count

# 指定文件夹路径
directory = 'D:/User/Desktop/straeberry-det/image'

# 调用函数并打印结果
image_sizes = count_image_sizes(directory)
for size, count in image_sizes.items():
    print(f"Size: {size}, Count: {count}")
