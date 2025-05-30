import cv2
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np

# 加载模型
model = YOLO('D:/User/Desktop/runs/segment/train12/weights/best.pt')  # 替换为你自己的模型路径

# 加载图像
img_path = 'D:/User/Desktop/ultralytics/data/split_data_seg/train/images/0001_0.jpg'  # 替换为你的图像路径
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 缩放图像以满足模型要求
def resize_image(image, target_size=640):
    height, width = image.shape[:2]
    max_dim = max(height, width)
    scale_factor = target_size / max_dim
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return resized_image

img = resize_image(img)

# 获取模型的特征图
def get_feature_maps(model, img):
    # 将图像转换为模型输入格式
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img_tensor = img_tensor.to(next(model.parameters()).device)

    # 获取模型的中间层输出
    feature_maps = []
    def hook_fn(module, input, output):
        feature_maps.append(output)

    # 注册钩子函数
    hooks = []
    for name, layer in model.named_modules():
        if 'conv' in name:  # 只获取卷积层的特征图
            hooks.append(layer.register_forward_hook(hook_fn))

    # 前向传播
    with torch.no_grad():
        _ = model(img_tensor)

    # 移除钩子函数
    for hook in hooks:
        hook.remove()

    return feature_maps

# 获取特征图
feature_maps = get_feature_maps(model, img)

# 可视化特征图
def visualize_feature_maps(feature_maps, num_rows=4, num_cols=4):
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 15))
    for i, ax in enumerate(axes.flatten()):
        if i < len(feature_maps):
            feature_map = feature_maps[i].squeeze().cpu().numpy()
            # 确保特征图的形状是二维的
            if feature_map.ndim > 2:
                feature_map = np.mean(feature_map, axis=0)  # 取平均值以减少维度
            ax.imshow(feature_map, cmap='viridis')
            ax.axis('off')
            ax.set_title(f'Feature Map {i+1}')
        else:
            ax.axis('off')
    plt.tight_layout()
    plt.show()

# 可视化特征图
visualize_feature_maps(feature_maps)
