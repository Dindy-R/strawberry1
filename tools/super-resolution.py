import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


# 定义几种传统的超分辨率方法
def bicubic_interpolation(image, target_size):
    """双三次插值"""
    return cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)


def nearest_neighbor_interpolation(image, target_size):
    """最近邻插值"""
    return cv2.resize(image, target_size, interpolation=cv2.INTER_NEAREST)


def linear_interpolation(image, target_size):
    """双线性插值"""
    return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)


def lanczos_interpolation(image, target_size):
    """Lanczos 插值"""
    return cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)


# 正则化方法
def regularized_interpolation(image, target_size, alpha=0.1):
    """正则化插值"""
    # 双三次插值
    bicubic_img = cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)
    # 计算梯度
    grad_x = cv2.Sobel(bicubic_img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(bicubic_img, cv2.CV_64F, 0, 1, ksize=3)
    # 计算正则化项
    regularization_term = alpha * (np.abs(grad_x) + np.abs(grad_y))
    # 应用正则化
    regularized_img = bicubic_img + regularization_term
    return np.clip(regularized_img, 0, 255).astype(np.uint8)


# 迭代反投影
def iterative_back_projection(image, target_size, iterations=10):
    """迭代反投影"""
    # 初始双三次插值
    bicubic_img = cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)
    # 计算降采样比例
    scale_factor = min(target_size[0] / image.shape[1], target_size[1] / image.shape[0])
    downsampled_size = (int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor))

    # 确保降采样后的图像与原始图像尺寸一致
    if downsampled_size != image.shape[:2][::-1]:
        image = cv2.resize(image, downsampled_size, interpolation=cv2.INTER_CUBIC)

    # 迭代反投影
    for _ in range(iterations):
        downsampled_img = cv2.resize(bicubic_img, downsampled_size, interpolation=cv2.INTER_CUBIC)
        error = image - downsampled_img
        error_upsampled = cv2.resize(error, target_size, interpolation=cv2.INTER_CUBIC)
        bicubic_img += error_upsampled
    return np.clip(bicubic_img, 0, 255).astype(np.uint8)


# 读取原始图像
image_path = 'D:/User/Desktop/ultralytics/data/test/images1/rgb_10.jpg'  # 替换为你的图像路径
original_image = cv2.imread(image_path)
if original_image is None:
    raise FileNotFoundError(f"Image not found at {image_path}")

# 设定目标分辨率
target_width = 1080
target_height = 1080
target_size = (target_width, target_height)

# 应用不同的超分辨率方法
bicubic_image = bicubic_interpolation(original_image, target_size)
nearest_image = nearest_neighbor_interpolation(original_image, target_size)
linear_image = linear_interpolation(original_image, target_size)
lanczos_image = lanczos_interpolation(original_image, target_size)
regularized_image = regularized_interpolation(original_image, target_size)
ibp_image = iterative_back_projection(original_image, target_size)

# 将图像从 BGR 转换为 RGB
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
bicubic_image = cv2.cvtColor(bicubic_image, cv2.COLOR_BGR2RGB)
nearest_image = cv2.cvtColor(nearest_image, cv2.COLOR_BGR2RGB)
linear_image = cv2.cvtColor(linear_image, cv2.COLOR_BGR2RGB)
lanczos_image = cv2.cvtColor(lanczos_image, cv2.COLOR_BGR2RGB)
regularized_image = cv2.cvtColor(regularized_image, cv2.COLOR_BGR2RGB)
ibp_image = cv2.cvtColor(ibp_image, cv2.COLOR_BGR2RGB)

# 保存图像
output_dir = 'D:/User/Desktop/ultralytics/data/test/images'
os.makedirs(output_dir, exist_ok=True)

cv2.imwrite(os.path.join(output_dir, '01.jpg'), cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))
cv2.imwrite(os.path.join(output_dir, '02.jpg'), cv2.cvtColor(bicubic_image, cv2.COLOR_RGB2BGR))
cv2.imwrite(os.path.join(output_dir, '03.jpg'), cv2.cvtColor(nearest_image, cv2.COLOR_RGB2BGR))
cv2.imwrite(os.path.join(output_dir, '04.jpg'), cv2.cvtColor(linear_image, cv2.COLOR_RGB2BGR))
cv2.imwrite(os.path.join(output_dir, '05.jpg'), cv2.cvtColor(lanczos_image, cv2.COLOR_RGB2BGR))
cv2.imwrite(os.path.join(output_dir, '06.jpg'), cv2.cvtColor(regularized_image, cv2.COLOR_RGB2BGR))
# cv2.imwrite(os.path.join(output_dir, '07.jpg'), cv2.cvtColor(ibp_image, cv2.COLOR_RGB2BGR))

# 绘制图像
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

# 原始图像
axes[0, 0].imshow(original_image)
axes[0, 0].set_title('Original Image')
axes[0, 0].axis('off')

# 双三次插值
axes[0, 1].imshow(bicubic_image)
axes[0, 1].set_title('Bicubic Interpolation')
axes[0, 1].axis('off')

# 最近邻插值
axes[0, 2].imshow(nearest_image)
axes[0, 2].set_title('Nearest Neighbor Interpolation')
axes[0, 2].axis('off')

# 双线性插值
axes[0, 3].imshow(linear_image)
axes[0, 3].set_title('Linear Interpolation')
axes[0, 3].axis('off')

# Lanczos 插值
axes[1, 0].imshow(lanczos_image)
axes[1, 0].set_title('Lanczos Interpolation')
axes[1, 0].axis('off')

# 正则化插值
axes[1, 1].imshow(regularized_image)
axes[1, 1].set_title('Regularized Interpolation')
axes[1, 1].axis('off')

# 迭代反投影
axes[1, 2].imshow(ibp_image)
axes[1, 2].set_title('Iterative Back Projection')
axes[1, 2].axis('off')

# 空白
axes[1, 3].axis('off')

plt.tight_layout()
# plt.show()
