import os
from PIL import Image
import numpy as np

def calculate_mean_and_variance(image_folder):
    # 初始化均值和方差的累加器
    mean_sum = np.zeros(3)
    var_sum = np.zeros(3)
    total_pixels = 0

    # 遍历文件夹中的所有图像文件
    for filename in os.listdir(image_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            # 打开图像并转换为numpy数组
            img_path = os.path.join(image_folder, filename)
            img = Image.open(img_path).convert('RGB')
            img_array = np.array(img) / 255.0  # 归一化到[0, 1]

            # 计算图像的像素总数
            height, width, channels = img_array.shape
            num_pixels = height * width
            total_pixels += num_pixels

            # 计算每个通道的均值和方差
            mean_sum += img_array.mean(axis=(0, 1)) * num_pixels
            var_sum += img_array.var(axis=(0, 1)) * num_pixels

    # 计算总体均值和方差
    mean = mean_sum / total_pixels
    variance = var_sum / total_pixels

    return mean, variance

# 示例用法
list_folder = './results'
filelist = ['real44_085','real44_09','real44','real44_11','real44_12']
for filename in filelist:
    image_folder = os.path.join(list_folder, filename)
    print(image_folder)
    mean, variance = calculate_mean_and_variance(image_folder)
    print(f"Mean: {mean}")
    print(f"Variance: {variance}")
