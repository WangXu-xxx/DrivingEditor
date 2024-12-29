import os
import cv2
import numpy as np

# 定义原始图像和mask图像所在的文件夹路径
original_folder = '/mnt/data/data/output/112033_1/image_8/'
mask_folder = '/mnt/data/data/output/112033_1/mask_8/Annotations/'

# 确保输出文件夹存在
output_folder = '/mnt/data/data/output/112033_1/dynamic_image_4/'
os.makedirs(output_folder, exist_ok=True)

# 获取原始图像文件和mask图像文件的列表
original_files = os.listdir(original_folder)
mask_files = os.listdir(mask_folder)

# 要保留的mask像素值列表
keep_values = [255]

# 定义膨胀操作的核（矩形核，大小可以根据需求调整）
kernel = np.ones((1, 1), np.uint8)

# 遍历每对原始图像和mask图像
for original_file in original_files:
    if original_file.endswith('.jpg') or original_file.endswith('.png'):
        # 构建mask文件路径，假设原始图像和mask图像文件名相同
        mask_file = os.path.join(mask_folder, original_file)
        
        if os.path.exists(mask_file):
            # 读取原始图像和mask图像
            original_image = cv2.imread(os.path.join(original_folder, original_file))
            mask_image = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            
            # 对mask图像进行膨胀操作
            mask_image = cv2.dilate(mask_image, kernel, iterations=1)
            
            # 创建一个空白的图像作为输出
            output_image = np.zeros_like(original_image)
            
            # 根据mask将原始图像中的动态物体保留
            for value in keep_values:
                output_image[mask_image == value] = original_image[mask_image == value]
            
            # 保存处理后的图像到输出文件夹
            output_file = os.path.join(output_folder, original_file)
            cv2.imwrite(output_file, output_image)
            
            print(f'Processed: {original_file}')
        else:
            print(f'Mask file not found for: {original_file}')
    else:
        print(f'Skipping non-image file: {original_file}')

print('Processing complete.')

