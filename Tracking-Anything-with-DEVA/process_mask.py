import os
import subprocess

# 设置主文件夹路径
main_folder = '/mnt/data/data/day/day_4/'

# 遍历主文件夹下的每个子文件夹
for subdir, dirs, files in os.walk(main_folder):
    # 检查当前子文件夹名称中是否包含'image_'
    if os.path.basename(subdir).startswith('image_4'):
        # 获取文件夹编号
        folder_number = os.path.basename(subdir).split('_')[-1]
        
        # 动态设置img_path和output路径
        img_path = os.path.join(subdir)
        output_path = os.path.join(os.path.dirname(subdir), f"mask_{folder_number}")

        # 构建指令
        command = [
            "python", "/home/cyber/code/gaussian-grouping/Tracking-Anything-with-DEVA/demo/demo_automatic.py",
            "--chunk_size", "4",
            "--img_path", "/mnt/data/data/waymo_scenes/0017085/image/image_5/",
            "--amp",
            "--temporal_setting", "semionline",
            "--size", "480",
            "--output", "/mnt/data/data/waymo_scenes/0017085/mask/mask_5/",
            "--use_short_id",
            "--suppress_small_objects",
            "--SAM_PRED_IOU_THRESHOLD", "0.7"
        ]

        # 打印执行的指令（可选）
        print("Running command:", " ".join(command))

        # 执行指令
        subprocess.run(command)

