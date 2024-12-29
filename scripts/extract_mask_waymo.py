"""
@file   extract_masks.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Extract semantic mask

Using SegFormer, 2021. Cityscapes 83.2%
Relies on timm==0.3.2 & pytorch 1.8.1 (buggy on pytorch >= 1.9)

Installation:
    NOTE: mmcv-full==1.2.7 requires another pytorch version & conda env.
        Currently mmcv-full==1.2.7 does not support pytorch>=1.9;
            will raise AttributeError: 'super' object has no attribute '_specify_ddp_gpu_num'
        Hence, a seperate conda env is needed.

    git clone https://github.com/NVlabs/SegFormer

    conda create -n segformer python=3.8
    conda activate segformer
    # conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=11.3 -c pytorch -c conda-forge
    pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

    pip install timm==0.3.2 pylint debugpy opencv-python attrs ipython tqdm imageio scikit-image omegaconf
    pip install mmcv-full==1.2.7 --no-cache-dir

    cd SegFormer
    pip install .

Usage:
    Direct run this script in the newly set conda env.
"""
import os
import numpy as np
import cv2
from tqdm import tqdm
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette

if __name__ == "__main__":
    segformer_path = '/home/cyber/code/SegFormer/'
    config = os.path.join(segformer_path, 'local_configs', 'segformer', 'B5',
                                   'segformer.b5.1024x1024.city.160k.py')
    checkpoint = os.path.join(segformer_path, 'segformer.b5.1024x1024.city.160k.pth')
    model = init_segmentor(config, checkpoint, device='cuda')

    root = '/mnt/data/data/waymo_scenes/0005056/sky/1'

    scenes = sorted(os.listdir(root))

    for scene in scenes:
        for cam_id in range(1):
            image_dir = os.path.join(root, scene, f'image_{cam_id}')
            sky_dir = os.path.join(root, scene, f'sky_{cam_id}')
            if os.path.exists(sky_dir):
                print(f"{sky_dir} exists")
                continue
            os.makedirs(sky_dir, exist_ok=True)
            for image_name in tqdm(sorted(os.listdir(image_dir))):
                if not image_name.endswith(".png"):
                    continue
                image_path = os.path.join(image_dir, image_name)
                mask_path = os.path.join(sky_dir, image_name)
                result = inference_segmentor(model, image_path)
                mask = result[0].astype(np.uint8) #*10
                #mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
                mask = ((mask == 10).astype(np.float32) * 255).astype(np.uint8)
                cv2.imwrite(mask_path, mask)
                #mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
		# OpenCV 默认使用 BGR 顺序保存图像，所以我们需要将 RGB 转换为 BGR
		#mask_bgr = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
		# 保存图像
		#cv2.imwrite(mask_path, mask_bgr)