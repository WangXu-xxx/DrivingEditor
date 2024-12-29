# DrivingEditor: 4D Composite Gaussian Splatting for Reconstruction and Edition of Dynamic Autonomous Driving Scenes
Official implementation of "DrivingEditor: 4D Composite Gaussian Splatting for Reconstruction and Edition of Dynamic Autonomous Driving Scenes". Submitted to T-IP on December **, 2024.

![main_branch](https://github.com/user-attachments/assets/48782666-ff6a-44d7-adcc-8c763c6ae76a)

## Abstract

In recent years, with the development of autonomous driving, 3D reconstruction for unbounded large-scale scenes has attracted researchers' attention. Existing methods have achieved outstanding reconstruction accuracy in autonomous driving scenes, but most of them lack the ability to edit scenes. Although some methods have the capability to edit scenarios, they are highly dependent on manually annotated 3D bounding boxes, leading to their poor scalability. To address the issues, we introduce a new Gaussian representation, called DrivingEditor, which decouples the scene into two parts and handles them by separate branches to individually model the dynamic foreground objects and the static background during the training process. By proposing a framework for decoupled modeling of scenarios, we can achieve accurate editing of any dynamic target, such as dynamic objects removal, adding and etc, meanwhile improving the reconstruction quality of autonomous driving scenes especially the dynamic foreground objects, without resorting to 3D bounding boxes. Extensive experiments on Waymo Open Dataset and KITTI benchmarks demonstrate the performance in 3D reconstruction for both dynamic and static scenes. Besides, we conduct extra experiments on unstructured large-scale scenarios, which can more convincingly demonstrate the performance and robustness of our proposed model when rendering the unstructured scenes.

## Get started
### Environment
```
# Make a conda environment.
cd PATH_TO_CODE
conda create --name DrivingEditor python=3.9
conda activate DrivingEditor

# Downloading our code
git clone https://github.com/WangXu-xxx/DrivingEditor.git

# Install requirements.
pip install -r requirements.txt
pip install open3d
pip install numpy-quaternion

# Install simple-knn
git clone https://gitlab.inria.fr/bkerbl/simple-knn.git
pip install ./simple-knn

# a modified gaussian splatting (for feature rendering)
git clone --recursive https://github.com/SuLvXiangXin/diff-gaussian-rasterization
pip install ./diff-gaussian-rasterization

# Install nvdiffrast (for Envlight)
git clone https://github.com/NVlabs/nvdiffrast
pip install ./nvdiffrast
```

### Dynamic_mask generation
```
(1) Environment Setup
cd Tracking-Anything-with-DEVA
pip install -e .
bash scripts/download_models.sh     # Download the pretrained models

git clone https://github.com/hkchengrex/Grounded-Segment-Anything.git
cd Grounded-Segment-Anything
export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True
python -m pip install -e segment_anything
python -m pip install -e GroundingDINO

(2) Generating Dynamic mask
cd Tracking-Anything-with-DEVA
python process_mask.py (modify the path_to_image in the .py)

```
Having obtained the instance id of the dynamic objects, extract the objects in the original mask by running `python mask.py`. The obtained dynamic_images are stored at the folders `dynamic_image_0{0, 1, 2, 3}`.

### Sky mask preparation

We provide an example script `scripts/extract_mask_waymo.py` to extract the sky mask from the extracted Waymo dataset, follow instructions [here](https://github.com/PJLab-ADG/neuralsim/blob/main/dataio/autonomous_driving/waymo/README.md#extract-mask-priors----for-sky-pedestrian-etc) to setup the Segformer environment.

### Data preparation
```
data
└── scenes
    └── sequence_id
        ├── calib
        │   └── frame_id.txt
        ├── image{0, 1, 2, 3, 4, 5}
        │   └── frame_id.png
        ├── sky{0, 1, 2, 3, 4, 5}
        │   └── frame_id.png
        |── pose
        |   └── frame_id.txt
        └── velodyne{0, 1, 2, 3}
        │   └── frame_id.bin
        |── dynamic_image{0, 1, 2, 3, 4, 5}
        |   └── frame_id.png
        |── points3d.ply
        |── points3d_dynamic_0{0, 1, 2}.ply
```

### Training (Background rendering)
Modify the arg including_dynamic in the `waymo_reconstruction.yaml` to True.
If set to False, meaning there is no dynamic objects in the scene.

```
# image reconstruction (with dynamic objects)
python train.py \
--config configs/waymo_reconstruction.yaml \
source_path=PATH_TO_DATA \
model_path=eval_output/scene_id \
including_dynamic=True

# image reconstruction (without dynamic objects)
python train.py \
--config configs/waymo_reconstruction.yaml \
source_path=PATH_TO_DATA \
model_path=eval_output/scene_id \
including_dynamic=False
```
After training, evaluation results can be found in `{EXPERIMENT_DIR}/eval` directory.

### Scene Reconstruction

```
By running this, we can obtain the rendered images that includes the static background without dynamic objects.

# image reconstruction (without dynamic objects)
python render.py \
--config configs/waymo_render.yaml \
source_path=PATH_TO_DATA \
model_path=eval_output/scene_id \
including_dynamic=False

# image reconstruction (removing dynamic objects)
python render.py \
--config configs/waymo_render.yaml \
source_path=PATH_TO_DATA \
model_path=eval_output/scene_id \
including_dynamic=True
```
## 🎥Video

视频演示:
 
[DrivingEditor: 4D Composite Gaussian Splatting for Reconstruction and Edition of Dynamic Autonomous Driving Scenes 
![1735461249473](https://github.com/user-attachments/assets/6e9351ff-e7ce-419b-afdc-5d312aef1e08)](https://youtu.be/dphhu2mNeyQ?si=ebsS6i-X9zZ5phVX)

## Visualization

### Scene Reconstruction

Waymo and KITTI:

![comparison_github](https://github.com/user-attachments/assets/0518ba14-c7ba-4410-91de-93094ccbd335)

Unstructured Scenes:

![mining_results_github](https://github.com/user-attachments/assets/b6a5a18c-b553-46f2-a718-18282b8fdb86)

### Scene Editing

Dynamic Removal:

![remove](https://github.com/user-attachments/assets/5f231741-bccc-4ae4-9f29-1692003e74db)

## 🎞️Demo

### Urban Scene 1:

https://github.com/user-attachments/assets/5aae487d-e05e-44ef-9566-e23da980f467

### Urban Scene 2:

https://github.com/user-attachments/assets/1e5fc249-94e9-456f-9db5-e32185455a42

### Unstructured Scene:

https://github.com/user-attachments/assets/a06b1f56-dbb7-4826-9123-3497297d2dfd

### Scene Recomposition

https://github.com/user-attachments/assets/bd80ef9f-659d-4604-a72a-453966fbee36

