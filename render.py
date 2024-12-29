#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import uuid
import glob
import os
import torch
from torchvision.utils import make_grid, save_image
from gaussian_renderer import render
from scene import Scene, GaussianModel, EnvLight, Dynamic_GaussianModel
from utils.general_utils import seed_everything, visualize_depth
from tqdm import tqdm
from argparse import ArgumentParser
from torchvision.utils import save_image
from omegaconf import OmegaConf
import imageio
import numpy as np
import concurrent.futures
import torchvision
import os
from os import makedirs
import cv2
from PIL import Image
import quaternion
from scene.cameras import Camera
import torch.nn.functional as F
from plyfile import PlyData, PlyElement
import concurrent.futures
from sklearn.decomposition import PCA
import colorsys
from nuscenes import save_nuscenes_image, save_attribute_json, save_calib, save_categories, save_ego_pose, save_sensor
from nuscenes import save_visibility, save_sample, save_sample_data, save_instance,save_scene_json, save_map_json,save_log_json

EPS = 1e-5
to8b = lambda x: (255 * np.clip(x.cpu().numpy(), 0, 1)).astype(np.uint8)


def pad_poses(p):
    """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
    bottom = np.broadcast_to([0, 0, 0, 1.], p[..., :1, :4].shape)
    return np.concatenate([p[..., :3, :4], bottom], axis=-2)


def unpad_poses(p):
    """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
    return p[..., :3, :4]

def euler_to_quaternion(roll, pitch, yaw):
    # 将欧拉角转换为四元数
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.quaternion(w, x, y, z)


def feature_to_rgb(features):
    # Input features shape: (16, H, W)

    # Reshape features for PCA
    H, W = features.shape[1], features.shape[2]
    features_reshaped = features.view(features.shape[0], -1).T

    # Apply PCA and get the first 3 components
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(features_reshaped.cpu().numpy())

    # Reshape back to (H, W, 3)
    pca_result = pca_result.reshape(H, W, 3)

    # Normalize to [0, 255]
    pca_normalized = 255 * (pca_result - pca_result.min()) / (pca_result.max() - pca_result.min())

    rgb_array = pca_normalized.astype('uint8')

    return rgb_array


def id2rgb(id, max_num_obj=256):
    if not 0 <= id <= max_num_obj:
        raise ValueError("ID should be in range(0, max_num_obj)")

    # Convert the ID into a hue value
    golden_ratio = 1.6180339887
    h = ((id * golden_ratio) % 1)  # Ensure value is between 0 and 1
    s = 0.5 + (id % 2) * 0.5  # Alternate between 0.5 and 1.0
    l = 0.5

    # Use colorsys to convert HSL to RGB
    rgb = np.zeros((3,), dtype=np.uint8)
    if id == 0:  # invalid region
        return rgb
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    rgb[0], rgb[1], rgb[2] = int(r * 255), int(g * 255), int(b * 255)

    return rgb


def visualize_obj(objects):
    rgb_mask = np.zeros((*objects.shape[-2:], 3), dtype=np.uint8)
    all_obj_ids = np.unique(objects)
    for id in all_obj_ids:
        colored_mask = id2rgb(id)
        rgb_mask[objects == id] = colored_mask
    return rgb_mask


def rotate_quaternions(quaternions, rotation_quaternion):
    rotated_quaternions = np.empty_like(quaternions)
    for i, q in enumerate(quaternions):
        q = np.quaternion(q[0], q[1], q[2], q[3])
        rotated_q = rotation_quaternion * q
        rotated_quaternions[i] = [rotated_q.w, rotated_q.x, rotated_q.y, rotated_q.z]
    return rotated_quaternions


def multithread_write(image_list, path):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=None)

    def write_image(image, count, path):
        try:
            torchvision.utils.save_image(image, os.path.join(path, '{0:05d}'.format(count) + ".png"))
            return count, True
        except:
            return count, False

    tasks = []
    for index, image in enumerate(image_list):
        tasks.append(executor.submit(write_image, image, index, path))
    executor.shutdown()
    for index, status in enumerate(tasks):
        if status == False:
            write_image(image_list[index], index, path)


def make_camera_like_input_camera(viewpoint_cam, add_xrot_val=0, add_zrot_val=0, add_tz=0):
    '''
        add_xrot_val: rotation around x-axis in camerea coordinate
        add_zrot_val: rotation around z-axis in world coordinate
        add_tz: translation in z-axis in world coordinate
    '''
    w2c = np.eye(4)
    w2c[:3, :3] = viewpoint_cam.R.transpose(1, 0)
    w2c[:3, 3] = viewpoint_cam.T.squeeze()

    c2w = np.linalg.inv(w2c)

    # ext for modification - x
    phi = add_xrot_val * np.pi / 180
    R_x = np.array([[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]])

    # ext for modification - z
    phi = add_zrot_val * np.pi / 180
    R_z = np.array([[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]])
    # t_z = add_tz

    # apply
    # c2w[0, 3] = c2w[0, 3] + t_x
    c2w[:3, :3] = np.matmul(c2w[:3, :3], R_x)

    # c2w[2, 3] = c2w[2, 3] + t_z
    c2w[:3, :3] = np.matmul(R_z, c2w[:3, :3])

    c2w[2, 3] = c2w[2, 3] + add_tz

    w2c = np.linalg.inv(c2w)

    # Return as AnnotatedCameraInstance
    viewpoint_cam_new = Camera(colmap_id=viewpoint_cam.colmap_id,
                               R=w2c[:3, :3].transpose(1, 0), T=w2c[:3, 3],
                               FoVx=viewpoint_cam.FoVx, FoVy=viewpoint_cam.FoVy,
                               cx=viewpoint_cam.cx, cy=viewpoint_cam.cy,
                               fx=viewpoint_cam.fx, fy=viewpoint_cam.fy,
                               image=viewpoint_cam.original_image,
                               image_name=viewpoint_cam.image_name,
                               uid=viewpoint_cam.uid,
                               timestamp=viewpoint_cam.timestamp,
                               resolution=viewpoint_cam.resolution,
                               image_path=viewpoint_cam.image_path,
                               pts_depth=viewpoint_cam.pts_depth,
                               sky_mask=viewpoint_cam.sky_mask,
                               )

    return viewpoint_cam_new


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]

    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )
    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
           F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
           ].reshape(batch_dim + (4,))


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    o_reshape = o.reshape(quaternions.shape[:-1] + (3, 3))
    return o_reshape


def interp_input_camera(views):
    n_jump = 1
    num_interp = 5 * n_jump

    # Use Cam 0 of stereo camera
    remove_first = 5
    views = views[remove_first:][::2][::n_jump]
    interp_views = []
    for view_t1, view_t2 in zip(views[:-1], views[1:]):
        R1 = view_t1.R
        T1 = view_t1.T  # t part of w2c

        w2c = np.eye(4)
        w2c[:3, :3] = R1.transpose(1, 0)
        w2c[:3, 3] = T1.squeeze()
        c2w = np.linalg.inv(w2c)

        R1 = c2w[:3, :3]
        T1 = c2w[:3, 3]

        q1 = matrix_to_quaternion(torch.from_numpy(R1)).numpy()
        q1 = np.quaternion(q1[0], q1[1], q1[2], q1[3])

        R2 = view_t2.R
        T2 = view_t2.T  # t part of w2c

        w2c = np.eye(4)
        w2c[:3, :3] = R2.transpose(1, 0)
        w2c[:3, 3] = T2.squeeze()
        c2w = np.linalg.inv(w2c)

        R2 = c2w[:3, :3]
        T2 = c2w[:3, 3]

        q2 = matrix_to_quaternion(torch.from_numpy(R2)).numpy()
        q2 = np.quaternion(q2[0], q2[1], q2[2], q2[3])
        point_world = (np.pad(view_t1.point_camera, ((0, 0), (0, 1)), constant_values=1) @ c2w.T)[:, :3]

        for i in range(num_interp):
            q = quaternion.slerp_evaluate(q1, q2, i / (num_interp))
            q = np.array([q.w, q.x, q.y, q.z])
            R = quaternion_to_matrix(torch.from_numpy(q)).numpy()
            T = T1 + (T2 - T1) * i / (num_interp)
            timestamp = view_t1.timestamp + (view_t2.timestamp - view_t1.timestamp) * i / (num_interp)
            c2w = np.eye(4)
            c2w[:3, :3] = R
            c2w[:3, 3] = T
            w2c = np.linalg.inv(c2w)
            R = c2w[:3, :3]
            T = w2c[:3, 3]
            point_camera = (np.pad(point_world, ((0, 0), (0, 1)), constant_values=1) @ w2c.T)[:, :3]
            i_cam = Camera(colmap_id=view_t1.colmap_id,
                           R=R, T=T,
                           FoVx=view_t1.FoVx, FoVy=view_t1.FoVy,
                           cx=view_t1.cx, cy=view_t1.cy,
                           fx=view_t1.fx, fy=view_t1.fy,
                           image=view_t1.original_image,
                           image_name=view_t1.image_name,
                           uid=view_t1.uid,
                           timestamp=timestamp,
                           resolution=view_t1.resolution,
                           image_path=view_t1.image_path,
                           pts_depth=view_t1.pts_depth,
                           sky_mask=view_t1.sky_mask,
                           objects=view_t1.objects,
                           point_camera=point_camera
                           )
            interp_views.append(i_cam)
    return interp_views


def generate_uuid():
    # 生成唯一标识符
    return str(uuid.uuid4())


@torch.no_grad()
def separation(scene: Scene, renderFunc, renderArgs, env_map=None, gaussians=None, dynamic_gaussians=None):
    scale = scene.resolution_scales[0]
    validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras(scale=scale)},
                          {'name': 'train', 'cameras': scene.getTrainCameras()})

    # we supppose area with altitude>0.5 is static
    # here z axis is downward so is gaussians.get_xyz[:, 2] < -0.5
    # high_mask = gaussians.get_xyz[:, 2] < -0.5
    # import pdb;pdb.set_trace()

    render_images = []
    gt_list = []
    render_list = []

    add_xrot_val = 0
    add_zrot_val = 0
    add_tz = 0
    num_categories = 17
    category_token = []
    for idx in range(num_categories):
        category_token.append(uuid.uuid4().hex)
    scene_id = generate_uuid()
    args_1, background_1 = renderArgs
    # save_attribute_json()
    # save_sensor(args_1.source_path)
    # save_calib(args_1.source_path)
    # save_categories(category_token=category_token, source_path=args_1.source_path)
    # save_ego_pose(args_1.source_path)
    # save_visibility(args_1.source_path)
    # save_map_json(args_1.source_path)
    # first_sample_token = uuid.uuid4().hex
    # last_sample_token = uuid.uuid4().hex
    # save_scene_json(scene_id=scene_id, first_sample_token=first_sample_token, last_sample_token=last_sample_token,
    #                 source_path=args_1.source_path)
    # save_log_json(args_1.source_path)
    # gaussians.removal_setup(args, v_mask)
    # dynamic_gaussians.add_setup(mask3d=None)
    v = gaussians.get_inst_velocity
    t_scale = gaussians.get_scaling_t.clamp_max(2)
    other = [t_scale, v]
    # other = [t_scale, v]
    if dynamic_gaussians is not None:
        dynamic_v = dynamic_gaussians.get_inst_velocity
        dynamic_t_scale = dynamic_gaussians.get_scaling_t.clamp_max(2)
        dynamic_other = [dynamic_t_scale, dynamic_v]
    else:
        dynamic_other = []
    if args.dynamic_removal:
        high_mask = gaussians.get_xyz[:, 2] < -1
        # # import pdb;pdb.set_trace()
        mask = (gaussians.get_scaling_t[:, 0] > args.separate_scaling_t) | high_mask
    else:
        mask = None
    # classifier = torch.nn.Conv2d(gaussians.num_objects, 128, kernel_size=1)
    # classifier.cuda()
    # classifier.load_state_dict(torch.load(
    # os.path.join(args.model_path, "point_cloud", "iteration_" + str(50000), "classifier.pth")))
    # velocity = gaussians.get_inst_velocity
    # v = torch.norm(velocity, dim=1)
    # v = v.detach().cpu().numpy()
    # mask = torch.from_numpy(v < 0.1).cuda()
    # mask = ((gaussians.get_xyz[:, 0] > -1.1) & (gaussians.get_xyz[:, 0] < 0.4) &
    #         (gaussians.get_xyz[:, 1] > -1.2) & (gaussians.get_xyz[:, 1] < -0.4) &
    #         (gaussians.get_xyz[:, 2] > -0.6) & (gaussians.get_xyz[:, 2] < 0.3))
    # mask = torch.from_numpy(mask).cuda()
    rotation_quaternion = euler_to_quaternion(0, 0, 0.2)
    source_transform = np.loadtxt(os.path.join(args.source_path, "transform_matrix.txt"))
    source_transform = np.linalg.inv(np.diag(np.array([5] * 3 + [1])) @ source_transform)

    for config in validation_configs:
        if config['cameras'] and len(config['cameras']) > 0:
            num_cams = len(config['cameras'])
            outdir = os.path.join(args.model_path, "separation1", config['name'])
            os.makedirs(outdir, exist_ok=True)
            render_path = os.path.join(args.model_path, "separation1", config['name'],
                                       "ours_{}".format(50000), "renders_add_3")
            static_path = os.path.join(args.model_path, "separation1", config['name'],
                                       "ours_{}".format(50000), "static")
            dynamic_path = os.path.join(args.model_path, "separation1", config['name'],
                                        "ours_{}".format(50000), "dynamic")
            gts_path = os.path.join(args.model_path, "separation1", config['name'],
                                    "ours_{}".format(50000), "gt")
            pred_obj_path = os.path.join(args.model_path, "separation1", config['name'],
                                         "ours_{}".format(50000), "objects")
            gt_obj_path = os.path.join(args.model_path, "separation1", config['name'],
                                       "ours_{}".format(50000), "gt_objects")
            pc_path = os.path.join(args.model_path, "separation1", config['name'],
                                   "ours_{}".format(50000), "pc")
            makedirs(render_path, exist_ok=True)
            makedirs(static_path, exist_ok=True)
            makedirs(gt_obj_path, exist_ok=True)
            makedirs(dynamic_path, exist_ok=True)
            makedirs(pc_path, exist_ok=True)
            i = 1
            makedirs(render_path, exist_ok=True)
            makedirs(gts_path, exist_ok=True)
            makedirs(pred_obj_path, exist_ok=True)
            viewpoint_cameras = config['cameras']
            viewpoint_cameras1 = []
            viewpoint_cameras2 = []
            viewpoint_cameras3 = []
            viewpoint_cameras4 = []
            viewpoint_cameras5 = []
            viewpoint_cameras6 = []
            viewpoint_cameras7 = []
            for viewpoint_camera in viewpoint_cameras:
                # if viewpoint_camera.colmap_id % 7 == 0:
                viewpoint_cameras1.append(viewpoint_camera)
                # elif viewpoint_camera.colmap_id % 7 == 1:
                #     viewpoint_cameras2.append(viewpoint_camera)
                # elif viewpoint_camera.colmap_id % 7 == 2:
                #     viewpoint_cameras3.append(viewpoint_camera)
                # elif viewpoint_camera.colmap_id % 7 == 3:
                #     viewpoint_cameras4.append(viewpoint_camera)
                # elif viewpoint_camera.colmap_id % 7 == 4:
                #     viewpoint_cameras5.append(viewpoint_camera)
                # elif viewpoint_camera.colmap_id % 7 == 5:
                #     viewpoint_cameras6.append(viewpoint_camera)
                # elif viewpoint_camera.colmap_id % 7 == 6:
                #     viewpoint_cameras7.append(viewpoint_camera)
            if args.enlarge:
                viewpoint_cameras1 = interp_input_camera(viewpoint_cameras1)
                viewpoint_cameras2 = interp_input_camera(viewpoint_cameras2)
                viewpoint_cameras3 = interp_input_camera(viewpoint_cameras3)
                viewpoint_cameras4 = interp_input_camera(viewpoint_cameras4)
                viewpoint_cameras5 = interp_input_camera(viewpoint_cameras5)
                viewpoint_cameras6 = interp_input_camera(viewpoint_cameras6)

            data_token = []
            i = 1
            current_camera_token_1 = uuid.uuid4().hex

            first_annotation_token = [uuid.uuid4().hex, uuid.uuid4().hex]
            last_annotation_token = [uuid.uuid4().hex, uuid.uuid4().hex]
            current_token_all = []
            for idx in range(int(num_cams/6)):
                current_token_all.append(uuid.uuid4().hex)

            # for idx, viewpoint in enumerate(tqdm(viewpoint_cameras6)):
            #     render_pkg = renderFunc(viewpoint, gaussians, dynamic_gaussians, *renderArgs, env_map=env_map,
            #                             other=other, dynamic_other=dynamic_other, mask=mask)
            #     rendering = render_pkg['render']
            #     rendering_depth = render_pkg['depth']
            #     alpha = render_pkg['alpha']
            #     if dynamic_gaussians is not None:
            #         render_static = render_pkg['render_s']
            #         rendering_depth = render_pkg['depth']
            #     if args.adding:
            #         render_static = render_pkg['render']
            #         rendering_depth = render_pkg['depth']
            #     sky_depth = 900
            #     depth = rendering_depth / alpha.clamp_min(EPS)
            #     if env_map is not None:
            #         if args.depth_blend_mode == 0:  # harmonic mean
            #             rendering_depth = 1 / (alpha / depth.clamp_min(EPS) + (1 - alpha) / sky_depth).clamp_min(EPS)
            #         elif args.depth_blend_mode == 1:
            #             rendering_depth = alpha * depth + (1 - alpha) * sky_depth
            #     rendering_depth_numpy = rendering_depth.cpu().numpy()
            #     rendering_depth = visualize_depth(rendering_depth)
            #     image_name = viewpoint.image_name
            #     folder_name = 'CAM_BACK_FISHEYE'
            #     makedirs(os.path.join(args.model_path, 'depth', folder_name), exist_ok=True)
            #     makedirs(os.path.join(args.model_path, 'image', folder_name), exist_ok=True)
            #     gt = viewpoint.original_image[0:3, :, :]
            #     torchvision.utils.save_image(rendering_depth, os.path.join(args.model_path, 'depth', 'CAM_BACK_FISHEYE',
            #                                                                image_name + ".png"))
            #     if dynamic_gaussians is not None:
            #         torchvision.utils.save_image(render_static, os.path.join(args.model_path, 'image', 'CAM_BACK_FISHEYE', image_name + ".png"))
            #     else:
            #         torchvision.utils.save_image(rendering, os.path.join(args.model_path, 'image', 'CAM_BACK_FISHEYE', image_name + ".png"))
            #     i += 1
            # i = 101

            for idx, viewpoint in enumerate(tqdm(viewpoint_cameras1)):
                render_pkg = renderFunc(viewpoint, gaussians, dynamic_gaussians, *renderArgs, env_map=env_map,
                                        other=other, dynamic_other=dynamic_other, mask=mask)
                rendering = render_pkg['render']
                rendering_depth = render_pkg['depth']
                alpha = render_pkg['alpha']
                if dynamic_gaussians is not None:
                    render_static = render_pkg['render_s']
                    rendering_depth = render_pkg['depth']
                if args.adding:
                    render_static = render_pkg['render']
                    rendering_depth = render_pkg['depth']
                sky_depth = 900
                depth = rendering_depth / alpha.clamp_min(EPS)
                if env_map is not None:
                    if args.depth_blend_mode == 0:  # harmonic mean
                        rendering_depth = 1 / (alpha / depth.clamp_min(EPS) + (1 - alpha) / sky_depth).clamp_min(EPS)
                    elif args.depth_blend_mode == 1:
                        rendering_depth = alpha * depth + (1 - alpha) * sky_depth
                rendering_depth = visualize_depth(rendering_depth)
                image_name = viewpoint.image_name
                folder_name = 'CAM_FRONT_RIGHT'
                makedirs(os.path.join(args.model_path, 'depth', folder_name), exist_ok=True)
                makedirs(os.path.join(args.model_path, 'image', folder_name), exist_ok=True)
                gt = viewpoint.original_image[0:3, :, :]
                torchvision.utils.save_image(rendering_depth, os.path.join(args.model_path, 'depth', 'CAM_FRONT_RIGHT',
                                                                           image_name + ".png"))
                if dynamic_gaussians is not None:
                    torchvision.utils.save_image(render_static, os.path.join(args.model_path, 'image', 'CAM_FRONT_RIGHT', image_name + ".png"))
                    # torchvision.utils.save_image(render_dynamic,
                    #                              os.path.join(dynamic_path, '{0:05d}'.format(i) + ".png"))
                else:
                    torchvision.utils.save_image(rendering, os.path.join(args.model_path, 'image', 'CAM_FRONT_RIGHT', image_name + ".png"))
                i += 1

            # i = 201
            #
            # for idx, viewpoint in enumerate(tqdm(viewpoint_cameras2)):
            #     render_pkg = renderFunc(viewpoint, gaussians, dynamic_gaussians, *renderArgs, env_map=env_map,
            #                             other=other, dynamic_other=dynamic_other, mask=mask)
            #     rendering = render_pkg['render']
            #     rendering_depth = render_pkg['depth']
            #     alpha = render_pkg['alpha']
            #     if dynamic_gaussians is not None:
            #         render_static = render_pkg['render_s']
            #         rendering_depth = render_pkg['depth']
            #     if args.adding:
            #         render_static = render_pkg['render']
            #         rendering_depth = render_pkg['depth']
            #     sky_depth = 900
            #     depth = rendering_depth / alpha.clamp_min(EPS)
            #     if env_map is not None:
            #         if args.depth_blend_mode == 0:  # harmonic mean
            #             rendering_depth = 1 / (alpha / depth.clamp_min(EPS) + (1 - alpha) / sky_depth).clamp_min(EPS)
            #         elif args.depth_blend_mode == 1:
            #             rendering_depth = alpha * depth + (1 - alpha) * sky_depth
            #     rendering_depth = visualize_depth(rendering_depth)
            #     image_name = viewpoint.image_name
            #     folder_name = 'CAM_FRONT_LEFT'
            #     makedirs(os.path.join(args.model_path, 'depth', folder_name), exist_ok=True)
            #     makedirs(os.path.join(args.model_path, 'image', folder_name), exist_ok=True)
            #     gt = viewpoint.original_image[0:3, :, :]
            #     torchvision.utils.save_image(rendering_depth, os.path.join(args.model_path, 'depth', 'CAM_FRONT_LEFT',
            #                                                                image_name + ".png"))
            #     if dynamic_gaussians is not None:
            #         torchvision.utils.save_image(render_static, os.path.join(args.model_path, 'image', 'CAM_FRONT_LEFT', image_name + ".png"))
            #     else:
            #         torchvision.utils.save_image(rendering, os.path.join(args.model_path, 'image', 'CAM_FRONT_LEFT', image_name + ".png"))
            #     i += 1
            #
            # i = 301
            #
            # for idx, viewpoint in enumerate(tqdm(viewpoint_cameras3)):
            #     render_pkg = renderFunc(viewpoint, gaussians, dynamic_gaussians, *renderArgs, env_map=env_map,
            #                             other=other, dynamic_other=dynamic_other, mask=mask)
            #     rendering = render_pkg['render']
            #     rendering_depth = render_pkg['depth']
            #     alpha = render_pkg['alpha']
            #     if dynamic_gaussians is not None:
            #         render_static = render_pkg['render_s']
            #         rendering_depth = render_pkg['depth']
            #     if args.adding:
            #         render_static = render_pkg['render']
            #         rendering_depth = render_pkg['depth']
            #     sky_depth = 900
            #     depth = rendering_depth / alpha.clamp_min(EPS)
            #     if env_map is not None:
            #         if args.depth_blend_mode == 0:  # harmonic mean
            #             rendering_depth = 1 / (alpha / depth.clamp_min(EPS) + (1 - alpha) / sky_depth).clamp_min(EPS)
            #         elif args.depth_blend_mode == 1:
            #             rendering_depth = alpha * depth + (1 - alpha) * sky_depth
            #     rendering_depth = visualize_depth(rendering_depth)
            #     image_name = viewpoint.image_name
            #     folder_name = 'CAM_BACK_LEFT'
            #     makedirs(os.path.join(args.model_path, 'depth', folder_name), exist_ok=True)
            #     makedirs(os.path.join(args.model_path, 'image', folder_name), exist_ok=True)
            #     gt = viewpoint.original_image[0:3, :, :]
            #     torchvision.utils.save_image(rendering_depth, os.path.join(args.model_path, 'depth', 'CAM_BACK_LEFT',
            #                                                                image_name + ".png"))
            #     if dynamic_gaussians is not None:
            #         torchvision.utils.save_image(render_static, os.path.join(args.model_path, 'image', 'CAM_BACK_LEFT', image_name + ".png"))
            #     else:
            #         torchvision.utils.save_image(rendering, os.path.join(args.model_path, 'image', 'CAM_BACK_LEFT', image_name + ".png"))
            #     i += 1
            #
            # i = 401
            # for idx, viewpoint in enumerate(tqdm(viewpoint_cameras4)):
            #     render_pkg = renderFunc(viewpoint, gaussians, dynamic_gaussians, *renderArgs, env_map=env_map,
            #                             other=other, dynamic_other=dynamic_other, mask=mask)
            #     rendering = render_pkg['render']
            #     rendering_depth = render_pkg['depth']
            #     alpha = render_pkg['alpha']
            #     if dynamic_gaussians is not None:
            #         render_static = render_pkg['render_s']
            #         rendering_depth = render_pkg['depth']
            #     if args.adding:
            #         render_static = render_pkg['render']
            #         rendering_depth = render_pkg['depth']
            #     sky_depth = 900
            #     depth = rendering_depth / alpha.clamp_min(EPS)
            #     if env_map is not None:
            #         if args.depth_blend_mode == 0:  # harmonic mean
            #             rendering_depth = 1 / (alpha / depth.clamp_min(EPS) + (1 - alpha) / sky_depth).clamp_min(EPS)
            #         elif args.depth_blend_mode == 1:
            #             rendering_depth = alpha * depth + (1 - alpha) * sky_depth
            #     rendering_depth = visualize_depth(rendering_depth)
            #     folder_name = 'CAM_BACK_RIGHT'
            #     image_name = viewpoint.image_name
            #     makedirs(os.path.join(args.model_path, 'depth', folder_name), exist_ok=True)
            #     makedirs(os.path.join(args.model_path, 'image', folder_name), exist_ok=True)
            #     gt = viewpoint.original_image[0:3, :, :]
            #     torchvision.utils.save_image(rendering_depth, os.path.join(args.model_path, 'depth', 'CAM_BACK_RIGHT',
            #                                                                image_name + ".png"))
            #     if dynamic_gaussians is not None:
            #         torchvision.utils.save_image(render_static, os.path.join(args.model_path, 'image', 'CAM_BACK_RIGHT', image_name + ".png"))
            #     else:
            #         torchvision.utils.save_image(rendering, os.path.join(args.model_path, 'image', 'CAM_BACK_RIGHT', image_name + ".png"))
            #     i += 1
            # i = 501
            #
            # for idx, viewpoint in enumerate(tqdm(viewpoint_cameras5)):
            #     render_pkg = renderFunc(viewpoint, gaussians, dynamic_gaussians, *renderArgs, env_map=env_map,
            #                             other=other, dynamic_other=dynamic_other, mask=mask)
            #     rendering = render_pkg['render']
            #     rendering_depth = render_pkg['depth']
            #     alpha = render_pkg['alpha']
            #     if dynamic_gaussians is not None:
            #         render_static = render_pkg['render_s']
            #         rendering_depth = render_pkg['depth']
            #     if args.adding:
            #         render_static = render_pkg['render']
            #         rendering_depth = render_pkg['depth']
            #     sky_depth = 900
            #     depth = rendering_depth / alpha.clamp_min(EPS)
            #     if env_map is not None:
            #         if args.depth_blend_mode == 0:  # harmonic mean
            #             rendering_depth = 1 / (alpha / depth.clamp_min(EPS) + (1 - alpha) / sky_depth).clamp_min(EPS)
            #         elif args.depth_blend_mode == 1:
            #             rendering_depth = alpha * depth + (1 - alpha) * sky_depth
            #     rendering_depth = visualize_depth(rendering_depth)
            #     folder_name = 'CAM_FRONT_FISHEYE'
            #     image_name = viewpoint.image_name
            #     makedirs(os.path.join(args.model_path, 'depth', folder_name), exist_ok=True)
            #     makedirs(os.path.join(args.model_path, 'image', folder_name), exist_ok=True)
            #     gt = viewpoint.original_image[0:3, :, :]
            #     torchvision.utils.save_image(rendering_depth,
            #                                  os.path.join(args.model_path, 'depth', 'CAM_FRONT_FISHEYE',
            #                                               image_name + ".png"))
            #     if dynamic_gaussians is not None:
            #         torchvision.utils.save_image(render_static, os.path.join(args.model_path, 'image', 'CAM_FRONT_FISHEYE', image_name + ".png"))
            #     else:
            #         torchvision.utils.save_image(rendering, os.path.join(args.model_path, 'image', 'CAM_FRONT_FISHEYE', image_name + ".png"))
            #     i += 1
            #
            # for idx, viewpoint in enumerate(tqdm(viewpoint_cameras7)):
            #     render_pkg = renderFunc(viewpoint, gaussians, dynamic_gaussians, *renderArgs, env_map=env_map,
            #                             other=other, dynamic_other=dynamic_other, mask=mask)
            #     rendering = render_pkg['render']
            #     rendering_depth = render_pkg['depth']
            #     alpha = render_pkg['alpha']
            #     if dynamic_gaussians is not None:
            #         render_static = render_pkg['render_s']
            #         rendering_depth = render_pkg['depth']
            #     if args.adding:
            #         render_static = render_pkg['render']
            #         rendering_depth = render_pkg['depth']
            #     sky_depth = 900
            #     depth = rendering_depth / alpha.clamp_min(EPS)
            #     if env_map is not None:
            #         if args.depth_blend_mode == 0:  # harmonic mean
            #             rendering_depth = 1 / (alpha / depth.clamp_min(EPS) + (1 - alpha) / sky_depth).clamp_min(EPS)
            #         elif args.depth_blend_mode == 1:
            #             rendering_depth = alpha * depth + (1 - alpha) * sky_depth
            #     rendering_depth = visualize_depth(rendering_depth)
            #     image_name = viewpoint.image_name
            #     folder_name = 'CAM_FRONT_LONGFOCUS'
            #     makedirs(os.path.join(args.model_path, 'depth', folder_name), exist_ok=True)
            #     makedirs(os.path.join(args.model_path, 'image', folder_name), exist_ok=True)
            #     gt = viewpoint.original_image[0:3, :, :]
            #     torchvision.utils.save_image(rendering_depth, os.path.join(args.model_path, 'depth', 'CAM_FRONT_LONGFOCUS',
            #                                               image_name + ".png"))
            #     if dynamic_gaussians is not None:
            #         torchvision.utils.save_image(render_static, os.path.join(args.model_path, 'image', 'CAM_FRONT_LONGFOCUS', image_name + ".png"))
            #     else:
            #         torchvision.utils.save_image(rendering, os.path.join(args.model_path, 'image', 'CAM_FRONT_LONGFOCUS', image_name + ".png"))
            #     i += 1
    out_path = os.path.join(render_path[:-8], 'renders')
    makedirs(out_path, exist_ok=True)
    fourcc = cv2.VideoWriter.fourcc(*'DIVX')
    size = (gt.shape[-1], gt.shape[-2])
    fps = float(10)  # if 'train' in out_path else float(1)
    writer = cv2.VideoWriter(os.path.join(out_path, 'result.mp4'), fourcc, fps, size)
    # i = 0
    for file_name in sorted(os.listdir(gts_path)):
        gt = np.array(Image.open(os.path.join(gts_path, file_name)))
        rgb = np.array(Image.open(os.path.join(render_path, file_name)))

        result = np.hstack([gt, rgb])
        # result = result.astype('uint8')
        result = rgb.astype('uint8')

        # Image.fromarray(result).save(os.path.join(out_path,file_name))
        writer.write(result[:, :, ::-1])
        # i = i + 1
    writer.release()

    imageio.mimwrite(os.path.join(out_path, 'video_rgb.mp4'), render_images,
                     fps=10)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--base_config", type=str, default="configs/base.yaml")
    args, _ = parser.parse_known_args()

    base_conf = OmegaConf.load(args.base_config)
    second_conf = OmegaConf.load(args.config)
    cli_conf = OmegaConf.from_cli()
    args = OmegaConf.merge(base_conf, second_conf, cli_conf)
    args.resolution_scales = args.resolution_scales[:1]
    print(args)

    seed_everything(args.seed)

    sep_path = os.path.join(args.model_path, 'separation2')
    os.makedirs(sep_path, exist_ok=True)

    gaussians = GaussianModel(args)
    if args.including_dynamic or args.adding:
        dynamic_gaussians = Dynamic_GaussianModel(args)
    else:
        dynamic_gaussians = None
    scene = Scene(args, gaussians, dynamic_gaussians, shuffle=False)

    if args.env_map_res > 0:
        env_map = EnvLight(resolution=args.env_map_res).cuda()
        env_map.training_setup(args)
    else:
        env_map = None

    static_checkpoints = glob.glob(os.path.join(args.model_path, "chkpnt*.pth"))
    assert len(static_checkpoints) > 0, "No checkpoints found."
    static_checkpoints = sorted(static_checkpoints, key=lambda x: int(x.split("chkpnt")[-1].split(".")[0]))[-1]
    (model_params, first_iter) = torch.load(static_checkpoints)
    gaussians.restore(model_params, args)
    if args.including_dynamic:
        dynamic_checkpoint = os.path.join(os.path.dirname(static_checkpoints),
                                      os.path.basename(static_checkpoints).replace("chkpnt", "dynamic_chkpnt"))
        (dynamic_model_params, second_iter) = torch.load(dynamic_checkpoint)
        # for param1, param2 in zip(dynamic_model_params_1, dynamic_model_params):
        #     if isinstance(param1, torch.Tensor) and isinstance(param2, torch.Tensor):
        #         concatenated_param = torch.cat((param1, param2), dim=0)
        #         dynamic_model_params_total.append(concatenated_param.detach().clone())
        #     else:
        #         # 对于非张量的元素，直接保留param1或者根据需要进行其他操作
        #         dynamic_model_params_total.append(param1)
        dynamic_gaussians.restore(dynamic_model_params, args)
    elif args.adding:
        dynamic_checkpoints = glob.glob(os.path.join(args.target_path, 'output', 'eval_output', "dynamic_chkpnt*.pth"))
        assert len(dynamic_checkpoints) > 0, "No checkpoints found."
        dynamic_checkpoint = (sorted(dynamic_checkpoints, key=lambda x: int(x.split("chkpnt")[-1].split(".")[0]))[-1])
        (dynamic_model_params, second_iter) = torch.load(dynamic_checkpoint)
        dynamic_gaussians.restore(dynamic_model_params, args)
    else:
        dynamic_gaussians = None

    if env_map is not None:
        env_checkpoint = os.path.join(os.path.dirname(static_checkpoints),
                                      os.path.basename(static_checkpoints).replace("chkpnt", "env_light_chkpnt"))
        (light_params, _) = torch.load(env_checkpoint)
        env_map.restore(light_params)

    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    separation(scene, render, (args, background), env_map=env_map, gaussians=gaussians,
               dynamic_gaussians=dynamic_gaussians)

    print("\Rendering statics and dynamics complete.")
