import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from scene.scene_utils import CameraInfo, SceneInfo, getNerfppNorm, fetchPly, storePly
from utils.graphics_utils import BasicPointCloud
import torch
import open3d as o3d
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation as R
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull


def pad_poses(p):
    """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
    bottom = np.broadcast_to([0, 0, 0, 1.], p[..., :1, :4].shape)
    return np.concatenate([p[..., :3, :4], bottom], axis=-2)


def unpad_poses(p):
    """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
    return p[..., :3, :4]


def transform_poses_pca(poses, fix_radius=0):
    """Transforms poses so principal components lie on XYZ axes.

  Args:
    poses: a (N, 3, 4) array containing the cameras' camera to world transforms.

  Returns:
    A tuple (poses, transform), with the transformed poses and the applied
    camera_to_world transforms.
    
    From https://github.com/SuLvXiangXin/zipnerf-pytorch/blob/af86ea6340b9be6b90ea40f66c0c02484dfc7302/internal/camera_utils.py#L161
  """
    t = poses[:, :3, 3]
    t_mean = t.mean(axis=0)
    t = t - t_mean

    eigval, eigvec = np.linalg.eig(t.T @ t)
    # Sort eigenvectors in order of largest to smallest eigenvalue.
    inds = np.argsort(eigval)[::-1]
    eigvec = eigvec[:, inds]
    rot = eigvec.T
    if np.linalg.det(rot) < 0:
        rot = np.diag(np.array([1, 1, -1])) @ rot

    transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
    poses_recentered = unpad_poses(transform @ pad_poses(poses))
    transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)

    # Flip coordinate system if z component of y-axis is negative
    if poses_recentered.mean(axis=0)[2, 1] < 0:
        poses_recentered = np.diag(np.array([1, -1, -1])) @ poses_recentered
        transform = np.diag(np.array([1, -1, -1, 1])) @ transform

    # Just make sure it's it in the [-1, 1]^3 cube
    if fix_radius > 0:
        scale_factor = 1. / fix_radius
    else:
        scale_factor = 1. / (np.max(np.abs(poses_recentered[:, :3, 3])) + 1e-5)
        scale_factor = min(1 / 10, scale_factor)

    poses_recentered[:, :3, 3] *= scale_factor
    transform = np.diag(np.array([scale_factor] * 3 + [1])) @ transform
    # print(transform)

    return poses_recentered, transform, scale_factor


def transform_point_cloud(point_cloud, transform_matrix):
    """
    将点云数据根据变换矩阵进行变换。

    参数:
    point_cloud (numpy.ndarray): 原始点云数据，每行是一个点 [x, y, z, intensity, ...]
    transform_matrix (numpy.ndarray): 4x4的变换矩阵

    返回:
    numpy.ndarray: 变换后的点云数据
    """
    # 将点云数据扩展为齐次坐标形式
    points_homogeneous = np.hstack((point_cloud[:, :3], np.ones((point_cloud.shape[0], 1))))

    # 应用变换矩阵
    transformed_points = np.dot(transform_matrix, points_homogeneous.T).T

    # 去除齐次坐标，只保留(x, y, z, intensity, ...)
    transformed_point_cloud = np.hstack((transformed_points[:, :3], point_cloud[:, 3:]))

    return transformed_point_cloud


import numpy as np
import cv2


def apply_rotation_to_all(w2c, rotation_vector):
    # 将旋转向量转换为旋转矩阵
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    # 将旋转矩阵扩展为齐次变换矩阵
    rotation_homogeneous = np.eye(4)
    rotation_homogeneous[:3, :3] = rotation_matrix

    # 创建存储变换后的c2w的数组
    w2c_transformed = np.zeros_like(w2c)

    # 对每个c2w矩阵应用旋转
    for i in range(w2c.shape[0]):
        w2c_transformed[i] = w2c[i] @ rotation_homogeneous

    return w2c_transformed


def compute_oriented_bounding_box(points):
    """
    计算点云的紧贴的3D包围盒（Oriented Bounding Box, OBB）。

    参数:
        points (numpy.ndarray): 形状为 (n, 3) 的数组，表示 n 个3D点。

    返回:
        tuple: (obb_center, obb_size, obb_rotation)
        obb_center: 包围盒中心点坐标。
        obb_size: 包围盒的尺寸（长、宽、高）。
        obb_rotation: 包围盒的旋转矩阵。
    """
    # PCA 分析
    pca = PCA(n_components=3)
    pca.fit(points)

    # PCA 变换后的点云
    points_transformed = pca.transform(points)

    # 在 PCA 空间中计算最小和最大值
    min_point = np.min(points_transformed, axis=0)
    max_point = np.max(points_transformed, axis=0)

    # 计算包围盒的中心和尺寸
    obb_center = (min_point + max_point) / 2.0
    obb_size = max_point - min_point
    obb_size[0], obb_size[1] = obb_size[1], obb_size[0]
    obb_center = pca.inverse_transform(obb_center)
    # 将包围盒的中心点从 PCA 空间转换回原始空间
    # obb_center = np.mean(points, axis=0)

    # 旋转矩阵
    obb_rotation = pca.components_
    quaternion = R.from_matrix(obb_rotation).as_quat()
    euler_angles = R.from_quat(quaternion).as_euler('xyz', degrees=False)

    # 将 pitch 和 roll 设置为 0
    euler_angles[0] = 0  # roll
    euler_angles[1] = 0  # pitch

    # 将欧拉角转换回四元数
    adjusted_quaternion = R.from_euler('xyz', euler_angles).as_quat()

    # 调整四元数顺序为 (w, x, y, z)
    adjusted_quaternion_wxyz = np.roll(adjusted_quaternion, 1)

    return obb_center, obb_size, adjusted_quaternion_wxyz


def l_shape_fitting(points_xy):
    """
    使用 L-Shape Fitting 算法在 XY 平面上拟合最小外接矩形。

    参数:
        points_xy (numpy.ndarray): 形状为 (n, 2) 的 2D 点云。

    返回:
        tuple: (rectangle_center, rectangle_size, rectangle_rotation)
    """
    # 计算点云的凸包
    hull = ConvexHull(points_xy)
    hull_points = points_xy[hull.vertices]

    # 初始化最小面积和相应的参数
    max_closeness = -float('inf')
    best_rectangle = None

    # 遍历所有凸包边来拟合矩形框
    for i in range(len(hull_points)):
        edge_vector = hull_points[(i + 1) % len(hull_points)] - hull_points[i]
        edge_angle = np.arctan2(edge_vector[1], edge_vector[0])
        rotation_matrix = np.array([
            [np.cos(edge_angle), -np.sin(edge_angle)],
            [np.sin(edge_angle), np.cos(edge_angle)]
        ])

        rotated_points = np.dot(points_xy, rotation_matrix)
        min_point = np.min(rotated_points, axis=0)
        max_point = np.max(rotated_points, axis=0)

        area = np.prod(max_point - min_point)

        # 计算在矩形框内的点数
        within_rect_mask = np.all((rotated_points >= min_point) & (rotated_points <= max_point), axis=1)
        points_inside = np.sum(within_rect_mask)

        # 计算紧密度（包含的点数与面积的比值）
        closeness = points_inside / area

        # 更新最大紧密度矩形框
        if closeness > max_closeness:
            max_closeness = closeness
            best_rectangle = (min_point, max_point, rotation_matrix)

    min_point, max_point, rotation_matrix = best_rectangle
    rectangle_center = (min_point + max_point) / 2.0
    rectangle_size = max_point - min_point
    rectangle_center_original = np.dot(rectangle_center, np.linalg.inv(rotation_matrix))

    return rectangle_center_original, rectangle_size, rotation_matrix


def compute_3d_bounding_box(points):
    """
    通过 L-Shape Fitting 算法计算 3D 包围盒。

    参数:
        points (numpy.ndarray): 形状为 (n, 3) 的数组，表示 n 个 3D 点。

    返回:
        tuple: (obb_center, obb_size, obb_rotation)
    """
    # Step 1: 压缩到 XY 平面
    points_xy = points[:, :2]

    # Step 2: 使用 L-Shape Fitting 拟合最小外接矩形
    rect_center_2d, rect_size_2d, rotation_matrix_2d = l_shape_fitting(points_xy)

    # Step 3: 恢复到 3D 包围盒
    min_z = np.min(points[:, 2])
    max_z = np.max(points[:, 2])
    height = max_z - min_z

    obb_center = np.array([rect_center_2d[0], rect_center_2d[1], (min_z + max_z) / 2.0])
    obb_size = np.array([rect_size_2d[0], rect_size_2d[1], height])
    obb_size[0], obb_size[1] = obb_size[1], obb_size[0]

    # 构建 3D 旋转矩阵
    yaw_angle = np.arctan2(rotation_matrix_2d[1, 0], rotation_matrix_2d[0, 0])
    yaw_angle += np.pi
    print(yaw_angle / 3.14 * 180)
    # Step 4: 将 yaw 角与 roll 和 pitch 为 0 一起转换为四元数
    euler_angles = [0, 0, yaw_angle]  # roll, pitch, yaw
    quaternion = R.from_euler('xyz', euler_angles).as_quat()

    # 调整四元数顺序为 (w, x, y, z)
    obb_rotation = np.roll(quaternion, 1)

    return obb_center, obb_size, obb_rotation

def transform_target_pc(target_path=None, source_transform=None, target_transform=None):
    target_left_path = os.path.join(target_path, "dynamic_velodyne_0")
    target_right_path = os.path.join(target_path, "dynamic_velodyne_1")
    target_front_path = os.path.join(target_path, "dynamic_velodyne_2")
    target_back_path = os.path.join(target_path, "dynamic_velodyne_3")
    file_names = sorted([f for f in os.listdir(target_left_path) if f.endswith('.bin')])
    # 初始化一个空列表来存储点云数组
    target_left_point_clouds = []
    target_right_point_clouds = []
    target_front_point_clouds = []
    target_back_point_clouds = []
    # 遍历文件夹内的所有 .bin 文件
    for file_name in file_names:
        if file_name.endswith('.bin'):
            # 构建完整的文件路径
            left_file_path = os.path.join(target_left_path, file_name)
            # 读取 .bin 文件，假设点云数据是 float32 类型的 n*4 数组
            if os.path.exists(left_file_path):
                point_cloud = np.fromfile(left_file_path, dtype=np.float32).reshape(-1, 4)
                xyz = point_cloud[:, :3]
                xyz = np.hstack([xyz, np.ones((len(xyz), 1))])
                # 应用旋转矩阵到 xyz
                rotated_xyz = (xyz @ target_transform.T @ source_transform.T)[:, :3]  # 或者使用 np.dot(xyz, rotation_matrix.T)
                # 保持 intensity 部分不变
                intensity = point_cloud[:, 3]
                # 重新组合为 N*4 的数组
                point_cloud = np.hstack((rotated_xyz, intensity[:, np.newaxis]))
                # 将点云数组添加到列表中
                target_left_point_clouds.append(point_cloud)
            right_file_path = os.path.join(target_right_path, file_name)
            # 读取 .bin 文件，假设点云数据是 float32 类型的 n*4 数组
            if os.path.exists(right_file_path):
                point_cloud = np.fromfile(right_file_path, dtype=np.float32).reshape(-1, 4)
                # 将点云数组添加到列表中
                xyz = point_cloud[:, :3]
                xyz = np.hstack([xyz, np.ones((len(xyz), 1))])
                # 应用旋转矩阵到 xyz
                rotated_xyz = (xyz @ target_transform.T @ source_transform.T)[:, :3]  # 或者使用 np.dot(xyz, rotation_matrix.T)
                # 保持 intensity 部分不变
                intensity = point_cloud[:, 3]
                # 重新组合为 N*4 的数组
                point_cloud = np.hstack((rotated_xyz, intensity[:, np.newaxis]))
                target_right_point_clouds.append(point_cloud)
            front_file_path = os.path.join(target_front_path, file_name)
            # 读取 .bin 文件，假设点云数据是 float32 类型的 n*4 数组
            if os.path.exists(front_file_path):
                point_cloud = np.fromfile(front_file_path, dtype=np.float32).reshape(-1, 4)
                # 将点云数组添加到列表中
                xyz = point_cloud[:, :3]
                xyz = np.hstack([xyz, np.ones((len(xyz), 1))])
                # 应用旋转矩阵到 xyz
                rotated_xyz = (xyz @ target_transform.T @ source_transform.T)[:, :3]  # 或者使用 np.dot(xyz, rotation_matrix.T)
                # 保持 intensity 部分不变
                intensity = point_cloud[:, 3]
                # 重新组合为 N*4 的数组
                point_cloud = np.hstack((rotated_xyz, intensity[:, np.newaxis]))
                target_front_point_clouds.append(point_cloud)
            back_file_path = os.path.join(target_back_path, file_name)
            # 读取 .bin 文件，假设点云数据是 float32 类型的 n*4 数组
            if os.path.exists(back_file_path):
                point_cloud = np.fromfile(back_file_path, dtype=np.float32).reshape(-1, 4)
                # 将点云数组添加到列表中
                xyz = point_cloud[:, :3]
                xyz = np.hstack([xyz, np.ones((len(xyz), 1))])
                # 应用旋转矩阵到 xyz
                rotated_xyz = (xyz @ target_transform.T @ source_transform.T)[:, :3]  # 或者使用 np.dot(xyz, rotation_matrix.T)
                # 保持 intensity 部分不变
                intensity = point_cloud[:, 3]
                # 重新组合为 N*4 的数组
                point_cloud = np.hstack((rotated_xyz, intensity[:, np.newaxis]))
                target_back_point_clouds.append(point_cloud)
    return target_left_point_clouds, target_right_point_clouds, target_front_point_clouds, target_back_point_clouds

def fast_distances(current_point, transformed_points):
    # Subtract the current_point from all transformed_points and compute the squared differences
    diff = transformed_points - current_point
    # Sum the squared differences along the last axis (for each point) and take the square root
    distances = np.sqrt(np.sum(diff ** 2, axis=1))
    return distances

def readWaymoInfo(args):
    cam_infos = []
    car_list = [f[:-4] for f in sorted(os.listdir(os.path.join(args.source_path, "calib"))) if f.endswith('.txt')]
    source_transform = np.loadtxt(os.path.join(args.source_path, "transform_matrix.txt"))
    target_transform = np.loadtxt(os.path.join(args.target_path, "transform_matrix.txt"))
    points = []
    source_transform = np.linalg.inv(np.diag(np.array([5] * 3 + [1])) @ source_transform)
    target_transform = np.diag(np.array([5] * 3 + [1])) @ target_transform
    points_time = []
    points1 = []
    frame_num = len(car_list)
    if args.frame_interval > 0:
        time_duration = [-args.frame_interval * (frame_num - 1) / 2, args.frame_interval * (frame_num - 1) / 2]
    else:
        time_duration = args.time_duration
    pcd = o3d.io.read_point_cloud(os.path.join(args.source_path, "points3d.ply"))
    # if args.require_boundingbox:
    dynamic_pcd_1 = o3d.io.read_point_cloud(os.path.join(args.source_path, "points3d_dynamic_0.ply"))
    dynamic_pcd_2 = o3d.io.read_point_cloud(os.path.join(args.source_path, "points3d_dynamic_1.ply"))
    dynamic_pcd_3 = o3d.io.read_point_cloud(os.path.join(args.source_path, "points3d_dynamic_2.ply"))
    dynamic_pcd_4 = o3d.io.read_point_cloud(os.path.join(args.source_path, "points3d_dynamic_3.ply"))
    target_dynamic_pcd = o3d.io.read_point_cloud(os.path.join(args.target_path, "points3d_dynamic_0.ply"))
    # else:
    #     dynamic_pcd_1 = o3d.io.read_point_cloud(os.path.join(args.source_path, "points3d_dynamic_8.ply"))
    #     dynamic_pcd_2 = o3d.io.read_point_cloud(os.path.join(args.source_path, "points3d_dynamic_9.ply"))
    pointcloud1 = np.asarray(pcd.points)
    target_pc = np.asarray(target_dynamic_pcd.points)
    target_pc = np.hstack([target_pc[:, :3], np.ones((len(target_pc), 1))])
    target_pc = (target_pc @ target_transform.T @ source_transform.T)[:, :3]
    bbox_translation, bbox_size, bbox_rotation = compute_3d_bounding_box(target_pc)
    pointcloud_time = np.asarray(pcd.normals)[:, 0].reshape(-1, 1)
    mask = np.ones(len(pointcloud1), dtype=bool)
    dynamic_pointcloud_1 = np.asarray(dynamic_pcd_1.points)
    pointcloud_dynamic_time_1 = np.asarray(dynamic_pcd_1.normals)[:, 0].reshape(-1, 1)
    dynamic_pointcloud_2 = np.asarray(dynamic_pcd_2.points)
    pointcloud_dynamic_time_2 = np.asarray(dynamic_pcd_2.normals)[:, 0].reshape(-1, 1)
    dynamic_pointcloud_3 = np.asarray(dynamic_pcd_3.points)
    pointcloud_dynamic_time_3 = np.asarray(dynamic_pcd_3.normals)[:, 0].reshape(-1, 1)
    dynamic_pointcloud_4 = np.asarray(dynamic_pcd_4.points)
    pointcloud_dynamic_time_4 = np.asarray(dynamic_pcd_4.normals)[:, 0].reshape(-1, 1)
    dynamic_pointcloud = np.vstack((dynamic_pointcloud_1, dynamic_pointcloud_2))
    dynamic_pointcloud = np.vstack((dynamic_pointcloud, dynamic_pointcloud_3))
    dynamic_pointcloud = np.vstack((dynamic_pointcloud, dynamic_pointcloud_4))
    pointcloud_dynamic_time = np.vstack((pointcloud_dynamic_time_1, pointcloud_dynamic_time_2))
    pointcloud_dynamic_time = np.vstack((pointcloud_dynamic_time, pointcloud_dynamic_time_3))
    pointcloud_dynamic_time = np.vstack((pointcloud_dynamic_time, pointcloud_dynamic_time_4))
    if args.including_dynamic:
        for dynamic_point in dynamic_pointcloud:
            # 计算每个动态物体点与原始点云中点的距离，根据需要调整阈值
            distances = np.linalg.norm(pointcloud1 - dynamic_point, axis=1)
            mask[distances < 0.1] = False
        static_indices = np.where(mask)[0]
        # dynamic_indices = np.where(~mask)[0]
        static_pointcloud = pointcloud1[static_indices]
        # dynamic_pointcloud_1 = pointcloud1[dynamic_indices]
        # dynamic_pointcloud = np.vstack((dynamic_pointcloud, dynamic_pointcloud_1))
        static_time = pointcloud_time[static_indices]
        # dynamic_time_1 = pointcloud_time[dynamic_indices]
        # dynamic_time = np.vstack((pointcloud_dynamic_time, dynamic_time_1))
        dynamic_time = pointcloud_dynamic_time
    else:
        static_pointcloud = pointcloud1
        static_time = pointcloud_time
        dynamic_time = pointcloud_dynamic_time_1
    # static_time = pointcloud_time[static_indices]
    # static_pointcloud = pointcloud1
    # indices = np.random.choice(static_pointcloud.shape[0], args.num_pts, replace=True)
    # pointcloud_static = pointcloud1[indices]
    add_middle = [4274.7, 756.0, -142.9, 1]
    # add_middle = np.asarray(add_middle) @ target_transform.T @ source_transform.T
    target_left_pc, target_right_pc, target_front_pc, target_back_pc = transform_target_pc(args.target_path,
                                                                                           source_transform,
                                                                                           target_transform)
    # ego_pose_world_to_car = np.linalg.inv(ego_pose)
    pose_id = 0
    last_bbox_translation_1 = None
    last_bbox_translation_2 = None
    direction_yaw_1 = None
    direction_yaw_2 = None
    for idx, car_id in tqdm(enumerate(car_list), desc="Loading data"):
        ego_pose = np.loadtxt(os.path.join(args.source_path, 'pose', car_id + '.txt'))

        # CAMERA DIRECTION: RIGHT DOWN FORWARDS
        with open(os.path.join(args.source_path, 'calib', car_id + '.txt')) as f:
            calib_data = f.readlines()
            L = [list(map(float, line.split()[1:])) for line in calib_data]
        Ks = np.array(L[:7]).reshape(-1, 3, 4)[:, :, :3]
        lidar2cam = np.array(L[-7:]).reshape(-1, 3, 4)
        lidar2cam = pad_poses(lidar2cam)
        # lidar2cam = np.linalg.inv(lidar2cam)

        cam2lidar = np.linalg.inv(lidar2cam)
        c2w = ego_pose @ cam2lidar
        w2c = np.linalg.inv(c2w)
        images = []
        image_paths = []
        dynamic_images = []
        HWs = []
        masks_data = []

        for subdir in ['image_0', 'image_1', 'image_2', 'image_3', 'image_4', 'image_5', 'image_6'][:args.cam_num]:
            image_path = os.path.join(args.source_path, 'image', subdir, car_id + '.png')
            im_data = Image.open(image_path)
            W, H = im_data.size
            image = np.array(im_data) / 255.
            HWs.append((H, W))
            images.append(image)
            image_paths.append(image_path)
        if args.including_dynamic:
            for subdir in ['dynamic_image_0', 'dynamic_image_1', 'dynamic_image_2', 'dynamic_image_3',
                           'dynamic_image_4', 'dynamic_image_5', 'dynamic_image_6'][:args.cam_num]:
                dynamic_image_path = os.path.join(args.source_path, 'dynamic_image', subdir, car_id + '.png')
                dynamic_im_data = Image.open(dynamic_image_path)
                # W, H = dynamic_im_data.size
                dynamic_image = np.array(dynamic_im_data) / 255.
                # HWs.append((H, W))
                dynamic_images.append(dynamic_image)
            # image_paths.append(image_path)
            for subdir in ['mask_0', 'mask_1', 'mask_2', 'mask_3', 'mask_4', 'mask_5', 'mask_6'][:args.cam_num]:
                mask_data = np.array(Image.open(os.path.join(args.source_path, 'mask', subdir, 'Annotations', car_id + '.png')))
                masks_data.append(mask_data.astype(np.float32))
        else:
            dynamic_images = None
            masks_data = None

        sky_masks = []
        for subdir in ['sky_0', 'sky_1', 'sky_2', 'sky_3', 'sky_4', 'sky_5', 'sky_6'][:args.cam_num]:
            sky_data = np.array(Image.open(os.path.join(args.source_path, 'sky', subdir, car_id + '.png')))
            sky_mask = sky_data > 0
            sky_masks.append(sky_mask.astype(np.float32))

        timestamp = time_duration[0] + (time_duration[1] - time_duration[0]) * idx / (len(car_list) - 1)
        # point_left = np.fromfile(os.path.join(args.source_path, "velodyne_0", car_id + ".bin"),
        #                          dtype=np.float32, count=-1).reshape(-1, 3)
        # point_right = np.fromfile(os.path.join(args.source_path, "velodyne_1", car_id + ".bin"),
        #                           dtype=np.float32, count=-1).reshape(-1, 3)
        # point_front = np.fromfile(os.path.join(args.source_path, "velodyne_2", car_id + ".bin"),
        #                           dtype=np.float32, count=-1).reshape(-1, 3)
        # point_back = np.fromfile(os.path.join(args.source_path, "velodyne_3", car_id + ".bin"),
        #                          dtype=np.float32, count=-1).reshape(-1, 3)
        # point_left = np.fromfile(os.path.join(args.source_path, "velodyne_0", car_id + ".bin"),
        #                          dtype=np.float32, count=-1).reshape(-1, 3)
        # point_right = np.fromfile(os.path.join(args.source_path, "velodyne_1", car_id + ".bin"),
        #                           dtype=np.float32, count=-1).reshape(-1, 3)
        # point_front = np.fromfile(os.path.join(args.source_path, "velodyne_2", car_id + ".bin"),
        #                           dtype=np.float32, count=-1).reshape(-1, 3)
        # point_back = np.fromfile(os.path.join(args.source_path, "velodyne_3", car_id + ".bin"),
        #                          dtype=np.float32, count=-1).reshape(-1, 3)
        point_left = np.fromfile(os.path.join(args.source_path, 'velodyne', "velodyne_0", car_id + ".bin"),
                                 dtype=np.float32, count=-1).reshape(-1, 4)[:, :3]
        point_right = np.fromfile(os.path.join(args.source_path, 'velodyne', "velodyne_1", car_id + ".bin"),
                                  dtype=np.float32, count=-1).reshape(-1, 4)[:, :3]
        point_front = np.fromfile(os.path.join(args.source_path, 'velodyne', "velodyne_2", car_id + ".bin"),
                                  dtype=np.float32, count=-1).reshape(-1, 4)[:, :3]
        point_back = np.fromfile(os.path.join(args.source_path, 'velodyne', "velodyne_3", car_id + ".bin"),
                                 dtype=np.float32, count=-1).reshape(-1, 4)[:, :3]
        intensity_left = np.fromfile(os.path.join(args.source_path, 'velodyne', "velodyne_0", car_id + ".bin"),
                                     dtype=np.float32, count=-1).reshape(-1, 4)[:, 3]
        intensity_right = np.fromfile(os.path.join(args.source_path, 'velodyne', "velodyne_1", car_id + ".bin"),
                                      dtype=np.float32, count=-1).reshape(-1, 4)[:, 3]
        intensity_front = np.fromfile(os.path.join(args.source_path, 'velodyne', "velodyne_2", car_id + ".bin"),
                                      dtype=np.float32, count=-1).reshape(-1, 4)[:, 3]
        intensity_back = np.fromfile(os.path.join(args.source_path, 'velodyne', "velodyne_3", car_id + ".bin"),
                                     dtype=np.float32, count=-1).reshape(-1, 4)[:, 3]
        point_left_intensity = np.fromfile(os.path.join(args.source_path, 'velodyne', "velodyne_0", car_id + ".bin"),
                                           dtype=np.float32, count=-1).reshape(-1, 4)
        point_right_intensity = np.fromfile(os.path.join(args.source_path, 'velodyne', "velodyne_1", car_id + ".bin"),
                                            dtype=np.float32, count=-1).reshape(-1, 4)
        point_front_intensity = np.fromfile(os.path.join(args.source_path, 'velodyne', "velodyne_2", car_id + ".bin"),
                                            dtype=np.float32, count=-1).reshape(-1, 4)
        point_back_intensity = np.fromfile(os.path.join(args.source_path, 'velodyne', "velodyne_3", car_id + ".bin"),
                                           dtype=np.float32, count=-1).reshape(-1, 4)
        left_trans_matrix = np.array([[-0.86303205, -0.50434758, 0.02844427, 6.70597696],
                                      [0.50486678, -0.86306342, 0.0151969, 2.31031561],
                                      [0.01688469, 0.02747598, 0.99947985, 0.64450169],
                                      [0., 0., 0., 1., ]])
        right_trans_matrix = np.array([[8.65860855e-01, -5.00166888e-01, 1.08666980e-02, 6.68131590e+00],
                                       [5.00280667e-01, 8.65738777e-01, -1.46848621e-02, -2.27303791e+00],
                                       [-2.06284082e-03, 1.81514459e-02, 9.99833121e-01, 6.50227487e-01],
                                       [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        front_trans_matrix = np.array([[9.99895246e-01, 6.14484977e-03, -1.31048631e-02, 6.42883158e+00],
                                       [-6.05805319e-03, 9.99959520e-01, 6.65267796e-03, -3.08932345e-02],
                                       [1.31452123e-02, -6.57259111e-03, 9.99891996e-01, 3.61602998e+00],
                                       [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        back_trans_matrix = np.array([[-0.00562814, 0.99924799, 0.03836436, -1.90121627],
                                      [-0.99998044, -0.00572901, 0.00251989, 0.06057056],
                                      [0.00273778, -0.03834943, 0.99926064, 0.87957275],
                                      [0., 0., 0., 1.]])
        num_pts = 0
        if len(target_left_pc) > 0:
            target_left_point_cloud = target_left_pc[idx][:, :3]
            target_left_point_cloud = np.hstack([target_left_point_cloud, np.ones((len(target_left_point_cloud), 1))])
            # 应用旋转矩阵到 xyz
            ego_pose_world_to_car = np.linalg.inv(ego_pose)
            rotated_xyz = (target_left_point_cloud @ ego_pose_world_to_car.T @ (np.linalg.inv(left_trans_matrix)).T)[:, :3]
            # 或者使用 np.dot(xyz, rotation_matrix.T)
            # 保持 intensity 部分不变
            intensity = target_left_pc[idx][:, 3]
            # 重新组合为 N*4 的数组
            target_left_point_cloud = np.hstack((rotated_xyz, intensity[:, np.newaxis]))
            num_pts += target_left_point_cloud.shape[0]
            point_left_intensity = np.vstack((point_left_intensity, target_left_point_cloud))
        if len(target_right_pc) > 0:
            target_right_point_cloud = target_right_pc[idx][:, :3]
            target_right_point_cloud = np.hstack([target_right_point_cloud, np.ones((len(target_right_point_cloud), 1))])
            # 应用旋转矩阵到 xyz
            ego_pose_world_to_car = np.linalg.inv(ego_pose)
            rotated_xyz = (target_right_point_cloud @ ego_pose_world_to_car.T @ (np.linalg.inv(right_trans_matrix)).T)[:, :3]
            # 或者使用 np.dot(xyz, rotation_matrix.T)
            # 保持 intensity 部分不变
            intensity = target_right_pc[idx][:, 3]
            # 重新组合为 N*4 的数组
            target_right_point_cloud = np.hstack((rotated_xyz, intensity[:, np.newaxis]))
            point_right_intensity = np.vstack((point_right_intensity, target_right_point_cloud))
            num_pts += target_right_point_cloud.shape[0]
        if len(target_front_pc) > 0:
            target_front_point_cloud = target_front_pc[idx][:, :3]
            target_front_point_cloud = np.hstack([target_front_point_cloud, np.ones((len(target_front_point_cloud), 1))])
            # 应用旋转矩阵到 xyz
            ego_pose_world_to_car = np.linalg.inv(ego_pose)
            rotated_xyz = (target_front_point_cloud @ ego_pose_world_to_car.T @ (np.linalg.inv(front_trans_matrix)).T)[:, :3]
            # 或者使用 np.dot(xyz, rotation_matrix.T)
            # 保持 intensity 部分不变
            intensity = target_front_pc[idx][:, 3]
            # 重新组合为 N*4 的数组
            target_front_point_cloud = np.hstack((rotated_xyz, intensity[:, np.newaxis]))
            point_front_intensity = np.vstack((point_front_intensity, target_front_point_cloud))
            num_pts += target_front_point_cloud.shape[0]
        if len(target_back_pc) > 0:
            target_back_point_cloud = target_back_pc[idx][:, :3]
            target_back_point_cloud = np.hstack([target_back_point_cloud, np.ones((len(target_back_point_cloud), 1))])
            # 应用旋转矩阵到 xyz
            ego_pose_world_to_car = np.linalg.inv(ego_pose)
            rotated_xyz = (target_back_point_cloud @ ego_pose_world_to_car.T @ (np.linalg.inv(back_trans_matrix)).T)[:, :3]
            # 或者使用 np.dot(xyz, rotation_matrix.T)
            # 保持 intensity 部分不变
            intensity = target_back_pc[idx][:, 3]
            # 重新组合为 N*4 的数组
            target_back_point_cloud = np.hstack((rotated_xyz, intensity[:, np.newaxis]))
            point_back_intensity = np.vstack((point_back_intensity, target_back_point_cloud))
            num_pts += target_back_point_cloud.shape[0]
        transformed_lidar1_cloud = transform_point_cloud(point_left, left_trans_matrix)
        transformed_lidar2_cloud = transform_point_cloud(point_right, right_trans_matrix)
        transformed_front_cloud = transform_point_cloud(point_front, front_trans_matrix)
        transformed_back_cloud = transform_point_cloud(point_back, back_trans_matrix)
        point = np.vstack((transformed_lidar1_cloud, transformed_lidar2_cloud))
        point = np.vstack((point, transformed_front_cloud))
        point = np.vstack((point, transformed_back_cloud))
        # point = np.vstack((point, transformed_front_cloud))
        point_xyz = point
        point_xyz_world = (np.pad(point_xyz, (0, 1), constant_values=1) @ ego_pose.T)[:, :3]
        points.append(point_xyz_world)
        point_time = np.full_like(point_xyz_world[:, :1], timestamp)
        points_time.append(point_time)

        if args.require_boundingbox:
            # ego_pose_world_to_car = np.linalg.inv(ego_pose)
            # point_dynamic_1 = (np.pad(dynamic_pointcloud_1, (0, 1), constant_values=1) @ ego_pose_world_to_car.T)[:, :3]
            # point_dynamic_2 = (np.pad(dynamic_pointcloud_2, (0, 1), constant_values=1) @ ego_pose_world_to_car.T)[:, :3]

            # min_bound_1 = np.min(dynamic_pointcloud_1, axis=0)
            # max_bound_1 = np.max(dynamic_pointcloud_1, axis=0)

            # min_bound_2 = np.min(dynamic_pointcloud_2, axis=0)
            # max_bound_2 = np.max(dynamic_pointcloud_2, axis=0)

            # 筛选出 point_xyz_world 中在 dynamic_pointcloud_1 和 dynamic_pointcloud_2 范围内的点
            # mask_1 = np.all((point_xyz_world >= min_bound_1) & (point_xyz_world <= max_bound_1), axis=1)
            # mask_2 = np.all((point_xyz_world >= min_bound_2) & (point_xyz_world <= max_bound_2), axis=1)
            dynamic_mask_left = np.ones(len(transformed_lidar1_cloud), dtype=bool)
            dynamic_mask_right = np.ones(len(transformed_lidar2_cloud), dtype=bool)
            dynamic_mask_front = np.ones(len(transformed_front_cloud), dtype=bool)
            dynamic_mask_back = np.ones(len(transformed_back_cloud), dtype=bool)
            if os.path.exists(os.path.join(args.source_path, 'points3d_dynamic_0.ply')):
                for current_point in dynamic_pointcloud:
                    # 计算每个动态物体点与原始点云中点的距离，根据需要调整阈值
                    transformed_lidar1_cloud_homogeneous = np.hstack(
                        [transformed_lidar1_cloud, np.ones((len(transformed_lidar1_cloud), 1))])
                    # 进行变换
                    transformed_points = (transformed_lidar1_cloud_homogeneous @ ego_pose.T)[:, :3]
                    distances_left = np.linalg.norm(current_point - transformed_points, axis=1)
                    dynamic_mask_left[distances_left < 0.2] = False
                dynamic_indices_left = np.where(~dynamic_mask_left)[0]
                static_indices_left = np.where(dynamic_mask_left)[0]
                point_dynamic_left = np.hstack(((np.hstack(
                    [transformed_lidar1_cloud, np.ones((len(transformed_lidar1_cloud), 1))]) @ ego_pose.T)[:, :3],
                                                intensity_left[:, np.newaxis]))[dynamic_indices_left]
                point_static_left = np.hstack(((np.hstack(
                    [transformed_lidar1_cloud, np.ones((len(transformed_lidar1_cloud), 1))]) @ ego_pose.T)[:, :3],
                                                intensity_left[:, np.newaxis]))[static_indices_left]
                num_pts_left = point_dynamic_left.shape[0]
                print(num_pts_left)
                for current_point in dynamic_pointcloud:
                    # 计算每个动态物体点与原始点云中点的距离，根据需要调整阈值
                    transformed_lidar1_cloud_homogeneous = np.hstack(
                        [transformed_lidar2_cloud, np.ones((len(transformed_lidar2_cloud), 1))])
                    # 进行变换
                    transformed_points = (transformed_lidar1_cloud_homogeneous @ ego_pose.T)[:, :3]
                    distances_right = np.linalg.norm(current_point - transformed_points, axis=1)
                    dynamic_mask_right[distances_right < 0.2] = False
                dynamic_indices_right = np.where(~dynamic_mask_right)[0]
                static_indices_right = np.where(dynamic_mask_right)[0]
                point_dynamic_right = np.hstack(((np.hstack(
                    [transformed_lidar2_cloud, np.ones((len(transformed_lidar2_cloud), 1))]) @ ego_pose.T)[:, :3],
                                                 intensity_right[:, np.newaxis]))[dynamic_indices_right]
                point_static_right = np.hstack(((np.hstack(
                    [transformed_lidar2_cloud, np.ones((len(transformed_lidar2_cloud), 1))]) @ ego_pose.T)[:, :3],
                                                 intensity_right[:, np.newaxis]))[static_indices_right]
                num_pts_right = point_dynamic_right.shape[0]
                print(num_pts_right)
                for current_point in dynamic_pointcloud:
                    # 计算每个动态物体点与原始点云中点的距离，根据需要调整阈值
                    transformed_lidar1_cloud_homogeneous = np.hstack(
                        [transformed_front_cloud, np.ones((len(transformed_front_cloud), 1))])
                    # 进行变换
                    transformed_points = (transformed_lidar1_cloud_homogeneous @ ego_pose.T)[:, :3]
                    distances_front = np.linalg.norm(current_point - transformed_points, axis=1)
                    dynamic_mask_front[distances_front < 0.2] = False
                dynamic_indices_front = np.where(~dynamic_mask_front)[0]
                static_indices_front = np.where(dynamic_mask_front)[0]
                point_dynamic_front = np.hstack(((np.hstack(
                    [transformed_front_cloud, np.ones((len(transformed_front_cloud), 1))]) @ ego_pose.T)[:, :3],
                                                 intensity_front[:, np.newaxis]))[dynamic_indices_front]
                point_static_front = np.hstack(((np.hstack(
                    [transformed_front_cloud, np.ones((len(transformed_front_cloud), 1))]) @ ego_pose.T)[:, :3],
                                                 intensity_front[:, np.newaxis]))[static_indices_front]
                num_pts_front = point_dynamic_front.shape[0]
                print(num_pts_front)
                for current_point in dynamic_pointcloud:
                    # 计算每个动态物体点与原始点云中点的距离，根据需要调整阈值
                    transformed_lidar1_cloud_homogeneous = np.hstack(
                        [transformed_back_cloud, np.ones((len(transformed_back_cloud), 1))])
                    # 进行变换
                    transformed_points = (transformed_lidar1_cloud_homogeneous @ ego_pose.T)[:, :3]
                    distances_back = np.linalg.norm(current_point - transformed_points, axis=1)
                    dynamic_mask_back[distances_back < 0.2] = False
                dynamic_indices_back = np.where(~dynamic_mask_back)[0]
                static_indices_back = np.where(dynamic_mask_back)[0]
                point_dynamic_back = np.hstack(((np.hstack(
                    [transformed_back_cloud, np.ones((len(transformed_back_cloud), 1))]) @ ego_pose.T)[:, :3],
                                                intensity_back[:, np.newaxis]))[dynamic_indices_back]
                point_static_back = np.hstack(((np.hstack(
                    [transformed_back_cloud, np.ones((len(transformed_back_cloud), 1))]) @ ego_pose.T)[:, :3],
                                                intensity_back[:, np.newaxis]))[static_indices_back]
                num_pts_back = point_dynamic_back.shape[0]
                print(num_pts_back)
                if not os.path.exists(os.path.join(args.model_path, 'LiDAR', 'velodyne_0')):
                    os.makedirs(os.path.join(args.model_path, 'LiDAR', 'velodyne_0'))
                    os.makedirs(os.path.join(args.model_path, 'LiDAR', 'velodyne_1'))
                    os.makedirs(os.path.join(args.model_path, 'LiDAR', 'velodyne_2'))
                    os.makedirs(os.path.join(args.model_path, 'LiDAR', 'velodyne_3'))
                if num_pts_left > 0:
                    point_static_left.astype(np.float32).tofile(
                        os.path.join(args.model_path, 'LiDAR', 'velodyne_0', car_id + ".bin"))
                if num_pts_right > 0:
                    point_static_right.astype(np.float32).tofile(
                        os.path.join(args.model_path, 'LiDAR', 'velodyne_1', car_id + ".bin"))
                if num_pts_front > 0:
                    point_static_front.astype(np.float32).tofile(
                        os.path.join(args.model_path, 'LiDAR', 'velodyne_2', car_id + ".bin"))
                if num_pts_back > 0:
                    point_static_back.astype(np.float32).tofile(
                        os.path.join(args.model_path, 'LiDAR', 'velodyne_3', car_id + ".bin"))
                if not os.path.exists(os.path.join(args.source_path, 'dynamic_LiDAR', 'velodyne_0')):
                    os.makedirs(os.path.join(args.source_path, 'dynamic_LiDAR', 'velodyne_0'))
                    os.makedirs(os.path.join(args.source_path, 'dynamic_LiDAR', 'velodyne_1'))
                    os.makedirs(os.path.join(args.source_path, 'dynamic_LiDAR', 'velodyne_2'))
                    os.makedirs(os.path.join(args.source_path, 'dynamic_LiDAR', 'velodyne_3'))
                if num_pts_left > 0:
                    point_dynamic_left.astype(np.float32).tofile(
                        os.path.join(args.source_path, 'dynamic_LiDAR', 'velodyne_0', car_id + ".bin"))
                if num_pts_right > 0:
                    point_dynamic_right.astype(np.float32).tofile(
                        os.path.join(args.source_path, 'dynamic_LiDAR', 'velodyne_1', car_id + ".bin"))
                if num_pts_front > 0:
                    point_dynamic_front.astype(np.float32).tofile(
                        os.path.join(args.source_path, 'dynamic_LiDAR', 'velodyne_2', car_id + ".bin"))
                if num_pts_back > 0:
                    point_dynamic_back.astype(np.float32).tofile(
                        os.path.join(args.source_path, 'dynamic_LiDAR', 'velodyne_3', car_id + ".bin"))
            else:
                if not os.path.exists(os.path.join(args.model_path, 'LiDAR', 'velodyne_0')):
                    os.makedirs(os.path.join(args.model_path, 'LiDAR', 'velodyne_0'))
                    os.makedirs(os.path.join(args.model_path, 'LiDAR', 'velodyne_1'))
                    os.makedirs(os.path.join(args.model_path, 'LiDAR', 'velodyne_2'))
                    os.makedirs(os.path.join(args.model_path, 'LiDAR', 'velodyne_3'))
                point_left_intensity.astype(np.float32).tofile(
                    os.path.join(args.model_path, 'LiDAR', 'velodyne_0', car_id + ".bin"))
                point_right_intensity.astype(np.float32).tofile(
                    os.path.join(args.model_path, 'LiDAR', 'velodyne_1', car_id + ".bin"))
                point_front_intensity.astype(np.float32).tofile(
                    os.path.join(args.model_path, 'LiDAR', 'velodyne_2', car_id + ".bin"))
                point_back_intensity.astype(np.float32).tofile(
                    os.path.join(args.model_path, 'LiDAR', 'velodyne_3', car_id + ".bin"))
            bbox_size[0] = 5.2
            bbox_size[1] = 10.22
            bbox_size[2] = 4.736
            bbox_info_1 = [bbox_translation, bbox_size, bbox_rotation, num_pts]
        else:
            bbox_translation_1, bbox_size_1, bbox_rotation_1 = 0, 0, 0
            bbox_translation_2, bbox_size_2, bbox_rotation_2 = 0, 0, 0
            num_pts_1, num_pts_2 = 0, 0
            bbox_info_1 = [0, 0, 0, 0]
        # bbox_info_1 = [0, 0, 0, 0]
        # pointcloud_car = (np.pad(pointcloud1, ((0, 0), (0, 1)), constant_values=1) @ ego_pose_world_to_car.T)[:, :3]
        # distances = np.linalg.norm(pointcloud_car, axis=1)
        # # 筛选出距离小于等于15米的点
        # filtered_points = pointcloud_car[distances <= 15]
        # # 将筛选出的点与原始点云数据进行拼接
        # point_xyz = np.vstack((point_xyz, filtered_points))
        # 筛选出距离小于等于15米的点
        # 将筛选出的点与原始点云数据进行拼接
        # point_xyz = np.vstack((point_xyz, filtered_points))
        for j in range(args.cam_num):
            if args.is_training:
                point_camera = (np.pad(point_xyz, ((0, 0), (0, 1)), constant_values=1) @ lidar2cam[j].T)[:, :3]
            else:
                point_camera = point_xyz
            R = np.transpose(w2c[j, :3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[j, :3, 3]
            K = Ks[j]
            fx = float(K[0, 0])
            fy = float(K[1, 1])
            cx = float(K[0, 2])
            cy = float(K[1, 2])
            FovX = FovY = -1.0
            cam_infos.append(CameraInfo(uid=idx * 7 + j, R=R, T=T, FovY=FovY, FovX=FovX,
                                        image=images[j],
                                        image_path=image_paths[j], image_name=car_id,
                                        width=HWs[j][1], height=HWs[j][0], timestamp=timestamp,
                                        pointcloud_camera=point_camera,
                                        fx=fx, fy=fy, cx=cx, cy=cy,
                                        sky_mask=sky_masks[j],
                                        object_mask=masks_data[j] if masks_data is not None else None,
                                        dynamic_image=dynamic_images[j] if dynamic_images is not None else None,
                                        pointcloud_front=point_front_intensity, pointcloud_left=point_left_intensity,
                                        pointcloud_right=point_right_intensity, pointcloud_back=point_back_intensity,
                                        bbx_info_1=bbox_info_1))

        if args.debug_cuda:
            break

    # pointcloud_timestamp = np.concatenate(points_time, axis=0)
    # pointcloud_timestamp = pointcloud_timestamp[indices]

    w2cs = np.zeros((len(cam_infos), 4, 4))
    Rs = np.stack([c.R for c in cam_infos], axis=0)
    Ts = np.stack([c.T for c in cam_infos], axis=0)
    w2cs[:, :3, :3] = Rs.transpose((0, 2, 1))
    w2cs[:, :3, 3] = Ts
    w2cs[:, 3, 3] = 1
    c2ws = unpad_poses(np.linalg.inv(w2cs))
    c2ws, transform, scale_factor = transform_poses_pca(c2ws, fix_radius=args.fix_radius)
    np.savetxt(os.path.join(args.source_path, 'transform_matrix.txt'), transform, fmt='%.6f')
    print(scale_factor)
    # pos_shift = 0.5
    c2ws = pad_poses(c2ws)
    # ego_pose_world_to_car = np.linalg.inv(ego_pose)
    for idx, cam_info in enumerate(tqdm(cam_infos, desc="Transform data")):
        c2w = c2ws[idx]
        # c2w = transform @ c2w
        # if args.pos_shift is not None:
        #     c2w[..., 3] += args.pos_shift.to(c2w)
        w2c = np.linalg.inv(c2w)
        # if args.pos_rotation:
        #     rotation_vector = np.array(args.pos_rotation)  # 示例值，可以根据需要进行调整
        #     w2c = apply_rotation_to_all(w2c, rotation_vector)
        # if args.pos_shift:
        #     pos_shift = torch.FloatTensor(args.pos_shift) if args.pos_shift is not None else None
        #     w2c = torch.from_numpy(w2c)
        #     w2c[..., :3, 3] += pos_shift.to(w2c)
        #     w2c = w2c.detach().cpu().numpy()
        cam_info.R[:] = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
        cam_info.T[:] = w2c[:3, 3]
        # cam_info.pointcloud_camera[:] = (np.pad(pointcloud, (0, 1), constant_values=1) @ ego_pose_world_to_car.T)[:, :3]
        if args.is_training:
            cam_info.pointcloud_camera[:] *= scale_factor
    # pointcloud = (np.pad(pointcloud, ((0, 0), (0, 1)), constant_values=1) @ transform.T)[:, :3]
    if args.eval:
        # ## for snerf scene
        # train_cam_infos = [c for idx, c in enumerate(cam_infos) if (idx // cam_num) % testhold != 0]
        # test_cam_infos = [c for idx, c in enumerate(cam_infos) if (idx // cam_num) % testhold == 0]

        # for dynamic scene
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if (idx // args.cam_num + 1) % args.testhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if (idx // args.cam_num + 1) % args.testhold == 0]

        # for emernerf comparison [testhold::testhold]
        if args.testhold == 10:
            train_cam_infos = [c for idx, c in enumerate(cam_infos) if
                               (idx // args.cam_num) % args.testhold != 0 or (idx // args.cam_num) == 0]
            test_cam_infos = [c for idx, c in enumerate(cam_infos) if
                              (idx // args.cam_num) % args.testhold == 0 and (idx // args.cam_num) > 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    nerf_normalization['radius'] = 1 / nerf_normalization['radius']

    ply_path = os.path.join(args.source_path, "points3d.ply")
    # if not os.path.exists(ply_path):
    #     rgbs = np.random.random((pointcloud.shape[0], 3))
    #     storePly(ply_path, pointcloud, rgbs, pointcloud_timestamp)
    # try:
    #     pcd = fetchPly(ply_path)
    # except:
    #     pcd = None
    static_pointcloud = (np.pad(static_pointcloud, ((0, 0), (0, 1)), constant_values=1) @ transform.T)[:, :3]
    dynamic_pointcloud = (np.pad(dynamic_pointcloud, ((0, 0), (0, 1)), constant_values=1) @ transform.T)[:, :3]
    static_pcd = BasicPointCloud(static_pointcloud, colors=np.zeros([static_pointcloud.shape[0], 3]), normals=None,
                                 time=static_time)
    dynamic_pcd = BasicPointCloud(dynamic_pointcloud, colors=np.zeros([dynamic_pointcloud.shape[0], 3]), normals=None,
                                  time=dynamic_time)
    time_interval = (time_duration[1] - time_duration[0]) / (len(car_list) - 1)
    scene_info = SceneInfo(static_point_cloud=static_pcd,
                           dynamic_point_cloud=dynamic_pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           time_interval=time_interval,
                           time_duration=time_duration)

    return scene_info
