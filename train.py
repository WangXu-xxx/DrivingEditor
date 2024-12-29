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
import cv2
import json
import os
from collections import defaultdict
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from random import randint
from utils.loss_utils import psnr, ssim
from gaussian_renderer import render
from scene import Scene, GaussianModel, EnvLight, Dynamic_GaussianModel
from utils.general_utils import seed_everything, visualize_depth
from tqdm import tqdm
from argparse import ArgumentParser
from torchvision.utils import make_grid, save_image
import numpy as np
import kornia
from os import makedirs, path
from errno import EEXIST
from omegaconf import OmegaConf
import torch
import torch.nn as nn
from torchvision import models, transforms
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

EPS = 1e-5

class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()  # 目标特征是不需要梯度的

    def forward(self, input):
        return torch.nn.functional.mse_loss(input, self.target)

# 定义风格损失
class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = self.gram_matrix(target_feature).detach()

    def gram_matrix(self, input):
        a, b, c, d = input.size()
        features = input.view(a * b, c * d)
        G = torch.mm(features, features.t())
        return G.div(a * b * c * d)

    def forward(self, input):
        G = self.gram_matrix(input)
        return torch.nn.functional.mse_loss(G, self.target)

# 加载预训练的 VGG19 网络，使用其特征提取部分
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# vgg = models.vgg19(pretrained=True).features.to(device).eval()
# for param in vgg.parameters():
#     param.requires_grad = False


def get_features_and_losses(model, content_img):
    content_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4']  # 假设我们用第4个卷积层的输出作为内容特征

    content_losses = []

    x = content_img.clone()  # 对内容图像处理
    for name, layer in model._modules.items():
        x = layer(x)
        if name in content_layers:
            content_loss = ContentLoss(x)
            content_losses.append(content_loss(x))

    total_content_loss = sum(content_losses)
    return total_content_loss

def loss_cls_3d(features, predictions, k=5, lambda_val=2.0, max_points=200000, sample_size=800):
    """
    Compute the neighborhood consistency loss for a 3D point cloud using Top-k neighbors
    and the KL divergence.

    :param features: Tensor of shape (N, D), where N is the number of points and D is the dimensionality of the feature.
    :param predictions: Tensor of shape (N, C), where C is the number of classes.
    :param k: Number of neighbors to consider.
    :param lambda_val: Weighting factor for the loss.
    :param max_points: Maximum number of points for downsampling. If the number of points exceeds this, they are randomly downsampled.
    :param sample_size: Number of points to randomly sample for computing the loss.

    :return: Computed loss value.
    """
    # Conditionally downsample if points exceed max_points
    if features.size(0) > max_points:
        indices = torch.randperm(features.size(0))[:max_points]
        features = features[indices]
        predictions = predictions[indices]

    # Randomly sample points for which we'll compute the loss
    indices = torch.randperm(features.size(0))[:sample_size]
    sample_features = features[indices]
    sample_preds = predictions[indices]

    # Compute top-k nearest neighbors directly in PyTorch
    dists = torch.cdist(sample_features, features)  # Compute pairwise distances
    _, neighbor_indices_tensor = dists.topk(k, largest=False)  # Get top-k smallest distances

    # Fetch neighbor predictions using indexing
    neighbor_preds = predictions[neighbor_indices_tensor]

    # Compute KL divergence
    kl = sample_preds.unsqueeze(1) * (torch.log(sample_preds.unsqueeze(1) + 1e-10) - torch.log(neighbor_preds + 1e-10))
    loss = kl.sum(dim=-1).mean()

    # Normalize loss into [0, 1]
    num_classes = predictions.size(1)
    normalized_loss = loss / num_classes

    return lambda_val * normalized_loss

def tv_loss(image, dynamic_image, penalty_factor=10):
    # 计算 TV Loss
    if dynamic_image.mean(dim=0, keepdim=False).min() > 0.01:
        mask = dynamic_image.mean(dim=0, keepdim=False) > 0.01
        mask = mask.bool()

        # 确保掩码与图像的形状一致，选择掩码所对应的区域
        static = image[:, mask]
        brightness_threshold = static.mean()

        # 转换为 numpy
        image = image.detach().cpu().numpy()

        # 计算总变差（TV）损失
        h_variation = np.sum(np.abs(image[:, :-1, :] - image[:, 1:, :]))
        v_variation = np.sum(np.abs(image[:, :, :-1] - image[:, :, 1:]))
        tv_loss_value = h_variation + v_variation

        # 计算亮度惩罚
        brightness_penalty = np.sum(np.maximum(0, image - brightness_threshold))
        brightness_penalty *= penalty_factor

        # 计算总损失
        c, h, w = image.shape
        loss = tv_loss_value / (c * h * w) + brightness_penalty / (h * w)

        return loss
    else:
        loss = 0

    return loss


def calculate_weighted_loss(img_tensor, dynamic_image, threshold=0.9):
    # 假设 img_tensor 是 [C, H, W] 形状的图像张量，并且已经被归一化到 [0, 1]

    # 确定白色区域
    if dynamic_image.mean().max() > threshold:
        # 确定白色区域
        white_mask = dynamic_image.mean(dim=0) > threshold  # 形状为 [H, W]

        # 使用膨胀操作找到白色区域的边界
        kernel_size = 3
        dilated_mask = F.max_pool2d(white_mask.float().unsqueeze(0).unsqueeze(0), kernel_size, stride=1, padding=1)
        dilated_mask = dilated_mask.squeeze()

        # 边界是膨胀区域减去原始白色区域
        border_mask = dilated_mask > white_mask.float()

        # 计算距离变换，1 - 白色区域作为输入
        distance_transform = cv2.distanceTransform(1 - white_mask.cpu().numpy().astype(np.uint8), cv2.DIST_L2, 5)
        distance_transform = torch.from_numpy(distance_transform).cuda()
        # 归一化距离，并创建一个衰减函数，使得距离越大权重越小
        distance_weights = torch.exp(-distance_transform / distance_transform.max() * 5)

        # 找到边界像素的平均值
        border_pixels = img_tensor[:, border_mask]  # 选择边界像素
        border_avg = border_pixels.mean(dim=1, keepdim=True)  # 计算平均值

        # 计算损失：白色区域像素与边界平均值之间的加权MSE
        white_pixels = img_tensor[:, white_mask]
        loss = (distance_weights[white_mask] * (white_pixels - border_avg.expand_as(white_pixels)) ** 2).mean()
    else:
        loss = torch.tensor(0.0, device=img_tensor.device)

    return loss

def convert_to_binary_mask(dynamic_image):
    binary_mask = (dynamic_image[0, :, :] > 0).float()
    return binary_mask

def extract_object_pixels(tensor, mask):
    return tensor * mask

def training(args):

    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        tb_writer = None
        print("Tensorboard not available: not logging progress")
    vis_path = os.path.join(args.model_path, 'visualization')
    os.makedirs(vis_path, exist_ok=True)

    gaussians = GaussianModel(args)
    if args.including_dynamic:
        dynamic_gaussians = Dynamic_GaussianModel(args)
    else:
        dynamic_gaussians = None
    scene = Scene(args, gaussians, dynamic_gaussians)
    gaussians.training_setup(args)
    if args.including_dynamic:
        dynamic_gaussians.training_setup(args)

    # classifier = torch.nn.Conv2d(gaussians.num_objects, 128, kernel_size=1)
    # cls_criterion = torch.nn.CrossEntropyLoss(reduction='none')
    # cls_optimizer = torch.optim.Adam(classifier.parameters(), lr=5e-4)
    # classifier.cuda()

    if args.env_map_res > 0:
        env_map = EnvLight(resolution=args.env_map_res).cuda()
        env_map.training_setup(args)
    else:
        env_map = None

    first_iter = 0
    if args.start_checkpoint:
        (model_params, first_iter) = torch.load(args.start_checkpoint)
        gaussians.restore(model_params, args)
        dynamic_gaussians.restore(model_params, args)

        if env_map is not None:
            env_checkpoint = os.path.join(os.path.dirname(args.checkpoint),
                                        os.path.basename(args.checkpoint).replace("chkpnt", "env_light_chkpnt"))
            (light_params, _) = torch.load(env_checkpoint)
            env_map.restore(light_params)

    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None

    ema_dict_for_log = defaultdict(int)
    progress_bar = tqdm(range(first_iter + 1, args.iterations + 1), desc="Training progress")

    for iteration in progress_bar:
        iter_start.record()
        gaussians.update_learning_rate(iteration)
        if dynamic_gaussians is not None:
            dynamic_gaussians.update_learning_rate(iteration)
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % args.sh_increase_interval == 0:
            gaussians.oneupSHdegree()
            if dynamic_gaussians is not None:
                dynamic_gaussians.oneupSHdegree()
        if not viewpoint_stack:
            viewpoint_stack = list(range(len(scene.getTrainCameras())))
        viewpoint_cam = scene.getTrainCameras()[viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))]

        # render v and t scale map
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

        if np.random.random() < args.lambda_self_supervision:
            time_shift = 3*(np.random.random() - 0.5) * scene.time_interval
        else:
            time_shift = None
        # time_shift = None
        render_pkg = render(viewpoint_cam, gaussians, dynamic_gaussians, args, background, env_map=env_map, other=other,
                            dynamic_other=dynamic_other, time_shift=time_shift, is_training=True)

        gt_image = viewpoint_cam.original_image.cuda()
        image = render_pkg["render"]
        depth = render_pkg["depth"]
        alpha = render_pkg["alpha"]
        # object = render_pkg["render_objects"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]
        if dynamic_gaussians is not None:
            gt_dynamic_image = viewpoint_cam.dynamic_image.cuda() if viewpoint_cam.dynamic_image is not None else None
            dynamic_image = render_pkg["render_d"]
            static_image = render_pkg["render_s"]
            static_alpha = render_pkg["static_alpha"]
            dynamic_alpha = render_pkg["dynamic_alpha"]
            static_feature = render_pkg['static_feature'] / alpha.clamp_min(EPS)
            static_t_map = static_feature[0:1]
            static_v_map = static_feature[1:]
            dynamic_mask = convert_to_binary_mask(viewpoint_cam.dynamic_image.cuda())
            dynamic_image = extract_object_pixels(dynamic_image, dynamic_mask)
            dynamic_image_1 = extract_object_pixels(image, dynamic_mask)
            gt_dynamic_image = extract_object_pixels(gt_dynamic_image, dynamic_mask)
            bg_color_from_envmap_s = render_pkg["env_map_s"]
            bg_color_from_envmap = render_pkg["env_map"]
            # dynamic_alpha = extract_object_pixels(dynamic_alpha, dynamic_mask)

        log_dict = {}
        feature = render_pkg['feature'] / alpha.clamp_min(EPS)
        t_map = feature[0:1]
        v_map = feature[1:]

        sky_mask = viewpoint_cam.sky_mask.cuda() if viewpoint_cam.sky_mask is not None else torch.zeros_like(alpha, dtype=torch.bool)

        sky_depth = 900
        depth = depth / alpha.clamp_min(EPS)

        # gt_obj = viewpoint_cam.objects.cuda().long() if viewpoint_cam.objects is not None else None
        # print(gt_obj.max())
        # torch.backends.cudnn.enabled = False
        # logits = classifier(object)
        # loss_obj = cls_criterion(logits.unsqueeze(0), gt_obj.unsqueeze(0)).squeeze().mean()
        # loss_obj = loss_obj / torch.log(torch.tensor(128))

        if env_map is not None:
            if args.depth_blend_mode == 0:  # harmonic mean
                depth = 1 / (alpha / depth.clamp_min(EPS) + (1 - alpha) / sky_depth).clamp_min(EPS)
            elif args.depth_blend_mode == 1:
                depth = alpha * depth + (1 - alpha) * sky_depth

        loss_l1 = F.l1_loss(image, gt_image)
        log_dict['loss_l1'] = loss_l1.item()
        loss_ssim = 1.0 - ssim(image, gt_image)
        log_dict['loss_ssim'] = loss_ssim.item()
        sky_mask_1 = convert_to_binary_mask(viewpoint_cam.sky_mask.long().cuda())
        sky_truth = extract_object_pixels(gt_image, sky_mask_1)
        if dynamic_gaussians is not None:
            if gt_dynamic_image.mean().max() > 0:
                static_image_1 = extract_object_pixels(static_image, 1 - dynamic_mask)
                static_truth = extract_object_pixels(gt_image, 1 - dynamic_mask)
                sky_image = extract_object_pixels(static_image, sky_mask_1)
                loss_ssim_d = 1.0 - ssim(dynamic_image, gt_dynamic_image)
                loss_ssim_sky = 1.0 - ssim(sky_image, sky_truth)
                loss_ssim_static = 1.0 - ssim(static_image_1, static_truth)
                loss_l1_d = F.l1_loss(dynamic_image, gt_dynamic_image)
                loss_l1_sky = F.l1_loss(sky_image, sky_truth)
                loss_l1_static = F.l1_loss(static_image_1, static_truth)
                loss_opacity = calculate_weighted_loss(static_alpha, gt_dynamic_image, 0.00001)
                loss_v = calculate_weighted_loss(static_v_map, gt_dynamic_image, 0.00001)
                # print(loss_dynamic)
                loss = ((1.0 - args.lambda_dssim) * loss_l1 + args.lambda_dssim * loss_ssim
                        + 5 * ((1.0 - args.lambda_dssim) * loss_l1_d + args.lambda_dssim * loss_ssim_d)
                        + 5 * ((1.0 - args.lambda_dssim) * loss_l1_sky + args.lambda_dssim * loss_ssim_sky)
                        + ((1.0 - args.lambda_dssim) * loss_l1_static + args.lambda_dssim * loss_ssim_static))
                # loss += loss_opacity
                # loss += loss_v
            else:
                sky_image = extract_object_pixels(image, sky_mask_1)
                loss_ssim_sky = 1.0 - ssim(sky_image, sky_truth)
                loss_l1_sky = F.l1_loss(sky_image, sky_truth)
                loss = ((1.0 - args.lambda_dssim) * loss_l1 + args.lambda_dssim * loss_ssim +
                        5 * ((1.0 - args.lambda_dssim) * loss_l1_sky + args.lambda_dssim * loss_ssim_sky))
        else:
            sky_image = extract_object_pixels(image, sky_mask_1)
            loss_ssim_sky = 1.0 - ssim(sky_image, sky_truth)
            loss_l1_sky = F.l1_loss(sky_image, sky_truth)
            loss = ((1.0 - args.lambda_dssim) * loss_l1 + args.lambda_dssim * loss_ssim +
                    5 * ((1.0 - args.lambda_dssim) * loss_l1_sky + args.lambda_dssim * loss_ssim_sky))
        # alpha_1 = alpha.clone().detach()
        # loss_opacity = F.mse_loss(static_alpha, alpha_1)
        # total_content_loss = tv_loss(static_image, dynamic_image, 10)
        # print(total_content_loss)
        # loss += total_content_loss
        # if iteration % args.reg3d_interval == 0:
        #     # regularize at certain intervals
        #     logits3d = classifier(gaussians._objects_dc.permute(2,0,1))
        #     prob_obj3d = torch.softmax(logits3d,dim=0).squeeze().permute(1,0)
        #     loss_obj_3d = loss_cls_3d(gaussians._xyz.squeeze().detach(), prob_obj3d, args.reg3d_k, args.reg3d_lambda_val, args.reg3d_max_points, args.reg3d_sample_size)
        #     loss += loss_obj + loss_obj_3d
        #     log_dict['loss_obj'] = (loss_obj + loss_obj_3d).item()
        # else:
        #     loss += loss_obj
        #     log_dict['loss_obj'] = loss_obj.item()

        if args.lambda_lidar > 0:
            assert viewpoint_cam.pts_depth is not None
            pts_depth = viewpoint_cam.pts_depth.cuda()

            mask = pts_depth > 0
            loss_lidar = torch.abs(1 / (pts_depth[mask] + 1e-5) - 1 / (depth[mask] + 1e-5)).mean()
            if args.lidar_decay > 0:
                iter_decay = np.exp(-iteration / 8000 * args.lidar_decay)
            else:
                iter_decay = 1
            log_dict['loss_lidar'] = loss_lidar.item()
            loss += iter_decay * args.lambda_lidar * loss_lidar

        if args.lambda_t_reg > 0:
            loss_t_reg = 1/torch.abs(static_t_map).mean()
            log_dict['loss_t_reg'] = loss_t_reg.item()
            loss += args.lambda_t_reg * loss_t_reg

        if args.lambda_v_reg > 0:
            loss_v_reg = torch.abs(static_v_map).mean()
            log_dict['loss_v_reg'] = loss_v_reg.item()
            loss += args.lambda_v_reg * loss_v_reg

        if args.lambda_inv_depth > 0:
            inverse_depth = 1 / (depth + 1e-5)
            loss_inv_depth = kornia.losses.inverse_depth_smoothness_loss(inverse_depth[None], gt_image[None])
            log_dict['loss_inv_depth'] = loss_inv_depth.item()
            loss = loss + args.lambda_inv_depth * loss_inv_depth

        if args.lambda_v_smooth > 0:
            loss_v_smooth = kornia.losses.inverse_depth_smoothness_loss(v_map[None], gt_image[None])
            log_dict['loss_v_smooth'] = loss_v_smooth.item()
            loss = loss + args.lambda_v_smooth * loss_v_smooth

        if args.lambda_sky_opa > 0:
            o = alpha.clamp(1e-6, 1-1e-6)
            sky = sky_mask.float()
            loss_sky_opa = (-sky * torch.log(1 - o)).mean()
            log_dict['loss_sky_opa'] = loss_sky_opa.item()
            loss = loss + args.lambda_sky_opa * loss_sky_opa

        if args.lambda_opacity_entropy > 0:
            o = alpha.clamp(1e-6, 1 - 1e-6)
            loss_opacity_entropy = -(o*torch.log(o)).mean()
            log_dict['loss_opacity_entropy'] = loss_opacity_entropy.item()
            loss = loss + args.lambda_opacity_entropy * loss_opacity_entropy

        if args.lambda_opacity_entropy > 0 and args.including_dynamic:
            o = dynamic_alpha.clamp(1e-6, 1 - 1e-6)
            target = torch.ones_like(dynamic_alpha)  # 目标值为1
            target = target * dynamic_mask  # 目标值为1
            loss_convergence_to_one = nn.MSELoss()(dynamic_alpha, target)
            dynamic_alpha = dynamic_alpha * dynamic_mask
            # loss_convergence_to_one = -(dynamic_mask * (1-o) * torch.log(o)).sum() / dynamic_mask.sum()
            # loss_convergence_to_one += nn.MSELoss()(static_alpha_1, target_1)
            # log_dict['loss_dynamic_opacity_entropy'] = loss_dynamic_opacity_entropy.item()
            loss = loss + loss_convergence_to_one * 100

        # torch.autograd.set_detect_anomaly(True)
        loss.backward()
        log_dict['loss'] = loss.item()

        iter_end.record()

        with torch.no_grad():
            psnr_for_log = psnr(image, gt_image).double()
            log_dict["psnr"] = psnr_for_log
            for key in ['loss', "loss_l1", "psnr"]:
                ema_dict_for_log[key] = 0.4 * log_dict[key] + 0.6 * ema_dict_for_log[key]

            if iteration % 10 == 0:
                postfix = {k[5:] if k.startswith("loss_") else k:f"{ema_dict_for_log[k]:.{5}f}" for k, v in ema_dict_for_log.items()}
                postfix["scale"] = scene.resolution_scales[scene.scale_index]
                progress_bar.set_postfix(postfix)
            if iteration % 1000 == 0:
                print(gaussians.get_xyz.shape[0])
                if dynamic_gaussians is not None:
                    print(dynamic_gaussians.get_xyz.shape[0])

                # print(log_dict['loss_lidar'], log_dict['loss_v_reg'], log_dict['loss_inv_depth']
                #       , log_dict['loss_sky_opa'], log_dict['loss_opacity_entropy'])
                # print(log_dict['loss_t_reg'])
                # print(log_dict['loss_v_reg'])
                # print(log_dict['loss_inv_depth'])
                # print(log_dict['loss_v_smooth'])
                # print(log_dict['loss_sky_opa'])
                # print(log_dict['loss_opacity_entropy'])
            log_dict['iter_time'] = iter_start.elapsed_time(iter_end)
            log_dict['total_points'] = gaussians.get_xyz.shape[0]
            # Log and save
            complete_eval(tb_writer, iteration, args.test_iterations, scene, render, (args, background),
                          log_dict, env_map=env_map, including_dynamic=args.including_dynamic)

            # Densification
            if iteration > args.densify_until_iter * args.time_split_frac:
                gaussians.no_time_split = False

            end_idx = gaussians.get_xyz.shape[0]
            cur_viewspace_point_tensor = slice_with_grad(viewspace_point_tensor, 0, end_idx)
            densification_and_optimization(gaussians, args, iteration, cur_viewspace_point_tensor,
                                           visibility_filter[:end_idx], scene, radii[:end_idx], dynamic=False)
            if dynamic_gaussians is not None:
                start_idx = end_idx
                    # Optimize box gaussians
                idx_length = dynamic_gaussians.get_xyz.shape[0]
                cur_viewspace_point_tensor = slice_with_grad(viewspace_point_tensor, start_idx,
                                                             start_idx + idx_length)
                densification_and_optimization(dynamic_gaussians,
                                               args,
                                               iteration,
                                               cur_viewspace_point_tensor,
                                               visibility_filter[start_idx:start_idx + idx_length],
                                               scene,
                                               radii[start_idx:start_idx + idx_length],
                                               dynamic=True)
                start_idx += idx_length
                # Keep track of max radii in image-space for pruning
                # gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                # gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                # if iteration > args.densify_from_iter and iteration % args.densification_interval == 0:
                #     size_threshold = args.size_threshold if (iteration > args.opacity_reset_interval and args.prune_big_point > 0) else None
                #
                #     if size_threshold is not None:
                #         size_threshold = size_threshold // scene.resolution_scales[0]
                #
                #     gaussians.densify_and_prune(args.densify_grad_threshold, args.thresh_opa_prune, scene.cameras_extent, size_threshold, args.densify_grad_t_threshold)
                #     dynamic_gaussians.densify_and_prune(args.densify_grad_threshold, args.thresh_opa_prune, scene.cameras_extent, size_threshold, args.densify_grad_t_threshold)
                #
                # if iteration % args.opacity_reset_interval == 0 or (args.white_background and iteration == args.densify_from_iter):
                #     gaussians.reset_opacity()
                #     dynamic_gaussians.reset_opacity()


            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)
            if dynamic_gaussians is not None:
                dynamic_gaussians.optimizer.step()
                dynamic_gaussians.optimizer.zero_grad(set_to_none=True)
            # cls_optimizer.step()
            # cls_optimizer.zero_grad()
            if env_map is not None and iteration < args.env_optimize_until:
                env_map.optimizer.step()
                env_map.optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            if dynamic_gaussians is not None:
                if iteration % args.vis_step == 0 or iteration == 1:
                    other_img = []
                    feature = render_pkg['feature'] / alpha.clamp_min(1e-5)
                    t_map = feature[0:1]
                    v_map = feature[1:]
                    v_norm_map = v_map.norm(dim=0, keepdim=True)
                    static_v_norm_map = static_v_map.norm(dim=0, keepdim=True)
                    et_color = visualize_depth(t_map, near=0.01, far=1)
                    static_v_color = visualize_depth(static_v_norm_map, near=0.01, far=1)
                    static_et_color = visualize_depth(static_t_map, near=0.01, far=1)

                    v_color = visualize_depth(v_norm_map, near=0.01, far=1)
                    dynamic_opacity_color = visualize_depth(dynamic_alpha, near=0.01, far=1)
                    static_opacity_color = visualize_depth(static_alpha, near=0.01, far=1)
                    opacity_color = visualize_depth(alpha, near=0.01, far=1)
                    other_img.append(et_color)
                    other_img.append(static_v_color)
                    other_img.append(v_color)
                    other_img.append(dynamic_opacity_color)
                    other_img.append(static_opacity_color)
                    other_img.append(opacity_color)

                    if viewpoint_cam.pts_depth is not None:
                        depth_vis = visualize_depth(depth)
                        other_img.append(depth_vis)
                        other_img.append(bg_color_from_envmap)
                        other_img.append(bg_color_from_envmap_s)
                        pts_depth_vis = visualize_depth(viewpoint_cam.pts_depth)
                        other_img.append(pts_depth_vis)
                    grid = make_grid([
                        image,
                        gt_image,
                        dynamic_image,
                        static_image,
                        static_et_color,
                    ] + other_img, nrow=4)

                    save_image(grid, os.path.join(vis_path, f"{iteration:05d}_{viewpoint_cam.colmap_id:03d}.png"))
            else:
                if iteration % args.vis_step == 0 or iteration == 1:
                    other_img = []
                    feature = render_pkg['feature'] / alpha.clamp_min(1e-5)
                    t_map = feature[0:1]
                    v_map = feature[1:]
                    v_norm_map = v_map.norm(dim=0, keepdim=True)
                    opacity_color = visualize_depth(alpha, near=0.01, far=1)
                    et_color = visualize_depth(t_map, near=0.01, far=1)

                    v_color = visualize_depth(v_norm_map, near=0.01, far=1)
                    other_img.append(et_color)
                    other_img.append(v_color)

                    if viewpoint_cam.pts_depth is not None:
                        pts_depth_vis = visualize_depth(viewpoint_cam.pts_depth)
                        other_img.append(pts_depth_vis)

                    grid = make_grid([
                                         image,
                                         gt_image,
                                         alpha.repeat(3, 1, 1),
                                         torch.logical_not(sky_mask[:1]).float().repeat(3, 1, 1),
                                         visualize_depth(depth),
                                     ] + other_img, nrow=4)

                    save_image(grid, os.path.join(vis_path, f"{iteration:05d}_{viewpoint_cam.colmap_id:03d}.png"))
            if iteration % args.scale_increase_interval == 0:
                scene.upScale()

            if iteration in args.checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                if dynamic_gaussians is not None:
                    torch.save((dynamic_gaussians.capture(), iteration), scene.model_path + "/dynamic_chkpnt" + str(iteration) + ".pth")
                torch.save((env_map.capture(), iteration), scene.model_path + "/env_light_chkpnt" + str(iteration) + ".pth")
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                # scene.save(iteration, viewpoint_cam.timestamp, time_shift)
                point_cloud_path = os.path.join(scene.model_path, "point_cloud/iteration_{}".format(iteration))
                makedirs(point_cloud_path, exist_ok=True)
                # torch.save(classifier.state_dict(), os.path.join(scene.model_path, "point_cloud/iteration_{}".format(iteration),'classifier.pth'))

# def mkdir_p(folder_path):
#     # Creates a directory. equivalent to using mkdir -p on the command line
#     try:
#         makedirs(folder_path)
#     except OSError as exc:  # Python >2.5
#         if exc.errno == EEXIST and path.isdir(folder_path):
#             pass
#         else:
#             raisepython train.py --config configs/waymo_reconstruction.yaml source_path=/mnt/data/data/waymo_scenes/0147030/ model_path=/mnt/data/data/waymo_scenes/0147030/output/eval_output_4 including_dynamic=true

def slice_with_grad(tensor, start, end):
    out = tensor[start:end]
    out.grad = tensor.grad[start:end]
    return out

def densification_and_optimization(gaussians, args, iteration, viewspace_point_tensor, visibility_filter, scene,
                                   radii, dynamic):
    # Densification
        # Keep track of max radii in image-space for pruning
    if dynamic:
        condition = (iteration < args.dynamic_densify_until_iter and (
            args.densify_until_num_points < 0 or gaussians.get_xyz.shape[0] < args.dynamic_densify_until_num_points))
    else:
        condition = (iteration < args.densify_until_iter and (
            args.densify_until_num_points < 0 or gaussians.get_xyz.shape[0] < args.densify_until_num_points))
    if condition:
        gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                             radii[visibility_filter])
        gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

        if iteration > args.densify_from_iter and iteration % args.densification_interval == 0:
            size_threshold = 20 if iteration > args.opacity_reset_interval else None
            densify_grad_threshold = args.densify_grad_threshold
            if dynamic:
                densify_grad_threshold *= 0.5
                if size_threshold is not None:
                    size_threshold *= 0.5
            # do_prune = (iteration < cfg_sd.start_guiding_from_iter) and cfg_sd.do_prune
            # gaussians.densify_and_prune(densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, prune=do_prune)
            gaussians.densify_and_prune(densify_grad_threshold, args.thresh_opa_prune, scene.cameras_extent, size_threshold, args.densify_grad_t_threshold)

    if (iteration % args.opacity_reset_interval == 0 or (
            args.white_background and iteration == args.densify_from_iter)) and not dynamic:
        gaussians.reset_opacity()


def complete_eval(tb_writer, iteration, test_iterations, scene : Scene, renderFunc, renderArgs, log_dict,
                  env_map=None, including_dynamic=False):
    from lpipsPyTorch import lpips
    args = renderArgs[0]
    if tb_writer:
        for key, value in log_dict.items():
            tb_writer.add_scalar(f'train/{key}', value, iteration)

    if iteration in test_iterations:
        scale = scene.resolution_scales[scene.scale_index]
        if iteration < args.iterations:
            validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras(scale=scale)},)
        else:
            if "kitti" in args.model_path:
                # follow NSG: https://github.com/princeton-computational-imaging/neural-scene-graphs/blob/8d3d9ce9064ded8231a1374c3866f004a4a281f8/data_loader/load_kitti.py#L766
                num = len(scene.getTrainCameras())//2
                eval_train_frame = num//5
                traincamera = sorted(scene.getTrainCameras(), key =lambda x: x.colmap_id)
                validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras(scale=scale)},
                                    {'name': 'train', 'cameras': traincamera[:num][-eval_train_frame:]+traincamera[num:][-eval_train_frame:]})
            else:
                validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras(scale=scale)},
                                {'name': 'train', 'cameras': scene.getTrainCameras()})



        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                outdir = os.path.join(args.model_path, "eval", config['name'] + f"_{iteration}" + "_render")
                os.makedirs(outdir,exist_ok=True)
                for idx, viewpoint in enumerate(tqdm(config['cameras'])):
                    if including_dynamic:
                        render_pkg = render(viewpoint, scene.gaussians, scene.dynamic_gaussians, *renderArgs,
                                            env_map=env_map)
                        image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                        depth = render_pkg['depth']
                        alpha = render_pkg['alpha']
                        static_image = render_pkg['render_s']
                        dynamic_image = render_pkg['render_d']
                        sky_depth = 900
                        depth = depth / alpha.clamp_min(EPS)
                        if env_map is not None:
                            if args.depth_blend_mode == 0:  # harmonic mean
                                depth = 1 / (alpha / depth.clamp_min(EPS) + (1 - alpha) / sky_depth).clamp_min(EPS)
                            elif args.depth_blend_mode == 1:
                                depth = alpha * depth + (1 - alpha) * sky_depth

                        depth = visualize_depth(depth)
                        alpha = alpha.repeat(3, 1, 1)

                        grid = [gt_image, image, static_image, dynamic_image]
                        grid = make_grid(grid, nrow=2)

                        save_image(grid, os.path.join(outdir, f"{viewpoint.colmap_id:03d}.png"))
                    else:
                        render_pkg = render(viewpoint, scene.gaussians, None, *renderArgs,
                                            env_map=env_map)
                        image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                        depth = render_pkg['depth']
                        alpha = render_pkg['alpha']
                        sky_depth = 900
                        depth = depth / alpha.clamp_min(EPS)
                        if env_map is not None:
                            if args.depth_blend_mode == 0:  # harmonic mean
                                depth = 1 / (alpha / depth.clamp_min(EPS) + (1 - alpha) / sky_depth).clamp_min(EPS)
                            elif args.depth_blend_mode == 1:
                                depth = alpha * depth + (1 - alpha) * sky_depth

                        depth = visualize_depth(depth)
                        depth = alpha.repeat(3, 1, 1)

                        grid = [gt_image, image, depth, depth]
                        grid = make_grid(grid, nrow=2)

                        save_image(grid, os.path.join(outdir, f"{viewpoint.colmap_id:03d}.png"))

                    l1_test += F.l1_loss(image, gt_image).double()
                    psnr_test += psnr(image, gt_image).double()
                    ssim_test += ssim(image, gt_image).double()
                    lpips_test += lpips(image, gt_image, net_type='vgg').double()  # very slow

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])

                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {}".format(iteration, config['name'],
                                                                                         l1_test, psnr_test,
                                                                                         ssim_test, lpips_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                with open(os.path.join(outdir, "metrics.json"), "w") as f:
                    json.dump({"split": config['name'], "iteration": iteration, "psnr": psnr_test.item(),
                               "ssim": ssim_test.item(), "lpips": lpips_test.item()}, f)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--base_config", type=str, default = "configs/base.yaml")
    args, _ = parser.parse_known_args()

    base_conf = OmegaConf.load(args.base_config)
    second_conf = OmegaConf.load(args.config)
    cli_conf = OmegaConf.from_cli()
    args = OmegaConf.merge(base_conf, second_conf, cli_conf)
    print(args)

    args.save_iterations.append(args.iterations)
    args.checkpoint_iterations.append(args.iterations)
    args.test_iterations.append(args.iterations)

    if args.exhaust_test:
        args.test_iterations += [i for i in range(0,args.iterations, args.test_interval)]

    print("Optimizing " + args.model_path)
    torch.cuda.set_device(torch.device("cuda:0"))
    seed_everything(args.seed)
    training(args)

    # All done
    print("\nTraining complete.")
