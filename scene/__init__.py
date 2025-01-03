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

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.gaussian_model import GaussianModel, Dynamic_GaussianModel
from scene.envlight import EnvLight
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from scene.waymo_loader import readWaymoInfo
from scene.waymo_loader2 import readWaymoInfo_add
from scene.kittimot_loader import readKittiMotInfo



class Scene:

    gaussians : GaussianModel
    dynamic_gaussians : Dynamic_GaussianModel

    def __init__(self, args, gaussians : GaussianModel, dynamic_gaussians: Dynamic_GaussianModel, load_iteration=None, shuffle=True):
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        if dynamic_gaussians is not None:
            self.dynamic_gaussians = dynamic_gaussians
        self.white_background = args.white_background

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        sceneLoadTypeCallbacks = {
            "Waymo": readWaymoInfo if args.nuscenes and not args.training else readWaymoInfo_add,
            "KittiMot": readKittiMotInfo,
        }
        scene_info = sceneLoadTypeCallbacks[args.scene_type](args)
        
        self.time_interval = args.frame_interval
        self.gaussians.time_duration = scene_info.time_duration
        if dynamic_gaussians is not None:
            self.dynamic_gaussians.time_duration = scene_info.time_duration
        print("time duration: ", scene_info.time_duration)
        print("frame interval: ", self.time_interval)

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        self.resolution_scales = args.resolution_scales
        self.scale_index = len(self.resolution_scales) - 1
        for resolution_scale in self.resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
            
        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                        "point_cloud",
                                                        "iteration_" + str(self.loaded_iter),
                                                        "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.static_point_cloud, 1)
            if dynamic_gaussians is not None:
                self.dynamic_gaussians.initialize(scene_info.dynamic_point_cloud, 1)

    def save(self, iteration, time, time_shift):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"), time, time_shift)
        # self.gaussians.save_deformation(point_cloud_path)

    def upScale(self):
        self.scale_index = max(0, self.scale_index - 1)

    def getTrainCameras(self):
        return self.train_cameras[self.resolution_scales[self.scale_index]]
    
    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

