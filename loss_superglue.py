import torch
from torch import nn as nn
from torchvision.transforms import Grayscale, Resize
from models.matching import Matching
from models.utils_superglue import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics)
from utils import quat2mat, rotate_forward, tvector2mat
from quaternion_distances import quaternion_distance
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_2D_lidar_projection(pcl, cam_intrinsic):
    pcl_xyz = cam_intrinsic @ pcl.T
    pcl_xyz = pcl_xyz.T
    pcl_z = pcl_xyz[:, 2]
    pcl_xyz = pcl_xyz / (pcl_xyz[:, 2, None] + 1e-10)
    pcl_uv = pcl_xyz[:, :2]
    return pcl_uv, pcl_z

def lidar_project_depth(pc_rotated, cam_calib, img_shape):
    pc_rotated = pc_rotated[:, :3].detach().cpu().numpy()
    cam_intrinsic = cam_calib
    pcl_uv, pcl_z = get_2D_lidar_projection(pc_rotated, cam_intrinsic)

    mask = (pcl_uv[:, 0] > 0) & (pcl_uv[:, 0] < img_shape[1]) & (pcl_uv[:, 1] > 0) & (
            pcl_uv[:, 1] < img_shape[0]) #& (pcl_z < 0)

    pcl_uv = pcl_uv[mask]
    pcl_z = pcl_z[mask]
    pcl_uv = pcl_uv.astype(np.uint32)
    pcl_z = pcl_z.reshape(-1, 1)
    depth_img = np.zeros((img_shape[0], img_shape[1], 1))
    depth_img[pcl_uv[:, 1], pcl_uv[:, 0]] = pcl_z
    depth_img = torch.from_numpy(depth_img.astype(np.float32))
    depth_img = depth_img.cuda()
    depth_img = depth_img.permute(2, 0, 1)

    return depth_img, pcl_uv

class SGLoss(nn.Module):
    def __init__(self,rescale_trans, rescale_rot):
        super(SGLoss,self).__init__()
        self.rescale_trans = rescale_trans
        self.rescale_rot = rescale_rot
        self.transl_loss = nn.SmoothL1Loss(reduction='none')
        self.K = np.array([[1.82146e3, 0, 9.44721e2],
							[0, 1.817312e3, 5.97134e2],
							[0,0,1]])
        config = {
            'superpoint': {
                'nms_radius': 4,
                'keypoint_threshold': 0.005,
                'max_keypoints': 1024
            },
            'superglue': {
                'weights': 'indoor',
                'sinkhorn_iterations': 20,
                'match_threshold': 0.05,
            }
        }
        self.matching = Matching(config).eval().to(device)
        self.loss = {}

    def forward(self, point_clouds, target_transl, target_rot, transl_err, rot_err, images):
        loss_transl = 0.
        if self.rescale_trans != 0.:
            loss_transl = self.transl_loss(transl_err, target_transl).sum(1).mean()
        loss_rot = 0.
        if self.rescale_rot != 0.:
            loss_rot = quaternion_distance(rot_err, target_rot, rot_err.device).mean()
        pose_loss = self.rescale_trans*loss_transl + self.rescale_rot*loss_rot

        #start = time.time()
        point_clouds_loss = torch.tensor([0.0]).to(transl_err.device)
        for i in range(len(point_clouds)):
            point_cloud_gt = point_clouds[i].to(transl_err.device)
            point_cloud_out = point_clouds[i].clone()

            R_target = quat2mat(target_rot[i])
            T_target = tvector2mat(target_transl[i])
            RT_target = torch.mm(T_target, R_target)

            R_predicted = quat2mat(rot_err[i])
            T_predicted = tvector2mat(transl_err[i])
            RT_predicted = torch.mm(T_predicted, R_predicted)

            RT_total = torch.mm(RT_target.inverse(), RT_predicted)

            point_cloud_out = rotate_forward(point_cloud_out, RT_total)

            error = (point_cloud_out - point_cloud_gt).norm(dim=0)
            error.clamp(100.)
            point_clouds_loss += error.mean()
        point_clouds_loss = point_clouds_loss/target_transl.shape[0]

        superglue_loss = 0.0
        
        
        for i in range(len(point_clouds)):
            pc = point_clouds[i]
            rgb = Grayscale()(images[i])
            rgb = torch.unsqueeze(rgb,0)
            R_predicted = quat2mat(rot_err[i])
            T_predicted = tvector2mat(transl_err[i])
            RT_predicted = torch.mm(T_predicted, R_predicted)
            # pc = rotate_forward(pc,RT_predicted)
            depth,_ = lidar_project_depth(pc,self.K,rgb.shape[2:])
            depth = torch.unsqueeze(depth,0)
            rgb = Resize((480,640))(rgb)
            depth = Resize((480,640))(depth)
            thresh = np.linalg.norm(rgb.shape,2)
            with torch.no_grad():
                pred = self.matching({'image0': rgb, 'image1': depth})
            pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
            kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
            matches, conf = pred['matches0'], pred['matching_scores0']
            valid = matches > -1
            mkpts0 = kpts0[valid]
            mkpts1 = kpts1[matches[valid]]
            mconf = conf[valid]
            dist = np.linalg.norm(mkpts0-mkpts1,ord=2,axis=1)
            dist[dist>thresh] = thresh
            superglue_loss += np.mean(dist)
        superglue_loss = torch.tensor([superglue_loss/len(point_clouds)]).to(device)
            
        #end = time.time()
        #print("3D Distance Time: ", end-start)
        total_loss = 0.2 * pose_loss + 0.2 * (point_clouds_loss/target_transl.shape[0]) + 0.6 * superglue_loss
        self.loss['total_loss'] = total_loss
        self.loss['transl_loss'] = loss_transl
        self.loss['rot_loss'] = loss_rot
        self.loss['point_clouds_loss'] = point_clouds_loss/target_transl.shape[0]
        self.loss['superglue_loss'] = superglue_loss

        return self.loss
