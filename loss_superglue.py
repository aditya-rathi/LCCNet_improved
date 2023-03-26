import torch
from torch import nn as nn
from torchvision.transforms import Grayscale, Resize
from models.matching import Matching
from models.utils_superglue import (plot_image_pair,plot_matches)
from utils import quat2mat, rotate_forward, tvector2mat
from quaternion_distances import quaternion_distance
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib
from scipy.interpolate import griddata

device = 'cuda' if torch.cuda.is_available() else 'cpu'

matplotlib.use('Qt5Agg')

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
    gridx,gridy = np.meshgrid(range(img_shape[1]),range(img_shape[0]))
    grid_z0 = griddata((pcl_uv[:,1],pcl_uv[:,0]),pcl_z,(gridy,gridx),method='linear',fill_value=np.max(pcl_z))
    grid_z0 = ((grid_z0/np.max(grid_z0))*255.0).astype(np.uint8)
    depth_img = torch.from_numpy(grid_z0/255.).float().to(device)
    depth_img = depth_img.cuda()
    depth_img = depth_img.permute(2, 0, 1)

    return depth_img, pcl_uv

class SGLoss(nn.Module):
    def __init__(self,rescale_trans, rescale_rot, do_viz = False):
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
        self.do_viz = do_viz

    def forward(self, point_clouds, target_transl, target_rot, transl_err, rot_err, images, real_shape):
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
        
        try:
            for i in range(len(point_clouds)):
                pc = point_clouds[i]
                rgb = images[i]

                R_target = quat2mat(target_rot[i])
                T_target = tvector2mat(target_transl[i])
                RT_target = torch.mm(T_target, R_target)

                R_predicted = quat2mat(rot_err[i])
                T_predicted = tvector2mat(transl_err[i])
                RT_predicted = torch.mm(T_predicted, R_predicted)

                RT_total = torch.mm(RT_target.inverse(), RT_predicted)
                pc = rotate_forward(pc,RT_total)
                depth,_ = lidar_project_depth(pc,self.K,real_shape)
                depth = torch.unsqueeze(depth,0)
                
                depth = Resize((480,640))(depth)
                thresh = 150
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
                dist = dist[dist<=thresh]
                if len(dist)>0:
                    superglue_loss += np.mean(dist)
                else: superglue_loss+=100

                if self.do_viz:
                # Visualize the matches.
                    plt.close()
                    color = cm.jet(mconf)
                    imgs = [torch.squeeze(rgb).cpu().numpy(), torch.squeeze(depth).cpu().numpy()]
                    n = 2
                    size = 6
                    pad = 0.5
                    dpi = 100
                    assert n == 2, 'number of images must be two'
                    figsize = (size*n, size*3/4) if size is not None else None
                    fig, ax = plt.subplots(1, n, figsize=figsize, dpi=dpi)
                    for i in range(n):
                        ax[i].imshow(imgs[i], cmap=plt.get_cmap('gray'))
                        ax[i].get_yaxis().set_ticks([])
                        ax[i].get_xaxis().set_ticks([])
                        for spine in ax[i].spines.values():  # remove frame
                            spine.set_visible(False)
                    plt.tight_layout(pad=pad)
                    fig.canvas.draw()

                    transFigure = fig.transFigure.inverted()
                    fkpts0 = transFigure.transform(ax[0].transData.transform(mkpts0))
                    fkpts1 = transFigure.transform(ax[1].transData.transform(mkpts1))

                    fig.lines = [matplotlib.lines.Line2D(
                        (fkpts0[i, 0], fkpts1[i, 0]), (fkpts0[i, 1], fkpts1[i, 1]), zorder=1,
                        transform=fig.transFigure, c=color[i], linewidth=1.5)
                                for i in range(len(mkpts0))]
                    ax[0].scatter(mkpts0[:, 0], mkpts0[:, 1], c=color, s=2)
                    ax[1].scatter(mkpts1[:, 0], mkpts1[:, 1], c=color, s=2)

                    plt.show(block=False)
                    
            superglue_loss = torch.tensor([superglue_loss/len(point_clouds)]).to(device)
        except:
            superglue_loss = torch.tensor([100]).to(device)
            
        #end = time.time()
        #print("3D Distance Time: ", end-start)
        if superglue_loss[0] == 100:
            total_loss = 0.8 * pose_loss + 0.1 * (point_clouds_loss/target_transl.shape[0]) + 0.1 * superglue_loss
        else:
            total_loss = 0.2 * pose_loss + 0.2 * (point_clouds_loss/target_transl.shape[0]) + 0.6 * superglue_loss
        self.loss['total_loss'] = total_loss
        self.loss['transl_loss'] = loss_transl
        self.loss['rot_loss'] = loss_rot
        self.loss['point_clouds_loss'] = point_clouds_loss/target_transl.shape[0]
        self.loss['superglue_loss'] = superglue_loss

        return self.loss
