import torch
from torch import nn as nn
from models.matching import Matching
from models.utils_superglue import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class SGLoss(nn.Module):
    def __init__(self,rescale_trans, rescale_rot, weight_point_cloud):
        super(SGLoss,self).__init__()
        self.rescale_trans = rescale_trans
        self.rescale_rot = rescale_rot
        self.transl_loss = nn.SmoothL1Loss(reduction='none')
        self.weight_point_cloud = weight_point_cloud
        self.loss = {}

    def forward(self, point_clouds, target_transl, target_rot, transl_err, rot_err, images):
