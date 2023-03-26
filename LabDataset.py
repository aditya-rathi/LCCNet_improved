import numpy as np
import math
import cv2
import os
import open3d as o3d

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils import (mat2xyzrpy, merge_inputs, overlay_imgs, quat2mat,
                   quaternion_from_matrix, rotate_back, rotate_forward,
                   tvector2mat)

from models.utils_superglue import read_image

from new_utils import (get_random_pointcloud, get_multiple_random_pointclouds, save_pointcloud, get_random_translation, get_random_orientation)


class LabDataset(Dataset):
	def __init__(self, root = "./data/", max_r=20.0, max_t=1.5):
		super(LabDataset, self).__init__()
	
		self.max_r = max_r
		self.max_t = max_t
		
		self.rgb_path = os.path.join(root, "camera.png")
		self.lidar_path = os.path.join(root, "lidar.ply")

		# self.rgb_list = sorted(os.listdir(self.rgb_path))
		# self.lidar_list = sorted(os.listdir(self.lidar_path))

		self.rand_gen = np.random.default_rng()
		
		
		# self.lidar_pcd = o3d.io.read_point_cloud(self.lidar_path)
		# self.lidar_pcd = self.lidar_pcd.voxel_down_sample(voxel_size=7) # 20 previously
		# self.lidar = np.asarray(self.lidar_pcd.points)
		
		# inds = 2000 > self.lidar[:,2] # thresholding
		# self.lidar = self.lidar[inds]
		
		# self.lidar = self.lidar * 1e-3
		# self.lidar = np.concatenate((self.lidar, np.ones((self.lidar.shape[0], 1))), axis=1) # N,4
		
		self.K = np.array([[1.82146e3, 0, 9.44721e2],
							[0, 1.817312e3, 5.97134e2],
							[0,0,1]])
		
		self.transform = transforms.Compose([
			transforms.ToTensor(),	
			transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])	
			# transforms.Grayscale()
		])
	
	# def custom_transform(self, rgb, img_rotation=0., flip=False):
	# 	to_tensor = transforms.ToTensor()
	# 	normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                          std=[0.229, 0.224, 0.225])
	# 	rgb = to_tensor(rgb)
	# 	rgb = normalization(rgb)
	# 	return rgb
	
	def __len__(self):
		# return len(self.rgb_list)
		return 500 
	
	def __getitem__(self, idx):
		# idx = self.rand_gen.integers(0,len(self.rgb_list))
		img = cv2.imread(self.rgb_path).astype(np.uint8)
		img = self.transform(img)
		img_shape = img.shape[1:]

		_,gray,_ = read_image(self.rgb_path,'cpu',[640,480],0,True)

		pc = o3d.io.read_point_cloud(self.lidar_path)
		pc = pc.voxel_down_sample(voxel_size=7)
		pc = np.asarray(pc.points)
		pc = np.concatenate((pc, np.ones((pc.shape[0], 1))), axis=1)
		pc = pc.astype(np.float32)
		
		r = get_random_orientation(self.max_r)
		t = get_random_translation(self.max_t)
		
		RT = np.concatenate((r, t), axis=1)
		RT = np.concatenate((RT, np.array([0,0,0,1]).reshape(1, 4)), axis=0)
		
#		print(RT)
		RT = np.linalg.inv(RT)
		
		init_pc = pc @ RT # N,4 
		
		init_pc = torch.from_numpy(init_pc.astype(np.float32))
		RT = torch.tensor(RT.astype(np.float32))

		R, T = torch.tensor(r), torch.tensor(t) 
		tr_error = T
		rot_error = quaternion_from_matrix(R)	
		
		sample = {'rgb': img, 'gray': gray, 'K':self.K, 'tr_error': tr_error, 'rot_error': rot_error, 'point_cloud': pc,	'init_pc':init_pc, 'RT':RT}

		return sample	
	
