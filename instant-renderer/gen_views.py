import os
import json
import math
import numpy as np
from PIL import Image
import cv2

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torchvision.transforms.functional as TF

import pytorch_lightning as pl

import datasets
from models.ray_utils import get_ortho_ray_directions_origins, get_ortho_rays, get_ray_directions
from utils.misc import get_rank

from glob import glob
import PIL.Image
import tqdm

import pdb

def RT_opengl2opencv(RT): # RT is w2c, RT.shape=[3,4]
     # Build the coordinate transform matrix from world to computer vision camera
    # R_world2cv = R_bcam2cv@R_world2bcam
    # T_world2cv = R_bcam2cv@T_world2bcam

    R = RT[:3, :3]
    t = RT[:3, 3]

    R_bcam2cv = np.asarray([[1, 0, 0], [0, -1, 0], [0, 0, -1]], np.float32)

    R_world2cv = R_bcam2cv @ R
    t_world2cv = R_bcam2cv @ t

    RT = np.concatenate([R_world2cv,t_world2cv[:,None]],1)

    return RT

def RT_blender2opencv(RT): # RT is c2w, RT.shape=[4,4]

    RT = RT @ np.array(
                [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    return RT

def inv_RT(RT):
    RT_h = np.concatenate([RT, np.array([[0,0,0,1]])], axis=0)
    RT_inv = np.linalg.inv(RT_h)

    return RT_inv[:3, :]

def pose_spherical(theta, phi, radius):
    trans_t = lambda t : torch.Tensor([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,t],
        [0,0,0,1]]).float()

    rot_phi = lambda phi : torch.Tensor([
        [1,0,0,0],
        [0,np.cos(phi),-np.sin(phi),0],
        [0,np.sin(phi), np.cos(phi),0],
        [0,0,0,1]]).float()

    rot_theta = lambda th : torch.Tensor([
        [np.cos(th),0,-np.sin(th),0],
        [0,1,0,0],
        [np.sin(th),0, np.cos(th),0],
        [0,0,0,1]]).float()

    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w

def generate_views(azimuths, elevations, radius, save_dir='datasets/new_poses_cv'):
    os.makedirs(save_dir, exist_ok=True)
    w2cs = []

    for i in tqdm.tqdm(range(len(azimuths))):
        for j in range(len(elevations)):
            for k in range(len(radius)):
                c2w = pose_spherical(theta=azimuths[i], phi=elevations[j], radius=radius[k]) # theta: azimuth, phi: elevation
                c2w = RT_blender2opencv(c2w)
                w2c = inv_RT(c2w[:3])
        
                np.savetxt(os.path.join(save_dir, '000_%d_%d_%d_RT_cv.txt'%(i, j, k)), w2c)
        # break

def views_txt2npz(txt_dir, npz_dir):
    pose_dirs = sorted(os.listdir(txt_dir))
    pose_dirs_tmp = []
    for pose_dir in pose_dirs:
        if pose_dir.split('.')[-1] == 'txt':
            pose_dirs_tmp.append(pose_dir)
    pose_dirs = pose_dirs_tmp
    del pose_dirs_tmp

    w2cs = []
    for pose_dir in tqdm.tqdm(pose_dirs):
        RT_cv = np.loadtxt('%s/%s' % (txt_dir, pose_dir))  # world2cam matrix
        w2cs.append(RT_cv)

    w2cs_dic = {}
    for i in range(len(w2cs)):
        w2cs_dic['world_mat_%d' % i] = np.concatenate([w2cs[i], np.array([[0,0,0,1]])], axis=0)
        w2cs_dic['scale_mat_%d' % i] = np.eye(4, dtype=np.float32)
    
    np.savez('%s/cameras_sphere.npz'%(npz_dir), **w2cs_dic)

if __name__ == '__main__':
    '''
    azimuths = []
    num_azimuths = 30
    for i in range(1, num_azimuths + 1):
        azimuths.append(360 // num_azimuths * i - 180)
    print(azimuths)
    
    elevations = [0, 30, 60, -30, -60]

    radius = [1.5]
    generate_views(azimuths, elevations, radius)
    '''
    views_txt2npz('datasets/new_poses_cv', 'datasets/new_poses_cv')
    a = np.load('datasets/new_poses_cv/cameras_sphere.npz')
    pdb.set_trace()