#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: data.py
@Time: 2018/10/13 6:21 PM
"""


import os
import sys
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset

def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget --no-check-certificate %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        #os.system('rm %s' % (zipfile))


def load_data(partition):
    download()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_normal = []
    all_label = []
    if partition == 'val':
      _partition = 'train'
    elif partition == 'train':
      _partition = 'train'
    else:
      _partition = partition
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%_partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        normal = f['normal'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_normal.append(normal)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_normal = np.concatenate(all_normal, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    if partition == 'train':
      all_data = all_data[:-984]
      all_normal = all_normal[:-984]
      all_label = all_label[:-984]
    elif partition == 'val':
      all_data = all_data[-984:]
      all_normal = all_normal[-984:]
      all_label = all_label[-984:]
      
    return all_data, all_normal, all_label


def translate_pointcloud(pointcloud):
    normals = pointcloud[:,3:]
    pointcloud = pointcloud[:,0:3]
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    #translated_pointcloud = np.add(pointcloud, xyz2).astype('float32')
    scaled_normals = np.multiply(normals, 1./xyz1).astype('float32')
    pc = np.concatenate((translated_pointcloud,scaled_normals),axis=1)
    return pc 


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.normal, self.label = load_data(partition)
        self.num_points = num_points
        self.partition = partition        

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        normals = self.normal[item][:self.num_points]
        pc = np.concatenate((pointcloud,normals),axis=1)
        label = self.label[item]
        if self.partition != 'test':
            pc = translate_pointcloud(pc)
            np.random.shuffle(pc)
        return pc, label

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    train = ModelNet40(1024)
    test = ModelNet40(1024, 'test')
    for data, label in train:
        print(data.shape)
        print(label.shape)
