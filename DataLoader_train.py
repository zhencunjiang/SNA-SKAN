# -*- coding: utf-8 -*-
"""
Created on 2022-06-25

@author: Mufeng Geng
"""
import os
import torch.utils.data as data
from PIL import Image
import numpy as np
from skimage.restoration import (denoise_wavelet, estimate_sigma)
import torch
import torch.nn as  nn

from numpy import linalg as la
def svd_denoise(img):
    u, sigma, vt = la.svd(img)
    h, w = img.shape[:2]
    h1 = int(h * 0.1) #取前10%的奇异值重构图像
    sigma1 = np.diag(sigma[:h1],0) #用奇异值生成对角矩阵
    u1 = np.zeros((h,h1), float)
    # print(u1.shape)
    u1[:,:] = u[:,:h1]
    vt1 = np.zeros((h1,w), float)
    vt1[:,:] = vt[:h1,:]
    return u1.dot(sigma1).dot(vt1)

# Divide the data set according to the txt files
noisy_1_txt = r"/home/ps/zhencunjiang/sna-skan/code/all_train_noisy.txt"

clean_txt = r"/home/ps/zhencunjiang/sna-skan/code/all_train_clean.txt"

noisy_1_list = list()

clean_list = list()
for line_noisy_1 in open(noisy_1_txt, "r"):
    noisy_1_list.append(line_noisy_1.strip())



for line_clean in open(clean_txt, "r"):
    clean_list.append(line_clean.strip())

def get_Training_Set():
    return DatasetFromFolder(noisy_1_list, clean_list)


def load_image(filepath):
    image =Image.open(filepath)
    image = np.array(image).astype('float32')/255.0
    return image

class DatasetFromFolder(data.Dataset):
    def __init__(self, noisy_1_list, noisy_2_list,clean_list):
        super(DatasetFromFolder, self).__init__()
        self.noisy_1_list = noisy_1_list
        self.noisy_2_list = noisy_2_list
        self.clean_list = clean_list

    def __getitem__(self, index):
        # noisy_1 = read_h5(self.noisy_1_list[index])
        # noisy_2 = read_h5(self.noisy_2_list[index])
        # clean = read_h5(self.clean_list[index])
        noisy_1 = load_image(self.noisy_1_list[index])
        noisy_2 = load_image(self.noisy_2_list[index])
        clean = load_image(self.clean_list[index])

        # print(noisy_1.shape)
        # print(clean.shape)
        return {"A": noisy_1, "B": noisy_2,  "C": clean}

    def __len__(self):
        return len(self.noisy_1_list)

def patch_shuffle(x, patch_size):
    # Reshape input tensor to patches
    batch_size, channels, height, width = x.size()
    unfolded = torch.nn.functional.unfold(x, kernel_size=patch_size, stride=patch_size)

    # Reshape patches to original shape
    unfolded_permuted = unfolded.permute(0, 2, 1).contiguous()
    folded_permuted = unfolded_permuted.view(batch_size, channels, -1, patch_size, patch_size)
    output = folded_permuted.permute(0, 1, 3, 2, 4).contiguous().view(batch_size, channels, height, width)
    return output

from skimage.util import random_noise
class svdDatasetFromFolder(data.Dataset):
    def __init__(self, noisy_1_list, clean_list):
        super(svdDatasetFromFolder, self).__init__()
        self.noisy_1_list = noisy_1_list
        self.clean_list = clean_list

    def __getitem__(self, index):
        # noisy_1 = read_h5(self.noisy_1_list[index])
        # noisy_2 = read_h5(self.noisy_2_list[index])
        # clean = read_h5(self.clean_list[index])
        noisy_1 = load_image(self.noisy_1_list[index])
        clean = load_image(self.clean_list[index])
        im_haar = denoise_wavelet(noisy_1, wavelet='db3', channel_axis=0)
        noisy_harr = noisy_1 - im_haar
        # noisy_harr = random_noise(clean, mode='speckle', mean=0, var=0.02) - clean
        # print(noisy_1.shape)
        im_svd = svd_denoise(noisy_1)
        noisy_svd = noisy_1 - im_svd
        # noisy_svd = random_noise(clean, mode='speckle', mean=0, var=0.02) - clean
        noisy_1 = np.expand_dims(noisy_1, axis=0)
        clean = np.expand_dims(clean, axis=0)
        noisy_svd = np.expand_dims(np.float32(noisy_svd), axis=0)
        noisy_harr = np.expand_dims(np.float32(noisy_harr), axis=0)


        # noisy_svd = noisy_svd.reshape([4, 320, 320])
        # noisy_svd = nn.PixelShuffle(2)(torch.from_numpy(noisy_svd)).numpy()
        # noisy_harr = noisy_harr.reshape([4, 320, 320])
        # noisy_harr = nn.PixelShuffle(2)(torch.from_numpy(noisy_harr)).numpy()
        # noisy_svd = patch_shuffle(torch.from_numpy(noisy_svd).unsqueeze(0), patch_size=16).squeeze(0)
        # noisy_harr = patch_shuffle(torch.from_numpy(noisy_harr).unsqueeze(0), patch_size=16).squeeze(0)
        # print(noisy_svd.shape)
        # print(noisy_harr.shape)
        return {"A": noisy_1, "B": noisy_harr, "C": noisy_svd, "D": clean}

    def __len__(self):
        return len(self.noisy_1_list)

def svd_get_Training_Set():
    return svdDatasetFromFolder(noisy_1_list, clean_list)