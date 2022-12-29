#!/usr/bin/env python
# -*- coding: utf-8 -*-


import torch
import os
from glob import glob
import nibabel as nib
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, ToPILImage, ToTensor
import numpy as np


class SHDataset(Dataset):
    def __init__(self, dirname, ext='.nii', patch_size=7):
        self.dirname = dirname
        self.ext = ext
        # self.num_class = 3
        self.patch_size = patch_size
        self.fmask = glob(os.path.join(self.dirname, '*mask*.nii'))[0]
        self.files = sorted(glob(os.path.join(self.dirname, '*SH*'+self.ext)))
        self.n_img = len(self.files)
        self.mask = nib.load(self.fmask).get_data()
        self.n_patch = np.sum(self.mask > 0)
        # self.n_coeff = 66

        # Fiber ODF in spherical harmonics
        fodf_sh = glob(os.path.join(self.dirname, '*fodf_sh.nii'))[0]
        self.odf_sh = nib.load(fodf_sh).get_data()

        # Load dwi_sh data
        temp = nib.load(self.files[0]).get_data()
        self.dwish = np.zeros(list(temp.shape) + [self.n_img])
        for i in range(self.n_img):
            self.dwish[..., i] = nib.load(self.files[i]).get_data()

        # Load direction data
        fdir1 = glob(os.path.join(self.dirname, '*dir1*.nii'))[0]
        fdir2 = glob(os.path.join(self.dirname, '*dir2*.nii'))[0]
        dir1 = nib.load(fdir1).get_data()
        dir2 = nib.load(fdir2).get_data()
        self.totaldir = np.concatenate((dir1, dir2), axis=-1)

        # Load fraction data
        find1 = glob(os.path.join(self.dirname, '*ind1*.nii'))[0]
        find2 = glob(os.path.join(self.dirname, '*ind2*.nii'))[0]
        ind1 = nib.load(find1).get_data()
        ind2 = nib.load(find2).get_data()
        self.totalind = np.concatenate((ind1[..., np.newaxis],
                                        ind2[..., np.newaxis]),
                                       axis=-1)
        self.x, self.y, self.z = (self.mask > 0).nonzero()

    def __len__(self):
        return self.n_img * self.n_patch

    def __getitem__(self, indx: int):
        indx_img = indx // self.n_patch
        indx_patch = indx % self.n_patch

        i, j, k = self.x[indx_patch], self.y[indx_patch], self.z[indx_patch]

        data = self.dwish[i-1:i+2, j-1:j+2, k-1:k+2, :, indx_img]
        data = torch.from_numpy(data).float()
        data = data.permute(3, 0, 1, 2)
        odf_sh = torch.from_numpy(self.odf_sh[i, j, k, :]).float()
        dir_patch = torch.from_numpy(self.totaldir[i, j, k, :]).float()
        ind_patch = torch.from_numpy(self.totalind[i, j, k, :]).float()
        return data, odf_sh, dir_patch, ind_patch


class SHNeighborDataset(SHDataset):
    def __init__(self, dirname, ext='.nii', patch_size=7):
        super().__init__(dirname=dirname, ext=ext, patch_size=patch_size)

    def __getitem__(self, indx: int):
        indx_img = indx // self.n_patch
        indx_patch = indx % self.n_patch

        i, j, k = self.x[indx_patch], self.y[indx_patch], self.z[indx_patch]

        m = self.patch_size // 2

        data = self.dwish[i-m:i+m+1, j-m:j+m+1, k-m:k+m+1, :, indx_img]
        data = torch.from_numpy(data).float()
        data = data.permute(3, 0, 1, 2)

        odf_sh = self.odf_sh[i-1:i+2, j-1:j+2, k-1:k+2, :]
        odf_sh = torch.from_numpy(odf_sh).float()
        odf_sh = odf_sh.permute(3, 0, 1, 2)

        dir_patch = self.totaldir[i-1:i+2, j-1:j+2, k-1:k+2, :]
        dir_patch = torch.from_numpy(dir_patch).float()
        dir_patch = dir_patch.permute(3, 0, 1, 2)

        ind_patch = self.totalind[i-1:i+2, j-1:j+2, k-1:k+2, :]
        ind_patch = torch.from_numpy(ind_patch).float()
        ind_patch = ind_patch.permute(3, 0, 1, 2)
        return data, odf_sh, dir_patch, ind_patch








