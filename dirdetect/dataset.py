#!/usr/bin/env python
# -*- coding: utf-8 -*-


import torch
import os
from glob import glob
import nibabel as nib
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, ToPILImage, ToTensor
import numpy as np
import pdb


class SHDataset(Dataset):
    def __init__(self, dirname, ext='.nii*', patch_size=7):
        self.dirname = dirname
        self.ext = ext
        self.patch_size = patch_size
        self.fmask = glob(os.path.join(self.dirname, '*mask*.nii*'))[0]
        self.files = sorted(glob(os.path.join(self.dirname, '*SH*'+self.ext)))
        self.n_img = len(self.files)
        self.mask = nib.load(self.fmask).get_data()
        self.n_patch = np.sum(self.mask > 0)

        # Fiber ODF in spherical harmonics
        fodf_sh = glob(os.path.join(self.dirname, '*fodf_sh.nii*'))[0]
        self.odf_sh = nib.load(fodf_sh).get_data()

        # Load dwi_sh data
        temp = nib.load(self.files[0]).get_data()
        self.dwish = np.zeros(list(temp.shape) + [self.n_img])
        for i in range(self.n_img):
            self.dwish[..., i] = nib.load(self.files[i]).get_data()

        # Load direction data
        fdir1 = glob(os.path.join(self.dirname, '*dir1*.nii*'))[0]
        fdir2 = glob(os.path.join(self.dirname, '*dir2*.nii*'))[0]
        dir1 = nib.load(fdir1).get_data()
        dir2 = nib.load(fdir2).get_data()
        self.totaldir = np.concatenate((dir1, dir2), axis=-1)

        # Load fraction data
        fnum = glob(os.path.join(self.dirname, '*num*.nii*'))[0]
        self.num = nib.load(fnum).get_data()[..., np.newaxis]
        self.num = self.num - 1
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
        num_patch = torch.from_numpy(self.num[i, j, k, :]).float()
        return data, odf_sh, dir_patch, num_patch


class SHNeighborDataset(SHDataset):
    def __init__(self, dirname, ext='.nii*', patch_size=7):
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

        dir_patch = self.totaldir[i:i+1, j:j+1, k:k+1, :]
        dir_patch = torch.from_numpy(dir_patch).float()
        dir_patch = dir_patch.permute(3, 0, 1, 2)

        num_patch = self.num[i:i+1, j:j+1, k:k+1, :]
        num_patch = torch.from_numpy(num_patch).float()
        num_patch = num_patch.permute(3, 0, 1, 2)
        return data, odf_sh, dir_patch, num_patch


class MultiSetDataset(Dataset):
    # Definition of 1 set: same mask, same fODF and same ground truth directions.
    def __init__(self, dirname, ext='.nii*', patch_size=5):
        self.dirname = dirname
        self.ext = ext
        self.patch_size = patch_size

        self.fsets = sorted(glob(os.path.join(self.dirname, 'data*')))
        self.n_sets = len(self.fsets)
        self.dwish = []
        self.odfsh = []
        self.totaldir = []
        self.totalnum = []
        self.n_patch = np.zeros(self.n_sets, dtype=np.int32)
        self.n_img = np.zeros(self.n_sets, dtype=np.int32)
        self.x = []
        self.y = []
        self.z = []

        for i in range(self.n_sets):
            fmask = glob(os.path.join(self.fsets[i], '*mask*'+self.ext))[0]
            fshlist = sorted(glob(os.path.join(self.fsets[i], '*SH*'+self.ext)))
            mask = nib.load(fmask).get_fdata()

            # Total number of images in the current set and number of patches
            # in each image.
            self.n_img[i] = len(fshlist)
            self.n_patch[i] = np.sum(mask > 0)

            # Coordinate in the current mask
            tempx, tempy, tempz = (mask > 0).nonzero()
            self.x.append(tempx)
            self.y.append(tempy)
            self.z.append(tempz)

            # Fiber ODF in current set
            temp_odfsh = glob(os.path.join(self.fsets[i], '*fodf_sh.nii*'))[0]
            self.odfsh.append(nib.load(temp_odfsh).get_fdata())
            # Fiber directions in current set
            fdir1 = glob(os.path.join(self.fsets[i], '*dir1*'+self.ext))[0]
            fdir2 = glob(os.path.join(self.fsets[i], '*dir2*'+self.ext))[0]
            dir1 = nib.load(fdir1).get_fdata()
            dir2 = nib.load(fdir2).get_fdata()
            self.totaldir.append(np.concatenate((dir1, dir2), axis=-1))
            # Fiber number in current set
            fnum = glob(os.path.join(self.fsets[i], '*num*'+self.ext))[0]
            num = nib.load(fnum).get_fdata()[..., np.newaxis]
            num = num - 1
            self.totalnum.append(num)

            # DWI SH
            temp = nib.load(fshlist[0]).get_fdata()
            tempsh = np.zeros(list(temp.shape) + [self.n_img[i]])
            for j in range(self.n_img[i]):
                tempsh[..., j] = nib.load(fshlist[j]).get_fdata()
            self.dwish.append(tempsh)

        self.n_patch_per_set = self.n_img * self.n_patch
        self.n_patch_upperbound = np.cumsum(self.n_patch_per_set)
        self.n_patch_lowerbound = self.n_patch_upperbound - self.n_patch_per_set

        # ################ Test
        # self.n_sets = 3
        # self.n_patch = np.array([25, 40, 50], dtype=np.int32)
        # self.n_img = np.array([4, 5, 6], dtype=np.int32)
        # self.n_patch_per_set = self.n_img * self.n_patch
        # self.n_patch_upperbound = np.cumsum(self.n_patch_per_set)
        # self.n_patch_lowerbound = self.n_patch_upperbound - self.n_patch_per_set

    def __len__(self):
        return self.n_patch_per_set.sum()

    def __getitem__(self, indx: int):
        indx_set = np.argwhere(indx < self.n_patch_upperbound)[0, 0]
        temp_dwish = self.dwish[indx_set]
        temp_odfsh = self.odfsh[indx_set]
        temp_num = self.totalnum[indx_set]
        temp_dir = self.totaldir[indx_set]
        indx_in_set = indx - self.n_patch_lowerbound[indx_set]

        indx_img = indx_in_set // self.n_patch[indx_set]
        indx_patch = indx_in_set % self.n_patch[indx_set]

        i = self.x[indx_set][indx_patch]
        j = self.y[indx_set][indx_patch]
        k = self.z[indx_set][indx_patch]

        m = self.patch_size // 2
        data = temp_dwish[i-m:i+m+1, j-m:j+m+1, k-m:k+m+1, :, indx_img]
        data = torch.from_numpy(data).float()
        data = data.permute(3, 0, 1, 2)

        mm = (self.patch_size - 4) // 2
        odf_sh = temp_odfsh[i-mm:i+mm+1, j-mm:j+mm+1, k-mm:k+mm+1, :]
        odf_sh = torch.from_numpy(odf_sh).float()
        odf_sh = odf_sh.permute(3, 0, 1, 2)

        num_patch = temp_num[i:i+1, j:j+1, k:k+1, :]
        num_patch = torch.from_numpy(num_patch).float()
        num_patch = num_patch.permute(3, 0, 1, 2)

        dir_patch = temp_dir[i:i+1, j:j+1, k:k+1, :]
        dir_patch = torch.from_numpy(dir_patch).float()
        dir_patch = dir_patch.permute(3, 0, 1, 2)
        return data, odf_sh, dir_patch, num_patch
