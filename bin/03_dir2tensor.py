#!/usr/bin/env python
# -*- coding: utf-8 -*-


import nibabel as nib
import numpy as np
from dipy.sims.voxel import all_tensor_evecs
import argparse


## Input
parser = argparse.ArgumentParser()
parser.add_argument('--fdir1', type=str, required=True)
parser.add_argument('--fdir2', type=str, required=True)
parser.add_argument('--fmask', type=str, required=True)
parser.add_argument('--foutpre', type=str, required=True)
args = parser.parse_args()

#### Fiber orientations to tensor
mask = nib.load(args.fmask)
dir1 = nib.load(args.fdir1)
dir1img = dir1.get_fdata()
mask1img = (((np.abs(dir1img)).sum(axis=-1)) > 0) * 1
dir2 = nib.load(args.fdir2)
dir2img = dir2.get_fdata()
mask2img = (((np.abs(dir2img)).sum(axis=-1)) > 0) * 1
nib.Nifti1Image(mask2img, mask.affine, mask.header).to_filename(args.foutpre + '_dir2_mask.nii.gz')

tensor1img = np.zeros(list(mask1img.shape)+[1, 6], dtype=np.float32)
x, y, z = (mask1img > 0).nonzero()
for pos in range(len(x)):
    i, j, k = x[pos], y[pos], z[pos]
    mevals = np.diag([0.005, 0.002, 0.002])
    # dir1
    R = all_tensor_evecs(dir1img[i, j, k])
    D = np.dot(np.dot(R, mevals), R.T)
    tensor1img[i, j, k, 0, 0] = D[0, 0]
    tensor1img[i, j, k, 0, 1] = D[1, 0]
    tensor1img[i, j, k, 0, 2] = D[1, 1]
    tensor1img[i, j, k, 0, 3] = D[0, 2]
    tensor1img[i, j, k, 0, 4] = D[1, 2]
    tensor1img[i, j, k, 0, 5] = D[2, 2]

tensor2img = np.zeros(list(mask2img.shape)+[1, 6], dtype=np.float32)
x, y, z = (mask2img > 0).nonzero()
for pos in range(len(x)):
    i, j, k = x[pos], y[pos], z[pos]
    mevals = np.diag([0.005, 0.002, 0.002])
    # dir2
    R = all_tensor_evecs(dir2img[i, j, k])
    D = np.dot(np.dot(R, mevals), R.T)
    tensor2img[i, j, k, 0, 0] = D[0, 0]
    tensor2img[i, j, k, 0, 1] = D[1, 0]
    tensor2img[i, j, k, 0, 2] = D[1, 1]
    tensor2img[i, j, k, 0, 3] = D[0, 2]
    tensor2img[i, j, k, 0, 4] = D[1, 2]
    tensor2img[i, j, k, 0, 5] = D[2, 2]

affine = dir1.affine
header = dir1.header
header['intent_code'] = 1005 # Change the 'intent_code' to 1005 (symmetric matrix)
header['intent_name'] = 'DTI'
nib.Nifti1Image(tensor1img, affine, header).to_filename(args.foutpre + '_dir1_tensor.nii.gz')
nib.Nifti1Image(tensor2img, affine, header).to_filename(args.foutpre + '_dir2_tensor.nii.gz')
