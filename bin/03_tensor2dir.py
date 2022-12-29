#!/usr/bin/env python
# -*- coding: utf-8 -*-


import nibabel as nib
import numpy as np
import argparse


### Tensor to fiber
## Input
parser = argparse.ArgumentParser()
parser.add_argument('--ftensor1', type=str, required=True)
parser.add_argument('--ftensor2', type=str, required=True)
parser.add_argument('--frefimg', type=str, required=True)
parser.add_argument('--foutpre', type=str, required=True)
parser.add_argument('--fmask1', type=str, required=False)
parser.add_argument('--fmask2', type=str, required=False)
args = parser.parse_args()


tensor1 = nib.load(args.ftensor1)
tensor2 = nib.load(args.ftensor2)

tensor1img = tensor1.get_fdata()
tensor1img = tensor1img[..., 0, :]
tensor2img = tensor2.get_fdata()
tensor2img = tensor2img[..., 0, :]
fixedImg = nib.load(args.frefimg)

dir1img = np.zeros(list(fixedImg.get_fdata().shape)+[3], dtype=np.float32)
mask1 = (((np.abs(tensor1img)).sum(axis=-1)) > 0) * 1
x, y, z = (mask1 > 0).nonzero()
for pos in range(len(x)):
    i, j, k = x[pos], y[pos], z[pos]
    #### Dir1
    D = np.zeros([3, 3], dtype=np.float32)
    D[0, 0] = tensor1img[i, j, k, 0]
    D[1, 0] = tensor1img[i, j, k, 1]
    D[1, 1] = tensor1img[i, j, k, 2]
    D[2, 0] = tensor1img[i, j, k, 3]
    D[2, 1] = tensor1img[i, j, k, 4]
    D[2, 2] = tensor1img[i, j, k, 5]

    D[0, 1] = D[1, 0]
    D[0, 2] = D[2, 0]
    D[1, 2] = D[2, 1]

    w, v = np.linalg.eig(D)
    indx = np.argmax(w)
    dir1img[i, j, k] = v[:, indx]


dir2img = np.zeros(list(fixedImg.get_fdata().shape)+[3], dtype=np.float32)
mask2 = (((np.abs(tensor2img)).sum(axis=-1)) > 0) * 1
x, y, z = (mask2 > 0).nonzero()
for pos in range(len(x)):
    i, j, k = x[pos], y[pos], z[pos]
    #### Dir2
    D = np.zeros([3, 3], dtype=np.float32)
    D[0, 0] = tensor2img[i, j, k, 0]
    D[1, 0] = tensor2img[i, j, k, 1]
    D[1, 1] = tensor2img[i, j, k, 2]
    D[2, 0] = tensor2img[i, j, k, 3]
    D[2, 1] = tensor2img[i, j, k, 4]
    D[2, 2] = tensor2img[i, j, k, 5]

    D[0, 1] = D[1, 0]
    D[0, 2] = D[2, 0]
    D[1, 2] = D[2, 1]

    w, v = np.linalg.eig(D)
    indx = np.argmax(w)
    dir2img[i, j, k] = v[:, indx]

out = nib.Nifti1Image(dir1img, fixedImg.affine, fixedImg.header)
out.to_filename(args.foutpre + '_dir1_tensor_trans_reorient_todir.nii.gz')
out = nib.Nifti1Image(dir2img, fixedImg.affine, fixedImg.header)
out.to_filename(args.foutpre + '_dir2_tensor_trans_reorient_todir.nii.gz')
