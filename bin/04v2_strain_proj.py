#!/usr/bin/env python
# -*- coding: utf-8 -*-


import nibabel as nib
import numpy as np
import os
from glob import glob
import argparse


## Input
parser = argparse.ArgumentParser()
parser.add_argument('--tflist', nargs="*", type=str, required=True)
parser.add_argument('--niidir', type=str, required=True)
parser.add_argument('--fdir1', type=str, required=True)
parser.add_argument('--fdir2', type=str, required=True)
parser.add_argument('--fmask', type=str, required=True)
parser.add_argument('--foutpre', type=str, required=True)
args = parser.parse_args()


for tfpair in args.tflist:
    tf1 = tfpair[0:2]
    tf2 = tfpair[2:4]

    fl1 = glob(os.path.join(args.niidir, '*_'+tf1+'_'+tf2+'_*l1.nii'))[0]
    fl2 = glob(os.path.join(args.niidir, '*_'+tf1+'_'+tf2+'_*l2.nii'))[0]
    fl3 = glob(os.path.join(args.niidir, '*_'+tf1+'_'+tf2+'_*l3.nii'))[0]
    fVs1 = glob(os.path.join(args.niidir, '*_'+tf1+'_'+tf2+'_*Vs1.nii'))[0]
    fVs2 = glob(os.path.join(args.niidir, '*_'+tf1+'_'+tf2+'_*Vs2.nii'))[0]
    fVs3 = glob(os.path.join(args.niidir, '*_'+tf1+'_'+tf2+'_*Vs3.nii'))[0]

    l1 = nib.load(fl1)
    affine = l1.affine
    header = l1.header
    l1 = l1.get_fdata()
    l2 = nib.load(fl2).get_fdata()
    l3 = nib.load(fl3).get_fdata()
    mask = nib.load(args.fmask).get_fdata()

    Vs1 = nib.load(fVs1).get_fdata()
    Vs2 = nib.load(fVs2).get_fdata()
    Vs3 = nib.load(fVs3).get_fdata()
    Vs1 = Vs1[..., 0, :]
    Vs2 = Vs2[..., 0, :]
    Vs3 = Vs3[..., 0, :]
    fiberDir1 = nib.load(args.fdir1).get_fdata()
    fiberDir2 = nib.load(args.fdir2).get_fdata()

    x, y, z = (mask > 0).nonzero()
    ratio1 = np.zeros(mask.shape)
    ratio2 = np.zeros(mask.shape)
    for pos in range(len(x)):
        i, j, k = x[pos], y[pos], z[pos]
        Sigma = np.diag([l1[i,j,k], l2[i,j,k], l3[i,j,k]])
        VsT = np.array([[Vs1[i,j,k,0], Vs1[i,j,k,1], Vs1[i,j,k,2]],
                        [Vs2[i,j,k,0], Vs2[i,j,k,1], Vs2[i,j,k,2]],
                        [Vs3[i,j,k,0], Vs3[i,j,k,1], Vs3[i,j,k,2]]], dtype=np.float32)
        Vs = np.transpose(VsT)
        C = np.dot(np.dot(Vs, Sigma*Sigma), VsT)
        dirtemp = np.array([fiberDir1[i, j, k]], dtype=np.float32)
        dirtemp_v = np.transpose(dirtemp)
        ratio1[i, j, k] = np.sqrt(np.dot(np.dot(dirtemp, C), dirtemp_v))
    
        dirtemp = np.array([fiberDir2[i, j, k]], dtype=np.float32)
        dirtemp_v = np.transpose(dirtemp)
        ratio2[i, j, k] = np.sqrt(np.dot(np.dot(dirtemp, C), dirtemp_v))

    out = nib.Nifti1Image(ratio1, affine, header)
    out.to_filename(args.foutpre+tf1+'_'+tf2+'_proj_on_cnn_fiberDir1.nii')
    out = nib.Nifti1Image(ratio2, affine, header)
    out.to_filename(args.foutpre+tf1+'_'+tf2+'_proj_on_cnn_fiberDir2.nii')
