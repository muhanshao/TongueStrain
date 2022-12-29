#!/usr/bin/env python
# -*- coding: utf-8 -*-


import nibabel as nib
import numpy as np
from dwave_qbsolv import QBSolv
import copy
import argparse


## Input
parser = argparse.ArgumentParser()
parser.add_argument('--fdir1', type=str, required=True)
parser.add_argument('--fdir2', type=str, required=True)
parser.add_argument('--fmask', type=str, required=True)
parser.add_argument('--foutpre', type=str, required=True)
args = parser.parse_args()

dir1 = nib.load(args.fdir1)
affine = dir1.affine
header = dir1.header
dir1 = dir1.get_fdata().astype(np.float32)
dir2 = nib.load(args.fdir2).get_fdata().astype(np.float32)
mask = nib.load(args.fmask).get_fdata().astype(np.float32)

x, y, z = (mask > 0).nonzero()
Q = dict()

n_node = len(x)
patch_size = 3
n_neighbor = patch_size**3-1
indx = list(range(n_neighbor//2)) + list(range(n_neighbor//2+1, n_neighbor+1))
m = patch_size // 2
sub_indx = [(x, y, z) for x in np.arange(patch_size)-m
            for y in np.arange(patch_size)-m
            for z in np.arange(patch_size)-m]
sub_indx = np.array(sub_indx, dtype=np.int32)

weight = np.sqrt((sub_indx**2).sum(axis=-1))
weight = np.exp(-weight)
weight = weight[indx]
weight = 1 / weight

node_coord = np.stack((x, y, z), axis=1)
sub_indx_neighbor = sub_indx[indx]
for pos in range(n_node):
    i, j, k = x[pos], y[pos], z[pos]
    neighbor_coord = np.array([i, j, k], dtype=np.int32)[None, :] + sub_indx_neighbor
    dir_i1 = dir1[i, j, k, :]
    dir_i2 = dir2[i, j, k, :]
    for n in range(neighbor_coord.shape[0]):
        n_coord = neighbor_coord[n, :]
        n_indx = np.where(np.all(n_coord == node_coord, axis=-1))[0]
        if n_indx.size > 0:
            if n_indx[0] > pos:
                dir_j1 = dir1[n_coord[0], n_coord[1], n_coord[2], :]
                dir_j2 = dir2[n_coord[0], n_coord[1], n_coord[2], :]
                sim11 = np.abs((dir_i1 * dir_j1).sum()) * weight[n]
                sim12 = np.abs((dir_i1 * dir_j2).sum()) * weight[n]
                sim21 = np.abs((dir_i2 * dir_j1).sum()) * weight[n]
                sim22 = np.abs((dir_i2 * dir_j2).sum()) * weight[n]
                uv_tuple = (pos, n_indx[0])
                uv_value = -2*(sim11 + sim22 - sim12 - sim21)
                Q.update({uv_tuple: uv_value})

                u_tuple = (pos, pos)
                u_value = -(sim12 + sim21 - sim11 - sim22)
                if u_tuple in Q.keys():
                    Q[u_tuple] = Q[u_tuple] + u_value
                else:
                    Q.update({u_tuple: u_value})

                v_tuple = (n_indx[0], n_indx[0])
                v_value = -(sim12 + sim21 - sim11 - sim22)
                if v_tuple in Q.keys():
                    Q[v_tuple] = Q[v_tuple] + v_value
                else:
                    Q.update({v_tuple: v_value})

response = QBSolv().sample_qubo(Q)
labels = np.zeros(n_node, dtype=np.uint8)
sample = next(iter(response))
for i in sample:
    labels[i] = sample[i]

dir1_match = copy.deepcopy(dir1)
dir2_match = copy.deepcopy(dir2)
switch_mask = np.zeros(mask.shape)

dir1_match[x[labels==1], y[labels==1], z[labels==1], :] = dir2[x[labels==1],
                                                               y[labels==1],
                                                               z[labels==1], :]
dir2_match[x[labels==1], y[labels==1], z[labels==1], :] = dir1[x[labels==1],
                                                               y[labels==1],
                                                               z[labels==1], :]
switch_mask[x[labels==1], y[labels==1], z[labels==1]] = 1

out = nib.Nifti1Image(dir1_match, affine, header)
out.to_filename(args.foutpre + '_dir1_pred_match_patch3.nii.gz')
out = nib.Nifti1Image(dir2_match, affine, header)
out.to_filename(args.foutpre + '_dir2_pred_match_patch3.nii.gz')
