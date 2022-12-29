#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import nibabel as nib
import numpy as np
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from dirdetect.trainer import load_model


class Predictor():
    def __init__(self, model, checkpoint, patch_size=7, batch_size=1024, use_gpu=True):
        self.use_gpu = (use_gpu and torch.cuda.is_available())
        self.model, _ = load_model(model, checkpoint, self.use_gpu)
        self.model.train(False)
        self.patch_size = patch_size
        self.batch_size = batch_size

    def predict(self, fdwish, fmask, foutpre):
        img, affine, header = self.load_nii(fdwish)
        mask = nib.load(fmask).get_data()

        dir1 = torch.zeros(list(mask.shape) + [3]).to(img.device)
        dir2 = torch.zeros(list(mask.shape) + [3]).to(img.device)
        odfsh = torch.zeros(list(mask.shape) + [self.model.out_channels]).to(img.device)

        testSH = SHBatchDataset(img=img, mask=mask,
                                patch_size=self.patch_size,
                                batch_size=self.batch_size)
        for i, inBatch in enumerate(testSH):
            inSH, indx = inBatch
            i_out, j_out, k_out = indx
            out_odfsh, out_dir1, out_dir2 = self.predict_batch(inSH)
            odfsh[i_out, j_out, k_out, :] = out_odfsh.reshape([len(i_out), self.model.out_channels])
            dir1[i_out, j_out, k_out, :] = out_dir1.reshape([len(i_out), 3])
            dir2[i_out, j_out, k_out, :] = out_dir2.reshape([len(i_out), 3])

        self.save_nii(odfsh, affine, header, foutpre + '_odfsh_pred.nii.gz')
        self.save_nii(dir1, affine, header, foutpre + '_dir1_pred.nii.gz')
        self.save_nii(dir2, affine, header, foutpre + '_dir2_pred.nii.gz')

    def predict_batch(self, shbatch):
        with torch.no_grad():
            out_odfsh, out_p1, out_p2, out_dir1, out_dir2 = self.model(shbatch)
            out_odfsh = out_odfsh[..., 1:2, 1:2, 1:2]
        out_dir1 = out_dir1 * (out_p1 > 0.5).float().to(out_dir1.device)
        out_dir2 = out_dir2 * (out_p2 > 0.5).float().to(out_dir2.device)
        return out_odfsh, out_dir1, out_dir2

    def load_nii(self, fn):
        imgfile = nib.load(fn)
        img = imgfile.get_data()
        img = torch.from_numpy(img).float()
        if self.use_gpu:
            img = img.cuda()
        return img, imgfile.affine, imgfile.header

    def save_nii(self, img, affine, header, fn):
        img = img.cpu().numpy().astype('float32')
        out = nib.Nifti1Image(img, affine, header)
        nib.save(out, fn)


class SHBatchDataset(Dataset):
    def __init__(self, img, mask, patch_size, batch_size):
        self.img = img
        self.mask = mask
        self.patch_size = patch_size
        self.batch_size = batch_size

        self.x, self.y, self.z = (mask > 0).nonzero()
        self.n_batch = np.int32(np.ceil(len(self.x) / self.batch_size))

        self.begin = np.arange(self.n_batch) * self.batch_size
        self.end = self.begin + self.batch_size
        self.end[-1] = len(self.x)

        indx = np.arange(self.patch_size) - self.patch_size // 2
        self.sub_i, self.sub_j, self.sub_k = np.meshgrid(indx, indx, indx,
                                                         indexing='ij')

        img_pad = self.img.permute(-1, 0, 1, 2)
        img_pad = F.pad(img_pad.unsqueeze(0),
                        (self.patch_size // 2,)*6,
                        mode='replicate')
        img_pad_temp = img_pad[0, ...]
        self.img_pad = img_pad_temp.permute(1, 2, 3, 0)

    def __len__(self):
        return self.n_batch

    def __getitem__(self, indx: int):
        i_batch = self.x[self.begin[indx]: self.end[indx]]
        j_batch = self.y[self.begin[indx]: self.end[indx]]
        k_batch = self.z[self.begin[indx]: self.end[indx]]
        i_batch_patch = i_batch.reshape([len(i_batch), 1, 1, 1]) + self.sub_i[None, ...] + self.patch_size // 2
        j_batch_patch = j_batch.reshape([len(j_batch), 1, 1, 1]) + self.sub_j[None, ...] + self.patch_size // 2
        k_batch_patch = k_batch.reshape([len(k_batch), 1, 1, 1]) + self.sub_k[None, ...] + self.patch_size // 2
        
        sh = self.img_pad[i_batch_patch, j_batch_patch, k_batch_patch, :]
        sh = sh.permute(0, -1, 1, 2, 3)
        return sh, (i_batch, j_batch, k_batch)
