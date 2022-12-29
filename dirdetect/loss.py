#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch import nn
import torch
import numpy as np


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='sum')
        self.bce_loss = nn.BCELoss(reduction='sum')

    def dirloss(self, dirhat, dirgt, indgt):
        dirhat = dirhat * indgt
        cos = (dirhat * dirgt).sum(dim=-1)
        return -cos.abs().sum()

    def sepcos(self, dir1hat, dir2hat, indgt):
        dir2hat = dir2hat * indgt
        cos = (dir1hat * dir2hat).sum(dim=-1)
        return cos.abs().sum()

    def unpack_gt(self, dir_gt, ind_gt):
        ind1_gt = ind_gt[..., 0].unsqueeze(dim=-1)
        ind2_gt = ind_gt[..., 1].unsqueeze(dim=-1)
        dir1_gt = dir_gt[..., 0:3]
        dir2_gt = dir_gt[..., 3:6]
        return ind1_gt, ind2_gt, dir1_gt, dir2_gt

    def forward(self, out, sh_gt, dir_gt, ind_gt):
        sh_hat, p1_hat, p2_hat, dir1_hat, dir2_hat = out
        ind1_gt, ind2_gt, dir1_gt, dir2_gt = self.unpack_gt(dir_gt, ind_gt)

        sh_mse = self.mse_loss(sh_hat, sh_gt)
        dir1_bce = self.bce_loss(p1_hat, ind1_gt)
        dir2_bce = self.bce_loss(p2_hat, ind2_gt)
        dir1loss = self.dirloss(dir1_hat, dir1_gt, ind1_gt)
        dir2loss = self.dirloss(dir2_hat, dir2_gt, ind2_gt)

        loss = sh_mse + dir1_bce + dir2_bce + dir1loss + dir2loss

        loss_dict = dict()
        loss_dict.update({'total': loss})
        loss_dict.update({'sh_mse': sh_mse})
        loss_dict.update({'dir1_bce': dir1_bce})
        loss_dict.update({'dir2_bce': dir2_bce})
        loss_dict.update({'dir1_cos': dir1loss})
        loss_dict.update({'dir2_cos': dir2loss})

        return loss_dict, loss


class PatchWeightedLossDice(Loss):
    def __init__(self, weight=np.ones(5)):
        super().__init__()
        self.weight = weight

    def unpack_gt(self, dir_gt, num_gt):
        num_gt = num_gt[:, 0:1, ...]
        dir1_gt = dir_gt[:, 0:3, ...]
        dir2_gt = dir_gt[:, 3:6, ...]
        return num_gt, dir1_gt, dir2_gt

    def dirloss(self, dirhat, dirgt):
        eps = 1e-8
        cos = (dirhat * dirgt).sum(dim=1)
        norm = torch.norm(dirhat, p=2, dim=1) + torch.norm(dirgt, p=2, dim=1)
        cos = (cos.abs() + eps) / (norm + eps)
        return -cos.sum()

    def sepcos(self, dir1hat, dir2hat, indgt):
        dir2hat = dir2hat * indgt
        cos = (dir1hat * dir2hat).sum(dim=1)
        return cos.abs().sum()

    def forward(self, out, sh_gt, dir_gt, num_gt):
        bce_wloss = nn.BCEWithLogitsLoss(reduction='sum',
                                         pos_weight=torch.tensor([0.9], device=sh_gt.device))
        sh_hat, p_hat, dir1_hat, dir2_hat = out
        num_gt, dir1_gt, dir2_gt = self.unpack_gt(dir_gt, num_gt)
        sh_mse = self.mse_loss(sh_hat, sh_gt)
        num_bce = bce_wloss(p_hat, num_gt)
        dir1loss = self.dirloss(dir1_hat, dir1_gt)
        dir2loss = self.dirloss(dir2_hat, dir2_gt)
        seploss = self.sepcos(dir1_hat, dir2_hat, num_gt)

        loss = self.weight[0]*sh_mse + self.weight[1]*num_bce + self.weight[2]*dir1loss + self.weight[3]*dir2loss + self.weight[4]*seploss
        loss_dict = dict()
        loss_dict['total'] = loss.item()
        loss_dict['sh_mse'] = sh_mse.item()
        loss_dict['num_bce'] = num_bce.item()
        loss_dict['dir1_cos'] = dir1loss.item()
        loss_dict['dir2_cos'] = dir2loss.item()
        loss_dict['sep_cos'] = seploss.item()

        return loss_dict, loss
