#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
import pdb
from torch.distributions.categorical import Categorical


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


class PredLoss(Loss):
    def __init__(self, weight=np.ones(6)):
        super(PredLoss, self).__init__()
        self.weight = weight

    def forward(self, out, sh_gt, dir_gt, ind_gt):
        sh_hat, p1_hat, p2_hat, dir1_hat, dir2_hat = out
        ind1_gt, ind2_gt, dir1_gt, dir2_gt = self.unpack_gt(dir_gt, ind_gt)

        sh_mse = self.mse_loss(sh_hat, sh_gt)
        dir1_bce = self.bce_loss(p1_hat, ind1_gt)
        dir2_bce = self.bce_loss(p2_hat, ind2_gt)
        dir1loss = self.dirloss(dir1_hat, dir1_gt, ind1_gt)
        dir2loss = self.dirloss(dir2_hat, dir2_gt, ind2_gt)
        seploss = self.sepcos(dir1_hat, dir2_hat, ind2_gt)

        loss = self.weight[0]*sh_mse + self.weight[1]*dir1_bce + self.weight[2]*dir2_bce + self.weight[3]*dir1loss + self.weight[4]*dir2loss + self.weight[5]*seploss
        loss_dict = dict()
        loss_dict.update({'total': loss})
        loss_dict.update({'sh_mse': sh_mse})
        loss_dict.update({'dir1_bce': dir1_bce})
        loss_dict.update({'dir2_bce': dir2_bce})
        loss_dict.update({'dir1_cos': dir1loss})
        loss_dict.update({'dir2_cos': dir2loss})
        loss_dict.update({'sep_cos': seploss})

        return loss_dict, loss


class PatchPredLoss(Loss):
    def __init__(self, weight=np.ones(6)):
        super().__init__()
        self.weight = weight

    def unpack_gt(self, dir_gt, ind_gt):
        ind1_gt = ind_gt[:, 0, ...].unsqueeze(dim=1)
        ind2_gt = ind_gt[:, 1, ...].unsqueeze(dim=1)
        dir1_gt = dir_gt[:, 0:3, ...]
        dir2_gt = dir_gt[:, 3:6, ...]
        return ind1_gt, ind2_gt, dir1_gt, dir2_gt

    def dirloss(self, dirhat, dirgt, indgt):
        dirhat = dirhat * indgt
        cos = (dirhat * dirgt).sum(dim=1)
        return -cos.abs().sum()

    def sepcos(self, dir1hat, dir2hat, indgt):
        dir2hat = dir2hat * indgt
        cos = (dir1hat * dir2hat).sum(dim=1)
        return cos.abs().sum()

    def forward(self, out, sh_gt, dir_gt, ind_gt):
        sh_hat, p1_hat, p2_hat, dir1_hat, dir2_hat = out
        ind1_gt, ind2_gt, dir1_gt, dir2_gt = self.unpack_gt(dir_gt, ind_gt)

        sh_mse = self.mse_loss(sh_hat, sh_gt)
        dir1_bce = self.bce_loss(p1_hat, ind1_gt)
        dir2_bce = self.bce_loss(p2_hat, ind2_gt)
        dir1loss = self.dirloss(dir1_hat, dir1_gt, ind1_gt)
        dir2loss = self.dirloss(dir2_hat, dir2_gt, ind2_gt)
        seploss = self.sepcos(dir1_hat, dir2_hat, ind2_gt)

        loss = self.weight[0]*sh_mse + self.weight[1]*dir1_bce + self.weight[2]*dir2_bce + self.weight[3]*dir1loss + self.weight[4]*dir2loss + self.weight[5]*seploss
        loss_dict = dict()
        loss_dict.update({'total': loss})
        loss_dict.update({'sh_mse': sh_mse})
        loss_dict.update({'dir1_bce': dir1_bce})
        loss_dict.update({'dir2_bce': dir2_bce})
        loss_dict.update({'dir1_cos': dir1loss})
        loss_dict.update({'dir2_cos': dir2loss})
        loss_dict.update({'sep_cos': seploss})

        return loss_dict, loss


class PatchWeightedLoss(Loss):
    def __init__(self, weight=np.ones(6)):
        super().__init__()
        self.weight = weight

    def unpack_gt(self, dir_gt, ind_gt):
        ind1_gt = ind_gt[:, 0:1, 1:2, 1:2, 1:2]
        ind2_gt = ind_gt[:, 1:2, 1:2, 1:2, 1:2]
        dir1_gt = dir_gt[:, 0:3, 1:2, 1:2, 1:2]
        dir2_gt = dir_gt[:, 3:6, 1:2, 1:2, 1:2]
        return ind1_gt, ind2_gt, dir1_gt, dir2_gt

    def dirloss(self, dirhat, dirgt, indgt):
        dirhat = dirhat * indgt
        cos = (dirhat * dirgt).sum(dim=1)
        return -cos.abs().sum()

    def sepcos(self, dir1hat, dir2hat, indgt):
        dir2hat = dir2hat * indgt
        cos = (dir1hat * dir2hat).sum(dim=1)
        return cos.abs().sum()

    def forward(self, out, sh_gt, dir_gt, ind_gt):
        sh_hat, p1_hat, p2_hat, dir1_hat, dir2_hat = out
        ind1_gt, ind2_gt, dir1_gt, dir2_gt = self.unpack_gt(dir_gt, ind_gt)
        sh_mse = self.mse_loss(sh_hat, sh_gt)
        dir1_bce = self.bce_loss(p1_hat, ind1_gt)
        dir2_bce = self.bce_loss(p2_hat, ind2_gt)
        dir1loss = self.dirloss(dir1_hat, dir1_gt, ind1_gt)
        dir2loss = self.dirloss(dir2_hat, dir2_gt, ind2_gt)
        seploss = self.sepcos(dir1_hat, dir2_hat, ind2_gt)

        loss = self.weight[0]*sh_mse + self.weight[1]*dir1_bce + self.weight[2]*dir2_bce + self.weight[3]*dir1loss + self.weight[4]*dir2loss + self.weight[5]*seploss
        loss_dict = dict()
        loss_dict.update({'total': loss})
        loss_dict.update({'sh_mse': sh_mse})
        loss_dict.update({'dir1_bce': dir1_bce})
        loss_dict.update({'dir2_bce': dir2_bce})
        loss_dict.update({'dir1_cos': dir1loss})
        loss_dict.update({'dir2_cos': dir2loss})
        loss_dict.update({'sep_cos': seploss})

        return loss_dict, loss


class PatchWeightedLossDice(Loss):
    def __init__(self, weight=np.ones(6)):
        super().__init__()
        self.weight = weight

    def unpack_gt(self, dir_gt, ind_gt):
        ind1_gt = ind_gt[:, 0:1, 1:2, 1:2, 1:2]
        ind2_gt = ind_gt[:, 1:2, 1:2, 1:2, 1:2]
        dir1_gt = dir_gt[:, 0:3, 1:2, 1:2, 1:2]
        dir2_gt = dir_gt[:, 3:6, 1:2, 1:2, 1:2]
        return ind1_gt, ind2_gt, dir1_gt, dir2_gt

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

    def forward(self, out, sh_gt, dir_gt, ind_gt):
        sh_hat, p1_hat, p2_hat, dir1_hat, dir2_hat = out
        ind1_gt, ind2_gt, dir1_gt, dir2_gt = self.unpack_gt(dir_gt, ind_gt)
        sh_mse = self.mse_loss(sh_hat, sh_gt)
        dir1_bce = self.bce_loss(p1_hat, ind1_gt)
        dir2_bce = self.bce_loss(p2_hat, ind2_gt)
        dir1loss = self.dirloss(dir1_hat, dir1_gt)
        dir2loss = self.dirloss(dir2_hat, dir2_gt)
        seploss = self.sepcos(dir1_hat, dir2_hat, ind2_gt)

        loss = self.weight[0]*sh_mse + self.weight[1]*dir1_bce + self.weight[2]*dir2_bce + self.weight[3]*dir1loss + self.weight[4]*dir2loss + self.weight[5]*seploss
        loss_dict = dict()
        loss_dict['total'] = loss.item()
        loss_dict['sh_mse'] = sh_mse.item()
        loss_dict['dir1_bce'] = dir1_bce.item()
        loss_dict['dir2_bce'] = dir2_bce.item()
        loss_dict['dir1_cos'] = dir1loss.item()
        loss_dict['dir2_cos'] = dir2loss.item()
        loss_dict['sep_cos'] = seploss.item()

        return loss_dict, loss


class PatchWeightedLossHemi(PatchWeightedLossDice):
    def __init__(self, weight=np.ones(6)):
        super().__init__(weight=weight)

    def dirloss(self, dirhat, dirgt):
        cos = (dirhat * dirgt).sum(dim=1)
        return -cos.sum()


class MultiStageLoss(PatchWeightedLoss):
    def __init__(self, weight=np.ones(9)):
        super().__init__(weight=weight)

    def sepcos(self, dir1hat, dir2hat, indgt, angle):
        dir2hat = dir2hat * indgt
        cos = (dir1hat * dir2hat).sum(dim=1)
        angle = torch.FloatTensor([angle/180*np.pi]).to(dir1hat.device)
        mse = (cos.abs() - torch.cos(angle))**2
        return mse.sum()

    def unpack_gt(self, dir_gt, ind_gt, s2_gt):
        ind1_gt = ind_gt[:, 0:1, 1:2, 1:2, 1:2]
        ind2_gt = ind_gt[:, 1:2, 1:2, 1:2, 1:2]
        dir1_gt = dir_gt[:, 0:3, 1:2, 1:2, 1:2]
        dir2_gt = dir_gt[:, 3:6, 1:2, 1:2, 1:2]
        s2_gt = s2_gt[..., 1:2, 1:2, 1:2]
        return ind1_gt, ind2_gt, dir1_gt, dir2_gt, s2_gt

    def forward(self, out, sh_gt, dir_gt, ind_gt, s2_gt):
        sh_hat, p1_hat, p2_hat, p2_stage2_hat, dir1_hat, dir2_hat, dir2_stage2_hat = out
        ind1_gt, ind2_gt, dir1_gt, dir2_gt, s2_gt = self.unpack_gt(dir_gt,
                                                                   ind_gt,
                                                                   s2_gt)
        sh_mse = self.mse_loss(sh_hat, sh_gt)
        dir1_bce = self.bce_loss(p1_hat, ind1_gt)
        dir1loss = self.dirloss(dir1_hat, dir1_gt, ind1_gt)

        dir2_bce = self.bce_loss(p2_hat, ind2_gt)
        dir2loss = self.dirloss(dir2_hat, dir2_gt, ind2_gt)

        dir2_stage2_bce = self.bce_loss(p2_stage2_hat, s2_gt)
        dir2_stage2_loss = self.dirloss(dir2_stage2_hat, dir2_gt, s2_gt)

        seploss1 = self.sepcos(dir1_hat, dir2_hat, ind2_gt, 90)
        seploss2 = self.sepcos(dir1_hat, dir2_stage2_hat, s2_gt, 60)

        loss = self.weight[0]*sh_mse + self.weight[1]*dir1_bce + self.weight[2]*dir2_bce + self.weight[3]*dir2_stage2_bce + self.weight[4]*dir1loss +self.weight[5]*dir2loss + self.weight[6]*dir2_stage2_loss + self.weight[7]*seploss1 +self.weight[8]*seploss2

        loss_dict = dict()
        loss_dict.update({'total': loss})
        loss_dict.update({'sh_mse': sh_mse})
        loss_dict.update({'dir1_bce': dir1_bce})
        loss_dict.update({'dir2_bce': dir2_bce})
        loss_dict.update({'dir2_stage2_bce': dir2_stage2_bce})
        loss_dict.update({'dir1_cos': dir1loss})
        loss_dict.update({'dir2_cos': dir2loss})
        loss_dict.update({'dir2_stage2_cos': dir2_stage2_loss})
        loss_dict.update({'sep_cos1': seploss1})
        loss_dict.update({'sep_cos2': seploss2})
        return loss_dict, loss


class PatchWeightedSepLoss(Loss):
    def __init__(self, weight=np.ones(6)):
        super().__init__()
        self.weight = weight

    def unpack_gt(self, dir_gt, ind_gt):
        ind1_gt = ind_gt[:, 0:1, 1:2, 1:2, 1:2]
        ind2_gt = ind_gt[:, 1:2, 1:2, 1:2, 1:2]
        dir1_gt = dir_gt[:, 0:3, 1:2, 1:2, 1:2]
        dir2_gt = dir_gt[:, 3:6, 1:2, 1:2, 1:2]
        return ind1_gt, ind2_gt, dir1_gt, dir2_gt

    def dirloss(self, dirhat, dirgt, indgt):
        dirhat = dirhat * indgt
        cos = (dirhat * dirgt).sum(dim=1)
        return -cos.abs().sum()

    def sepcos(self, dir1hat, dir2hat, dir1gt, dir2gt, indgt):
        dir2hat = dir2hat * indgt
        dir2gt = dir2gt * indgt
        coshat = ((dir1hat * dir2hat).sum(dim=1)).abs()
        cosgt = ((dir1gt * dir2gt).sum(dim=1)).abs()
        return self.mse_loss(coshat, cosgt)

    def forward(self, out, sh_gt, dir_gt, ind_gt):
        sh_hat, p1_hat, p2_hat, dir1_hat, dir2_hat = out
        ind1_gt, ind2_gt, dir1_gt, dir2_gt = self.unpack_gt(dir_gt, ind_gt)
        sh_mse = self.mse_loss(sh_hat, sh_gt)
        dir1_bce = self.bce_loss(p1_hat, ind1_gt)
        dir2_bce = self.bce_loss(p2_hat, ind2_gt)
        dir1loss = self.dirloss(dir1_hat, dir1_gt, ind1_gt)
        dir2loss = self.dirloss(dir2_hat, dir2_gt, ind2_gt)
        seploss = self.sepcos(dir1_hat, dir2_hat, dir1_gt, dir2_gt, ind2_gt)

        loss = self.weight[0]*sh_mse + self.weight[1]*dir1_bce + self.weight[2]*dir2_bce + self.weight[3]*dir1loss + self.weight[4]*dir2loss + self.weight[5]*seploss
        loss_dict = dict()
        loss_dict.update({'total': loss})
        loss_dict.update({'sh_mse': sh_mse})
        loss_dict.update({'dir1_bce': dir1_bce})
        loss_dict.update({'dir2_bce': dir2_bce})
        loss_dict.update({'dir1_cos': dir1loss})
        loss_dict.update({'dir2_cos': dir2loss})
        loss_dict.update({'sep_cos': seploss})

        return loss_dict, loss


class VoxelWeightedLoss(PatchWeightedLossDice):
    def __init__(self, weight=np.ones(6)):
        super().__init__(weight=weight)

    def unpack_gt(self, dir_gt, ind_gt):
        ind1_gt = ind_gt[:, 0:1, ...]
        ind2_gt = ind_gt[:, 1:2, ...]
        dir1_gt = dir_gt[:, 0:3, ...]
        dir2_gt = dir_gt[:, 3:6, ...]
        return ind1_gt, ind2_gt, dir1_gt, dir2_gt
