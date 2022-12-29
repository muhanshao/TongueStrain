#!/usr/bin/env python

from torch import nn


class DirPatchWeightNet(nn.Module):
    def __init__(self, in_channels=45, c1=1024, out_channels=45):
        super().__init__()
        self.in_channels = in_channels
        self.c1 = c1
        self.out_channels = out_channels

        self.conv = nn.Sequential(
            nn.Conv3d(self.in_channels, c1, kernel_size=3),
            nn.BatchNorm3d(c1, affine=True),
            nn.ReLU(),
            nn.Conv3d(c1, c1//2, kernel_size=3),
            nn.BatchNorm3d(c1//2, affine=True),
            nn.ReLU(),
            nn.Conv3d(c1//2, c1//2, kernel_size=1),
            nn.BatchNorm3d(c1//2, affine=True),
            nn.ReLU(),
            nn.Conv3d(c1//2, c1//4, kernel_size=1),
            nn.BatchNorm3d(c1//4, affine=True),
            nn.ReLU(),
            nn.Conv3d(c1//4, c1//2, kernel_size=1),
            nn.BatchNorm3d(c1//2, affine=True),
            nn.ReLU(),
            nn.Conv3d(c1//2, self.out_channels, kernel_size=1),
            nn.BatchNorm3d(self.out_channels, affine=True)
        )

        self.weightedSH = nn.Sequential(
            nn.Conv3d(self.out_channels, self.out_channels, kernel_size=3),
            nn.BatchNorm3d(self.out_channels, affine=True)
        )

        self.dir1ind = nn.Sequential(
            nn.Conv3d(self.out_channels, c1//2, kernel_size=1),
            nn.BatchNorm3d(c1//2, affine=True),
            nn.ReLU(),
            nn.Conv3d(c1//2, c1//2, kernel_size=1),
            nn.BatchNorm3d(c1//2, affine=True),
            nn.ReLU(),
            nn.Conv3d(c1//2, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.dir2ind = nn.Sequential(
            nn.Conv3d(self.out_channels, c1//2, kernel_size=1),
            nn.BatchNorm3d(c1//2, affine=True),
            nn.ReLU(),
            nn.Conv3d(c1//2, c1//2, kernel_size=1),
            nn.BatchNorm3d(c1//2, affine=True),
            nn.ReLU(),
            nn.Conv3d(c1//2, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.dir1regress = nn.Sequential(
            nn.Conv3d(self.out_channels, self.c1//8, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(self.c1//8, self.c1//4, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(self.c1//4, self.c1//2, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(self.c1//2, self.c1//4, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(self.c1//4, self.c1//8, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(self.c1//8, 3, kernel_size=1)
        )

        self.dir2regress = nn.Sequential(
            nn.Conv3d(self.out_channels, self.c1//8, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(self.c1//8, self.c1//4, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(self.c1//4, self.c1//2, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(self.c1//2, self.c1//4, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(self.c1//4, self.c1//8, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(self.c1//8, 3, kernel_size=1)
        )

    def forward(self, shpatch):
        odf_sh = self.conv(shpatch)
        print('after conv', torch.cuda.max_memory_allocated() / 1024 / 1024,
                odf_sh.shape)
        odf_sh_weighted = self.weightedSH(odf_sh)
        print('after SH', torch.cuda.max_memory_allocated() / 1024 / 1024)
        p1 = self.dir1ind(odf_sh_weighted)
        # p2 = self.dir2ind(odf_sh_weighted)
        print('after dir', torch.cuda.max_memory_allocated() / 1024 / 1024,
                p1.shape)
        # dir1 = self.dir1regress(odf_sh_weighted)
        # dir2 = self.dir2regress(odf_sh_weighted)
        print('after regress', torch.cuda.max_memory_allocated() / 1024 / 1024)
        return odf_sh, p1, p2, dir1, dir2


t_dataset = SHNeighborDataset(dirname='../traindata')                           
v_dataset = SHNeighborDataset(dirname='../validdata')                           
 25 t_loader = DataLoader(t_dataset, batch_size=batch_size,                         
 26                       shuffle=True, num_workers=1)                              
 27 v_loader = DataLoader(v_dataset, batch_size=batch_size,                         
 28                       shuffle=True, num_workers=1)
