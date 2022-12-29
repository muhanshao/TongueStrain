#!/usr/bin/env python
# -*- coding: utf-8 -*-


import torch
from torch.utils.data import DataLoader
import numpy as np
import sys


sys.path.append('../dirdetect')
from dirdetect.dataset import SHNeighborDataset
from dirdetect.net import DirPatchWeightNet
from dirdetect.trainer import Trainer
from dirdetect.loss import PatchWeightedLossDice

out_prefix = '../model/dirdetect_patch_net'

batch_size = 1024
n_epochs = 200
save_period = 5

t_dataset = SHNeighborDataset(dirname='/your/train/data')
v_dataset = SHNeighborDataset(dirname='/your/validation/data')
t_loader = DataLoader(t_dataset, batch_size=batch_size,
                      shuffle=True, num_workers=32)
v_loader = DataLoader(v_dataset, batch_size=batch_size,
                      shuffle=False, num_workers=32)

in_channels = 45
c1 = 1024

net = DirPatchWeightNet(in_channels=in_channels, c1=c1)

print(net)
weight = np.array([1, 1, 1, 1, 1, 0.3], dtype='float32')
loss_func = PatchWeightedLossDice(weight=weight)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
trainer = Trainer(model=net, loss_func=loss_func,
                  out_prefix=out_prefix,
                  train_loader=t_loader,
                  valid_loader=v_loader,
                  optimizer=optimizer)

trainer.train(n_epochs=n_epochs, save_period=save_period)
