#!/usr/bin/env python
# -*- coding: utf-8 -*-


import sys
import argparse

sys.path.append('../dirdetect')
from dirdetect.preprocessing import BasicDWI
from dirdetect.net import DirPatchWeightNet
from dirdetect.predictor import Predictor


## Input
parser = argparse.ArgumentParser()
parser.add_argument('--fdwi', type=str, required=True)
parser.add_argument('--fmask', type=str, required=True)
parser.add_argument('--fbval', type=str, required=True)
parser.add_argument('--fbvec', type=str, required=True)
parser.add_argument('--fsh', type=str, required=True)
parser.add_argument('--foutpre', type=str, required=True)
args = parser.parse_args()

## Generate SH
dwi = BasicDWI(args.fbval, args.fbvec, args.fmask)
dwi.dwi2sh(args.fdwi, 8, args.fsh)

## Predictor
epoch = '150'
in_channels = 45
c1 = 1024
net = DirPatchWeightNet(in_channels=in_channels, c1=c1)
checkpoint = '../model/dirdetect_patch_net_' + epoch + '.pt'
pred = Predictor(net, checkpoint, batch_size=500)

## Run prediction
pred.predict(args.fsh, args.fmask, args.foutpre)
