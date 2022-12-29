#!/usr/bin/env python
# -*- coding: utf-8 -*-


import torch
import os
import numpy as np
from collections import defaultdict


class Trainer:
    def __init__(self, model, loss_func=None, out_prefix='',
                 use_gpu=True, train_loader=None, valid_loader=None,
                 optimizer=None, pre_model=''):
        self.use_gpu = (use_gpu and torch.cuda.is_available())
        self.model = model.float()
        self.model.train(True)
        self.loss_func = loss_func
        self.out_prefix = out_prefix
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = optimizer
        self.pre_model = pre_model

    def train_epoch(self, data_loader, trainBool, loss_record):
        for i, (inData, inODFSH, inDir, inInd) in enumerate(data_loader):
            if self.use_gpu:
                inData = inData.cuda()
                inODFSH = inODFSH.cuda()
                inDir = inDir.cuda()
                inInd = inInd.cuda()

            if trainBool:
                self.optimizer.zero_grad()

            out = self.model(inData)
            loss_dict, loss = self.loss_func(out, inODFSH, inDir, inInd)
            for key, value in loss_dict.items():
                loss_record[key].append(value)

            if trainBool:
                loss.backward()
                self.optimizer.step()

    def train(self, n_epochs, save_period=1):
        if os.path.isfile(self.pre_model):
            self.model, start_epoch = load_model(self.model, self.pre_model,
                                                 self.use_gpu)
            start_epoch = start_epoch + 1
            self.optimizer = load_optimizer(self.optimizer, self.pre_model)
        else:
            self.model = self.model.cuda() if self.use_gpu else self.model
            start_epoch = 1
        use_valid = self.valid_loader is not None

        for epoch in range(start_epoch, n_epochs + 1):
            print('Epoch = ', epoch)
            if use_valid: self.model.train(True)

            t_loss_record = defaultdict(list)

            self.train_epoch(self.train_loader,
                             trainBool=True,
                             loss_record=t_loss_record)

            if epoch % save_period == 0:
                epoch_str = "%03d" % epoch
                fn_model = self.out_prefix + '_' + epoch_str + '.pt'
                self.save_model(fn_model, epoch)

            # Validation after each epoch
            v_loss_record = defaultdict(list)
            if use_valid:
                self.model.train(False)
                with torch.set_grad_enabled(False):
                    self.train_epoch(self.valid_loader,
                                     trainBool=False,
                                     loss_record=v_loss_record)

            # Write loss to csv file
            fn_loss = self.out_prefix + '_log.csv'
            write_csv(fn_loss, epoch, t_loss_record, v_loss_record)

    def save_model(self, fn, epoch):
        state = {'epoch': epoch,
                 'model_state_dict': self.model.state_dict(),
                 'opt_state_dict': self.optimizer.state_dict()}
        torch.save(state, fn)


def load_model(model, fn, use_gpu):
    checkpoint = torch.load(fn, map_location='cpu')
    last_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model_state_dict'])
    if use_gpu:
        model.cuda()
    return model, last_epoch


def load_optimizer(optimizer, fn):
    checkpoint = torch.load(fn)
    optimizer.load_state_dict(checkpoint['opt_state_dict'])
    return optimizer


def write_csv(fn, epoch, train_loss, valid_loss={}):
    import csv
    if not os.path.isfile(fn):
        head = ['epoch']
        for key in train_loss.keys():
            head.append('t_' + key)
        if valid_loss:
            for key in valid_loss.keys():
                head.append('v_' + key)
        with open(fn, "w") as f:
            wr = csv.writer(f)
            wr.writerow(head)
        f.close()

    out = [epoch]
    for value in train_loss.values():
        out.append(np.mean(value))

    if valid_loss:
        for value in valid_loss.values():
            out.append(np.mean(value))

    with open(fn, "a") as f:
        wr = csv.writer(f)
        wr.writerow(out)
    f.close()
