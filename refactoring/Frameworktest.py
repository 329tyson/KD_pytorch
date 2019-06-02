from trainers import SingleResTrainer
from trainers import KDTrainer
from trainers import FeatureMSETrainer
from trainers import GradientMSETrainer
from argparser import parse
from alexnet import AlexNet
from preprocess import load_weight
from preprocess import generate_dataset
from logger import Logger
from logger import AverageMeter
from plot import Plotter

import datetime
import torch.optim as optim
import torch
import torch.nn as nn
import os
import numpy as np


if __name__ == '__main__':
    args = parse()
    writerName = '_'.join(('runs/',datetime.datetime.now().strftime('%mm-%dd'), args.message))

    net = AlexNet(0.5, args.classes, ['fc8'])
    teacher_net = AlexNet(0.5, args.classes, ['fc8'])
    load_weight(net, args.pretrain_path)
    load_weight(teacher_net, args.pretrain_path)

    optimizer= optim.SGD(
        [{'params': net.finetuning_params()},
         {'params': net.fc8.parameters()   , 'lr': args.lr * 10}],
        lr=args.lr,
        momentum = 0.9,
        weight_decay = 0.0005)

    dataloaders = generate_dataset(
        dataset          = args.dataset,
        batch_size       = args.batch,
        annotation_train = args.annotation_train,
        annotation_val   = args.annotation_val,
        image_path       = args.data,
        low_ratio        = args.low_ratio,
        is_KD            = args.kd
        )

    logger = Logger(
        name    = args.log_dir + '/{}'.format(args.message),
        is_test = args.test,
        stdout  = False)
    writer =  Plotter(writerName, args.test)

    logger.write_args(args)

    SingleTrainer = SingleResTrainer(
        network           = net,
        optimizer         = optimizer,
        lr                = args.lr,
        logger            = logger,
        writer            = writer,
        lossfunc          = nn.CrossEntropyLoss(),
        val_period        = args.val_period,
        train_loader      = dataloaders[0],
        eval_train_loader = dataloaders[1],
        validation_loader = dataloaders[2])


    KDBaseline = KDTrainer(
        teacher           = teacher_net,
        temperature       = args.kd_temperature,
        kdlossfunc        = nn.KLDivLoss(),
        network           = net,
        optimizer         = optimizer,
        lr                = args.lr,
        logger            = logger,
        writer            = writer,
        lossfunc          = nn.CrossEntropyLoss(),
        val_period        = args.val_period,
        train_loader      = dataloaders[0],
        eval_train_loader = dataloaders[1],
        validation_loader = dataloaders[2])

    FeatureMSEBaseline = FeatureMSETrainer(
        teacher           = teacher_net,
        temperature       = args.kd_temperature,
        regression_layers = args.rg_layers,
        ftlossfunc       = nn.MSELoss(),
        kdlossfunc        = nn.KLDivLoss(),
        network           = net,
        optimizer         = optimizer,
        lr                = args.lr,
        logger            = logger,
        writer            = writer,
        lossfunc          = nn.CrossEntropyLoss(),
        val_period        = args.val_period,
        train_loader      = dataloaders[0],
        eval_train_loader = dataloaders[1],
        validation_loader = dataloaders[2])

    GradientMSEBaseline = GradientMSETrainer(
        gradient_layers   = args.grad_layers,
        teacher           = teacher_net,
        temperature       = args.kd_temperature,
        regression_layers = args.rg_layers,
        ftlossfunc        = nn.MSELoss(),
        kdlossfunc        = nn.KLDivLoss(),
        network           = net,
        optimizer         = optimizer,
        lr                = args.lr,
        logger            = logger,
        writer            = writer,
        lossfunc          = nn.CrossEntropyLoss(),
        val_period        = args.val_period,
        train_loader      = dataloaders[0],
        eval_train_loader = dataloaders[1],
        validation_loader = dataloaders[2])

    if args.kd is False:
        SingleTrainer.train(args.epochs, args.lr_decay)
    elif len(args.rg_layers) > 0 and len(args.grad_layers) > 0:
        GradientMSEBaseline.train(args.epochs, args.lr_decay)
    elif len(args.rg_layers) > 0:
        FeatureMSEBaseline.train(args.epochs, args.lr_decay)
    else:
        KDBaseline.train(args.epochs, args.lr_decay)
