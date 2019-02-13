from argparser import parse
from alexnet import AlexNet
from VGG_gap import VGG_gap
from alexnet import RACNN
from train import training
from train import training_KD
from train import training_Gram_KD
from preprocess import load_weight
from preprocess import generate_dataset
from logger import getlogger

from torchvision import models

import os
import torch.optim as optim
import torch


if __name__ == '__main__':
    '''
    Parse argument from user input
    args are set to default as followed
    - args.root = CWD
    - args.data = ./data
    - args.annotation_train = ./annotations_train
    - args.annotation_val = ./annotations_val
    - args.result = ./results
    - args.dataset = None
    - args.classes = 0
    - args.lr = 0.001
    - args.batch = 111
    - args.epochs = 200
    - args.resume = False
    - args.checkpoint = 10
    - args.low_ratio = 0
    - args.verbose
    - args.kd_enabled = False
    - args.kd_temperature = 3
    - args.log_dir =./logs
    - args.gpu = 0
    - args.noise = False
    - args.style_weight = 1
    - args.gram_enabled = False
    - args.path_norm = 0 # no normalization
    - args.path_num = 1
    - args.hint = False
    - args.save = False
    - args.vgg_gap = False
    '''
    args = parse()
    args.annotation_train = os.path.join(args.root, args.annotation_train)
    args.annotation_val = os.path.join(args.root, args.annotation_val)
    args.data = os.path.join(args.root, args.data)
    args.result = os.path.join(args.root, args.result)
    args.log_dir = os.path.join(args.root, args.log_dir)

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

    if args.pretrain_path != 'NONE':
        args.pretrain_path = os.path.join(args.root, args.pretrain_path)

    if args.dataset.lower() == 'cub':
        args.classes = 200
    else:
        args.classes = 196

    if args.vgg_gap :
        vgg16 = models.vgg16(True)
        net = VGG_gap(vgg16, args.classes)

        if args.pretrain_path != 'NONE':
            net.load_state_dict(torch.load(args.pretrain_path))

        optimizer = optim.SGD(
            net.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=0.0005)

    else:
        net = AlexNet(0.5, args.classes, ['fc8'])
        load_weight(net, args.pretrain_path)

        optimizer= optim.SGD(
            [{'params':net.conv1.parameters()},
             {'params':net.conv2.parameters()},
             {'params':net.conv3.parameters()},
             {'params':net.conv4.parameters()},
             {'params':net.conv5.parameters()},
             {'params':net.fc6.parameters()},
             {'params':net.fc7.parameters()},
             {'params':net.fc8.parameters(), 'lr':args.lr * 10}],
            lr=args.lr,
            momentum = 0.9,
            weight_decay = 0.0005)

    net.cuda()

    if args.verbose is True:
        print('Training arguments settings')
        for arg in vars(args):
            print('\t',arg, getattr(args, arg))

    if args.kd_enabled is True:
        if args.low_ratio == 0:
            print('Invalid argument, choose low resolution (50 | 25)')
        elif args.noise is False:
            print('\nTraining Knowledge Distillation model')
            print('\t on ',args.dataset,' with hyper parameters above')
            print('\tLow resolution scaling = {} x {}'.format(args.low_ratio, args.low_ratio))
            if not args.vgg_gap:
                teacher_net = AlexNet(0.5, args.classes, ['fc8'])
                # load_weight(teacher_net, args.pretrain_path)
                teacher_net.load_state_dict(net.state_dict())
                teacher_net.cuda()
            else:
                teacher_net = VGG_gap(vgg16, args.classes)
                teacher_net.load_state_dict(net.state_dict())
                teacher_net.cuda()
            try:
                train_loader, eval_train_loader, eval_validation_loader, num_training, num_validation = generate_dataset(
                    args.dataset,
                    args.batch,
                    args.annotation_train,
                    args.annotation_val,
                    args.data,
                    args.low_ratio,
                    args.ten_batch_eval,
                    args.verbose,
                    args.kd_enabled)
            except ValueError:
                print('inapproriate dataset, please put cub or stanford')

            print('\nTraining starts')
            logger = getlogger(args.log_dir + '/KD_DATASET_{}_LOW_{}'.format(args.dataset, str(args.low_ratio)))
            for arg in vars(args):
                logger.info('{} - {}'.format(str(arg), str(getattr(args, arg))))
            logger.info('\nTraining Knowledge Distillation model, Low resolution of {}x{}'.format(str(args.low_ratio), str(args.low_ratio)))
            logger.info('\t on '+args.dataset.upper()+' dataset, with hyper parameters above\n\n')
            training_KD(
                teacher_net,
                net,
                optimizer,
                args.kd_temperature,
                args.lr,
                args.lr_decay,
                args.epochs,
                args.ten_batch_eval,
                train_loader,
                eval_train_loader,
                eval_validation_loader,
                num_training,
                num_validation,
                args.low_ratio,
                args.result,
                logger,
                args.vgg_gap,
                args.save)
        else:
            print('\nTraining Noise added Knowledge Distillation model')
            print('\t on ',args.dataset,' with hyper parameters above')
            print('\tLow resolution scaling = {} x {}'.format(args.low_ratio, args.low_ratio))
            noise_net = RACNN(0.5, args.classes, ['fc8'],
                              alex_weights_path = args.pretrain_path,
                              alex_pretrained = True)
            teacher_net = AlexNet(0.5, args.classes, ['fc8'])
            noise_net.cuda()
            optimizer = optim.SGD([{'params':noise_net.srLayer.parameters(), 'lr': 2.0 * args.lr},
                                   {'params':noise_net.get_all_params_except_last_fc(), 'lr': 0.0},
                                   {'params':noise_net.classificationLayer.fc8.weight, 'lr': 1.0 * args.lr,
                                    'weight_decay': 1.0 * 0.0005},
                                   {'params':noise_net.classificationLayer.fc8.bias, 'lr': 2.0 * args.lr,
                                    'weight_decay': 0.0}],
                                  momentum=0.95, weight_decay=0.0005)
            load_weight(teacher_net, args.pretrain_path)
            try:
                train_loader, eval_train_loader, eval_validation_loader, num_training, num_validation = generate_dataset(
                    args.dataset,
                    args.batch,
                    args.annotation_train,
                    args.annotation_val,
                    args.data,
                    args.low_ratio,
                    args.ten_batch_eval,
                    args.verbose,
                    args.kd_enabled)
            except ValueError:
                print('inapproriate dataset, please put cub or stanford')

            print('\nTraining starts')
            logger = getlogger(args.log_dir + '/NOISE_KD_DATASET_{}_LOW_{}'.format(args.dataset, str(args.low_ratio)))
            for arg in vars(args):
                logger.info('{} - {}'.format(str(arg), str(getattr(args, arg))))
            logger.info('\nTraining Noise added Knowledge Distillation model, Low resolution of {}x{}'.format(str(args.low_ratio), str(args.low_ratio)))
            logger.info('\t on '+args.dataset.upper()+' dataset, with hyper parameters above\n\n')
            training_KD(
                teacher_net,
                noise_net,
                optimizer,
                args.kd_temperature,
                args.lr,
                args.lr_decay,
                args.epochs,
                args.ten_batch_eval,
                train_loader,
                eval_train_loader,
                eval_validation_loader,
                num_training,
                num_validation,
                args.low_ratio,
                args.result,
                logger,
                args.vgg_gap,
                args.save)

            print('\nTraining starts')
            logger = getlogger(args.log_dir + '/KD_DATASET_{}_LOW_{}'.format(args.dataset, str(args.low_ratio)))
            for arg in vars(args):
                logger.info('{} - {}'.format(str(arg), str(getattr(args, arg))))
            logger.info('\nTraining Knowledge Distillation model, Low resolution of {}x{}'.format(str(args.low_ratio), str(args.low_ratio)))
            logger.info('\t on '+args.dataset.upper()+' dataset, with hyper parameters above\n\n')
            training_KD(
                teacher_net,
                net,
                optimizer,
                args.kd_temperature,
                args.lr,
                args.lr_decay,
                args.epochs,
                args.ten_batch_eval,
                train_loader,
                eval_train_loader,
                eval_validation_loader,
                num_training,
                num_validation,
                args.low_ratio,
                args.result,
                logger,
                args.vgg_gap,
                args.save)
            if args.gram_enabled:
                print('\nTraining starts')
                logger = getlogger(args.log_dir + '/KD_WITH_GRAM_DATASET_{}_LOW_{}'.format(args.dataset, str(args.low_ratio)))
                for arg in vars(args):
                    logger.info('{} - {}'.format(str(arg), str(getattr(args, arg))))
                logger.info(
                    '\nTraining model with KD & Gram, Low resolution of {}x{}'.format(str(args.low_ratio),
                                                                                              str(args.low_ratio)))
                logger.info('\t on ' + args.dataset.upper() + ' dataset, with hyper parameters above\n\n')
                training_Gram_KD(
                    teacher_net,
                    net,
                    optimizer,
                    args.kd_temperature,
                    args.lr,
                    args.lr_decay,
                    args.epochs,
                    args.ten_batch_eval,
                    train_loader,
                    eval_train_loader,
                    eval_validation_loader,
                    num_training,
                    num_validation,
                    args.low_ratio,
                    args.result,
                    logger,
                    args.style_weight,
                    args.norm_type,
                    args.patch_num,
                    args.gram_features,
                    args.hint,
                    args.at_enabled,
                    args.at_ratio,
                    args.save
                    )

            else:
                print('\nTraining starts')
                logger = getlogger(args.log_dir + '/KD_DATASET_{}_LOW_{}'.format(args.dataset, str(args.low_ratio)))
                for arg in vars(args):
                    logger.info('{} - {}'.format(str(arg), str(getattr(args, arg))))
                logger.info('\nTraining Knowledge Distillation model, Low resolution of {}x{}'.format(str(args.low_ratio), str(args.low_ratio)))
                logger.info('\t on '+args.dataset.upper()+' dataset, with hyper parameters above\n\n')
                training_KD(
                    teacher_net,
                    net,
                    optimizer,
                    args.kd_temperature,
                    args.lr,
                    args.lr_decay,
                    args.epochs,
                    args.ten_batch_eval,
                    train_loader,
                    eval_train_loader,
                    eval_validation_loader,
                    num_training,
                    num_validation,
                    args.low_ratio,
                    args.result,
                    logger,
                    args.vgg_gap,
                    args.save
                    )

    else :
        if args.low_ratio == 0:
            print('\nTraining High Resolution images')
            print('\t on ',args.dataset.upper(),' dataset, with hyper parameters above')

            try:
                train_loader, eval_train_loader, eval_validation_loader, num_training, num_validation = generate_dataset(
                    args.dataset,
                    args.batch,
                    args.annotation_train,
                    args.annotation_val,
                    args.data,
                    args.low_ratio,
                    args.ten_batch_eval,
                    args.verbose)
            except ValueError:
                print('inapproriate dataset, please put cub or stanford')
                exit

            print('\nTraining starts')
            logger = getlogger(args.log_dir + '/DATASET_{}_HIGH_RES'.format(args.dataset))
            for arg in vars(args):
                logger.info('{} - {}'.format(str(arg), str(getattr(args, arg))))
            logger.info('\nTraining High Resolution images')
            logger.info('\t on '+args.dataset.upper()+' dataset, with hyper parameters above\n\n')
            training(net, optimizer,
                     args.lr, args.lr_decay, args.epochs, args.ten_batch_eval,
                     train_loader, eval_train_loader, eval_validation_loader, num_training, num_validation,
                     args.low_ratio, args.result,
                     logger,
                     args.vgg_gap,
                     args.save
                     )
        else:
            print('\nTraining Low Resolution images')
            print('\t on ',args.dataset,' with hyper parameters above')
            print('\tLow resolution scaling = {} x {}'.format(args.low_ratio, args.low_ratio))

            try:
                train_loader, eval_train_loader, eval_validation_loader, num_training, num_validation = generate_dataset(
                    args.dataset,
                    args.batch,
                    args.annotation_train,
                    args.annotation_val,
                    args.data,
                    args.low_ratio,
                    args.ten_batch_eval,
                    args.verbose)
            except ValueError:
                print('inapproriate dataset, please put type or stanford')

            print('\nTraining starts')
            logger = getlogger(args.log_dir + '/DATASET_{}_LOW_{}'.format(args.dataset, str(args.low_ratio)))
            for arg in vars(args):
                logger.info('{} - {}'.format(str(arg), str(getattr(args, arg))))
            logger.info('\nTraining Low Resolution images, Low resolution of {}x{}'.format(str(args.low_ratio), str(args.low_ratio)))
            logger.info('\t on '+args.dataset.upper()+' dataset, with hyper parameters above\n\n')
            training(net, optimizer,
                     args.lr, args.lr_decay, args.epochs, args.ten_batch_eval,
                     train_loader, eval_train_loader, eval_validation_loader, num_training, num_validation,
                     args.low_ratio, args.result,
                     logger,
                     args.vgg_gap,
                     args.save
                     )

