from argparser import parse
from alexnet import AlexNet
from remodelling import training
from remodelling import training_KD
from preprocess import load_weight
from preprocess import generate_dataset

import os
import torch.optim as optim

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
    '''
    args = parse()
    args.annotation_train = os.path.join(args.root, args.annotation_train)
    args.annotation_val = os.path.join(args.root, args.annotation_val)
    args.data = os.path.join(args.root, args.data)
    args.result = os.path.join(args.root, args.result)

    if args.pretrain_path != 'NONE':
        args.pretrain_path = os.path.join(args.root, args.pretrain_path)

    if args.dataset.lower() == 'cub':
        args.classes = 200
    else:
        args.classes = 196


    net = AlexNet(0.5, args.classes, ['fc8'])

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

    if args.verbose is True:
        print('Training arguments settings')
        for arg in vars(args):
            print('\t',arg, getattr(args, arg))

    if args.kd_enabled is True:
        print('\nTraining Knowledge Distillation model')
        print('\t on ',args.dataset,' with hyper parameters above')
        if args.low_ratio == 0:
            print('Invalid argument, choose low resolution ( 50 | 25)')
        else :
            print('\tLow resolution scaling = {} x {}'.format(args.low_ratio, args.low_ratio))
            teacher_net = AlexNet(0.5, args.classes, ['fc8'])
            load_weight(net, args.pretrain_path)
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
                exit

            print('\nTraining starts')
            training_KD(
                teacher_net,
                net,
                optimizer,
                args.kd_temperature,
                args.lr,
                args.lr_decay,
                args.epochs,
                train_loader,
                eval_train_loader,
                eval_validation_loader,
                num_training,
                num_validation)

    else :
        if args.low_ratio == 0:
            print('\nTraining High Resolution images')
            print('\t on ',args.dataset.upper(),' dataset, with hyper parameters above')
            load_weight(net, args.pretrain_path)
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
            training(net, optimizer, args.lr, args.lr_decay, args.epochs, train_loader, eval_train_loader, eval_validation_loader, num_training, num_validation)
        else:
            print('\nTraining Low Resolution images')
            print('\t on ',args.dataset,' with hyper parameters above')
            print('\tLow resolution scaling = {} x {}'.format(args.low_ratio, args.low_ratio))
            load_weight(net, args.pretrain_path)
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
            training(net, optimizer, args.lr, args.lr_decay, args.epochs, train_loader, eval_train_loader, eval_validation_loader, num_training, num_validation)

