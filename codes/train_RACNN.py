import torch
import alexnet
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from tensorboardX import SummaryWriter
# from dataloader import CUBDataset
from preprocess import generate_dataset
from argparser import parse

# from dataloader import ImageNetDataset
# from dataloader import ImageNetTestDataset
from torchvision import transforms
from torch.utils import data
import torch.backends.cudnn as cudnn
import math

import os.path as osp
import cv2

BATCH_SIZE = 8
NUM_WORKERS = 1
ITER_SIZE = 1
DATA_DIRECTORY = './../data/ILSVRC2013_DET_val/'
TEST_DIRECTORY = './../data/Test/'
LEARNING_RATE = 1e-6
GAMMA = 0.1
WEIGHT_DECAY = 0.0005
NUM_EPOCH = 51
GAMMA = 0.1
MOMENTUM = 0.95
DECAY_PERIOD = 50

CUB_CSV_FILE = '../labels/label_train_cub200_2011.csv'
CUB_IMG_DIR = '../CUB_200_2011/images/'

BASE_LR = 1e-6

PRETRAIN_SR = False

# From RSRCNN caffer version
# gamma:0.5
# base_lr: 2e-9
# momentum: 0.9
# weight_decay: 0

# From VDSR
# gamma:0.1
# lr:0.1
# step:10
# momentum:0.9
# weight_decay: 1e-4


# def get_arguments():
#     parser = argparse.ArgumentParser(description="pretrain SR Layer")
#     parser.add_argument("--dataset", type=str, default="CUB")
#     # parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY)
#     parser.add_argument("--data", type=str, default="./data")
#     parser.add_argument("--annotation_train", type=str, default="./annotataions_train")
#     parser.add_argument("--annotation_val", type=str, default="./annotataions_val")
#     parser.add_argument("--batch", type=int, default=BATCH_SIZE)
#     parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
#     parser.add_argument("--gamma", type=float, default=GAMMA)
#     parser.add_argument("--momentum", type=float, default=MOMENTUM)
#     parser.add_argument("--decay-period", type=int, default=DECAY_PERIOD)
#     parser.add_argument("--clip", type=float, default=CLIP)
#     parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
#     parser.add_argument("--is-normalized", action="store_true", help="subtract mean?")
#     parser.add_argument("--pretrain", action="store_true")
#     parser.add_argument("--low_ratio", type=int, default=25)
#
#     return parser.parse_args()
#
#
# args = get_arguments()
args = parse()


# Here is the function for PSNR calculation
def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)


def calculate_PSNR(img, img_low, img_sr):
    im_sr_refined = img_sr
    im_sr_refined[im_sr_refined < 0] = 0
    im_sr_refined[im_sr_refined > 255.] = 255.

    im_ori_ycbcr = np.array(cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB))
    im_low_ycbcr = np.array(cv2.cvtColor(img_low, cv2.COLOR_RGB2YCR_CB))
    im_sr_ycbcr = np.array(cv2.cvtColor(im_sr_refined, cv2.COLOR_RGB2YCR_CB))

    im_ori_y = im_ori_ycbcr[:, :, 0].astype(float)
    im_low_y = im_low_ycbcr[:, :, 0].astype(float)
    im_sr_y  = im_sr_ycbcr[:, :, 0].astype(float)

    psnr_bicubic = PSNR(im_ori_y, im_low_y)
    psnr_super = PSNR(im_ori_y, im_sr_y)

    return psnr_super - psnr_bicubic


def adjust_learning_rate(optimizer, epoch):
    # lr = LEARNING_RATE * (args.gamma ** (epoch // args.decay_period))

    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        lr = lr * (GAMMA ** (epoch // DECAY_PERIOD))
        param_group['lr'] = lr

writer = SummaryWriter()

# net = alexnet.RACNN(0.5, 200, ['fc8'], 'bvlc_alexnet.npy', True, './models/sr_50_0.4_0.1_0.0_0.0.pth')
net = alexnet.RACNN(0.5, 200, ['fc8'], 'bvlc_alexnet.npy', True)

train_loader, eval_train_loader, eval_validation_loader, num_training, num_validation = generate_dataset(
                    args.dataset,
                    args.batch,
                    args.annotation_train,
                    args.annotation_val,
                    args.data,
                    args.low_ratio,
                    args.ten_batch_eval,
                    args.verbose,
                    True)   # To get image & low image, set args.kd_enabled = True
                    # args.kd_enabled)

net.cuda()

MSE_loss = nn.MSELoss()
CE_loss = nn.CrossEntropyLoss()

# optimizer = optim.SGD([{'params': net.get_all_params_except_last_fc(), 'lr': 0.1, 'weight_decay': 0},
#                        {'params': net.classificationLayer.fc8.parameters(), 'lr':1.0, 'weight_decay': 1.0}],
#                       momentum=args.`, weight_decay=args.weight_decay)
"""
optimizer = optim.SGD([{'params':net.srLayer.parameters(), 'lr': 0.1 * args.base_lr},
                       {'params':net.get_all_params_except_last_fc(), 'lr': 0.1 * args.base_lr},
                       {'params':net.classificationLayer.fc8.weight, 'lr': 1.0 * args.base_lr,
                        'weight_decay': 1.0 * args.weight_decay},
                       {'params':net.classificationLayer.fc8.bias, 'lr': 2.0 * args.base_lr,
                        'weight_decay': 0.0}],
                       momentume=args.momentum, weight_decay=args.weight_decay)
"""

#FIXME: should weight deacy be multiplied?
# optimizer = optim.SGD([{'params':net.srLayer.sconv1.parameters(), 'lr': 0.1 * BASE_LR, 'weight_decay': 0.1 * WEIGHT_DECAY},
#                        {'params':net.srLayer.sconv2.parameters(), 'lr': 0.1 * BASE_LR, 'weight_deacy': 0.1 * WEIGHT_DECAY},
#                        {'params':net.srLayer.sconv3.parameters(), 'lr': 0.1 * BASE_LR, 'weight_deacy': 0.1 * WEIGHT_DECAY},
#                        {'params':net.get_all_params_except_last_fc(), 'lr': 0.1 * BASE_LR},
#                        {'params':net.classificationLayer.fc8.parameters(), 'lr': 1.0 * BASE_LR,
#                         'weight_decay': 1.0 * WEIGHT_DECAY}],
#                        momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

# from g_RACNN_alexnet.prototxt
optimizer = optim.SGD([{'params': net.srLayer.sconv1.weight, 'lr': 1 * BASE_LR},
                       {'params': net.srLayer.sconv1.bias, 'lr': 0.1 * BASE_LR},
                       {'params': net.srLayer.sconv2.weight, 'lr': 1 * BASE_LR},
                       {'params': net.srLayer.sconv2.bias, 'lr': 0.1 * BASE_LR},
                       {'params': net.srLayer.sconv3.weight, 'lr': 0.1 * BASE_LR},
                       {'params': net.srLayer.sconv3.bias, 'lr': 0.1 * BASE_LR},
                       {'params': net.get_all_params_except_last_fc(), 'lr': 0, 'weight_deacy': 0},
                       {'params': net.classificationLayer.fc8.weight, 'lr': 1 * BASE_LR, 'weight_decay': WEIGHT_DECAY},
                       {'params': net.classificationLayer.fc8.bias, 'lr': 2 * BASE_LR, 'weight_decay': 0}],
                      momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)


version = str(args.batch) + '_lr' + str(BASE_LR)

if PRETRAIN_SR:
    for epoch in range(30):
        adjust_learning_rate(optimizer, epoch, args)

        for x, x_low, y in train_loader:
            x = x.cuda().float()
            x_low = x_low.cuda().float()
            y = y.cuda() - 1

            optimizer.zero_grad()

            output, sr_images = net(x_low)

            sr_loss = MSE_loss(x, sr_images) / 2

            sr_loss.backward()
            optimizer.step()

        im_ori = np.array(x[0]).transpose((1, 2, 0))
        im_low = np.array(x_low[0]).transpose((1, 2, 0))

        im_sr = output[0].data.cpu().numpy().transpose((1, 2, 0))

        psnr_improvement = calculate_PSNR(im_ori, im_low, im_sr)

        print 'PSNR_improvement of a test img = ', psnr_improvement

        if args.save_model and (epoch+1) % 10 == 0:
            print "Save SRLayer model (epoch: ", epoch, ")"
            torch.save(net.SRLayer.state_dict(), osp.join('./models/', 'sr_' + version + '_epoch' + str(epoch) + '.pth'))

        writer.add_scalar('PSNR_improve', psnr_improvement, epoch)


# optimizer = optim.SGD([{'params':net.srLayer.sconv1.parameters(), 'lr': 0.1 * BASE_LR, 'weight_decay': 0.1 * WEIGHT_DECAY},
#                        {'params':net.srLayer.sconv2.parameters(), 'lr': 0.1 * BASE_LR, 'weight_deacy': 0.1 * WEIGHT_DECAY},
#                        {'params':net.srLayer.sconv3.parameters(), 'lr': 0.1 * BASE_LR, 'weight_deacy': 0.1 * WEIGHT_DECAY},
#                        {'params':net.get_all_params_except_last_fc(), 'lr': 0.1 * BASE_LR},
#                        {'params':net.classificationLayer.fc8.parameters(), 'lr': 1.0 * BASE_LR,
#                         'weight_decay': 1.0 * WEIGHT_DECAY}],
#                        momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

for epoch in range(100):
    adjust_learning_rate(optimizer, epoch)
    net.train()

    for x, x_low, y in train_loader:
        x = x.cuda().float()
        x_low = x_low.cuda().float()
        y = y.cuda() - 1

        optimizer.zero_grad()

        output, sr_images = net(x_low)

        loss = CE_loss(output, y)

        print loss.data.cpu()

        loss.backward()
        optimizer.step()

        loss_value = loss.data.cpu()

    net.eval()

    if (epoch + 1) % 10 == 0:
        writer.add_scalar('loss', loss, epoch)
        print('Epoch : {}, training loss : {}'.format(epoch + 1, loss_value))
        if args.save:
            torch.save(net.state_dict(), './models/RACNN_' + str(epoch) + '_epoch.pt')

        # Test only 10, 20, 30... epochs
        hit_training = 0
        hit_validation = 0

        for x_low, y in eval_train_loader:
            # To CUDA tensors
            x_low = torch.squeeze(x_low)
            x_low = x_low.cuda().float()
            y -= 1

            # Network output
            output, _ = net(x_low)

            if args.ten_batch_eval:
                prediction = torch.mean(output, dim=0)
                prediction = prediction.cpu().detach().numpy()

                if np.argmax(prediction) == y:
                    hit_training += 1
            else:
                _, prediction = torch.max(output, 1)
                prediction = prediction.cpu().detach().numpy()
                hit_training += (prediction == y.numpy()).sum()

        for x_low, y in eval_validation_loader:
            # To CUDA tensors
            x_low = torch.squeeze(x_low)
            x_low = x_low.cuda().float()
            y -= 1

            # Network output
            output, _ = net(x_low)

            if args.ten_batch_eval:
                prediction = torch.mean(output, dim=0)
                prediction = prediction.cpu().detach().numpy()

                if np.argmax(prediction) == y:
                    hit_validation += 1

            else:
                _, prediction = torch.max(output,1)
                prediction = prediction.cpu().detach().numpy()
                hit_validation += (prediction == y.numpy()).sum()

        # Trace
        acc_training = float(hit_training) / num_training
        acc_validation = float(hit_validation) / num_validation
        print('Epoch : {}, training loss : {}'.format(epoch + 1, loss))
        print('    Training   set accuracy : {0:.2f}%, for {1:}/{2:}'
              .format(acc_training*100, hit_training, num_training))
        print('    Validation set accuracy : {0:.2f}%, for {1:}/{2:}\n'
              .format(acc_validation*100, hit_validation, num_validation))

        # Log Tensorboard
        writer.add_scalar('loss', loss, epoch)
        writer.add_scalars('Accuracies',
                           {'Training accuracy': acc_training,
                            'Validation accuracy': acc_validation}, epoch)

print('Finished Training')
writer.close()
