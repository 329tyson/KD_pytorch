import torch
import alexnet
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from dataloader import CUBDataset
from dataloader import ImageNetDataset
from dataloader import ImageNetTestDataset
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
DECAY_PERIOD = 20
CLIP = 0.4

CUB_CSV_FILE = '../labels/label_train_cub200_2011.csv'
CUB_IMG_DIR = '../CUB_200_2011/images/'

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


def get_arguments():
    parser = argparse.ArgumentParser(description="pretrain SR Layer")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--gamma", type=float, default=GAMMA)
    parser.add_argument("--momentum", type=float, default=MOMENTUM)
    parser.add_argument("--decay-period", type=int, default=DECAY_PERIOD)
    parser.add_argument("--clip", type=float, default=CLIP)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--is-normalized", action="store_true", help="subtract mean?")

    return parser.parse_args()


args = get_arguments()


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
        lr = lr * (args.gamma ** (epoch // args.decay_period))
        param_group['lr'] = lr


# Training Params
params = {'batch_size': 8,
          'shuffle': True,
          'num_workers': 6,
          'drop_last' : True}

eval_params = {'batch_size': 8,
               'shuffle': True,
               'num_workers': 6,
               'drop_last' : True}

writer = SummaryWriter()

net = alexnet.RACNN(0.5, 200, ['fc8'], True, 'bvlc_alexnet.npy', False, True,
                          './models/sr_50_0.4_0.1_0.0_0.0.pth', True)

if args.is_normalized:
    training_set = CUBDataset(CUB_CSV_FILE, CUB_IMG_DIR, True)
    training_generator = data.DataLoader(training_set, **params)

    eval_trainset = CUBDataset(CUB_CSV_FILE, CUB_IMG_DIR, True)
    eval_trainset_generator = data.DataLoader(eval_trainset, **eval_params)

    eval_validationset = CUBDataset(CUB_CSV_FILE, CUB_IMG_DIR, True)
    eval_validationset_generator = data.DataLoader(eval_validationset, **eval_params)
else:
    training_set = CUBDataset(CUB_CSV_FILE, CUB_IMG_DIR, True, mean=(.0,.0,.0))
    training_generator = data.DataLoader(training_set, **params)

    eval_trainset = CUBDataset(CUB_CSV_FILE, CUB_IMG_DIR, True, mean=(.0,.0,.0))
    eval_trainset_generator = data.DataLoader(eval_trainset, **eval_params)

    eval_validationset = CUBDataset(CUB_CSV_FILE, CUB_IMG_DIR, True, mean=(.0,.0,.0))
    eval_validationset_generator = data.DataLoader(eval_validationset, **eval_params)

num_training = len(training_set)
num_eval_trainset = len(eval_trainset)
num_eval_validationset = len(eval_validationset)

net.cuda()

MSE_loss = nn.MSELoss()
CE_loss = nn.CrossEntropyLoss()

# optimizer = optim.SGD([{'params': net.get_all_params_except_last_fc(), 'lr': 0.1, 'weight_decay': 0},
#                        {'params': net.classificationLayer.fc8.parameters(), 'lr':1.0, 'weight_decay': 1.0}],
#                       momentum=args.momentum, weight_decay=args.weight_decay)
optimizer = optim.SGD([{'params':net.srLayer.parameters(), 'lr': 0.1 * args.base_lr},
                       {'params':net.get_all_params_except_last_fc(), 'lr': 0.0},
                       {'params':net.classificationLayer.fc8.weight, 'lr': 1.0 * args.base_lr,
                        'weight_decay': 1.0 * args.weight_decay},
                       {'params':net.classificationLayer.fc8.bias, 'lr': 2.0 * args.base_lr,
                        'weight_decay': 0.0}],
                       momentume=args.momentum, weight_decay=args.weight_decay)

for epoch in range(100):
    adjust_learning_rate(optimizer, epoch)
    net.train()

    for i_iter, batch in enumerate(training_generator):
        images, low_images, y = batch

        images = Variable(images.float(), requires_grad=False).cuda()
        low_images = Variable(low_images.float(), requires_grad=True).cuda()
        print low_images
        y = y - 1
        y = Variable(y, requires_grad=False).cuda()

        optimizer.zero_grad()

        output, sr_images = net(low_images)

        loss1 = MSE_loss(images, sr_images) / 2
        loss2 = CE_loss(output, y)

        loss = loss1 + loss2
        loss.backward()
        # nn.utils.clip_grad_norm(net.parameters(), args.clip)
        optimizer.step()

        loss1_value = loss1.data.cpu()
        loss2_value = loss2.data.cpu()
        loss_value = loss.data.cpu()

        # print loss1_value
        # print loss2_value
    net.eval()

    if (epoch + 1) % 10 == 0:
        writer.add_scalar('loss', loss, epoch)
        print('Epoch : {}, training loss : {}, {}, {}'.format(epoch + 1, loss_value, loss1_value, loss2_value))
        torch.save(net.state_dict(), './models/RACNN_' + str(epoch) + '_epoch.pt')

    # Test only 10, 20, 30... epochs
    hit_training = 0
    hit_validation = 0

    for _, x, y in eval_trainset_generator:
        # To CUDA tensors
        # x = torch.squeeze(x)
        x = x.cuda().float()
        y -= 1

        # Network output
        output, _ = net(x)
        # prediction = torch.mean(output, dim=0)
        # prediction = prediction.cpu().detach().numpy()

        # if np.argmax(prediction) == y:
        #     hit_training += 1

        # Count prediction hit on training set
        prediction = torch.max(output, 1)[1]
        hit_training += np.sum(prediction.cpu().numpy() ==  y.numpy())

    for _, x, y in eval_validationset_generator:
        # To CUDA tensors
        # x = torch.squeeze(x)
        x = x.cuda().float()
        y -= 1

        # Network output
        output, _ = net(x)
        # prediction = torch.mean(output, dim=0)
        # prediction = prediction.cpu().detach().numpy()

        # if np.argmax(prediction) == y:
        #     hit_validation += 1

        # Count prediction hit on training set
        prediction = torch.max(output, 1)[1]
        hit_validation += np.sum(prediction.cpu().numpy() ==  y.numpy())

    # Trace
    acc_training = float(hit_training) / num_eval_trainset
    acc_validation = float(hit_validation) / num_eval_validationset
    print('Epoch : {}, training loss : {}'.format(epoch + 1, loss))
    print('    Training   set accuracy : {0:.2f}%, for {1:}/{2:}'
          .format(acc_training*100, hit_training, num_eval_trainset))
    print('    Validation set accuracy : {0:.2f}%, for {1:}/{2:}\n'
          .format(acc_validation*100, hit_validation, num_eval_validationset))

    # Log Tensorboard
    writer.add_scalar('loss', loss, epoch)
    writer.add_scalars('Accuracies',
                       {'Training accuracy': acc_training,
                        'Validation accuracy': acc_validation}, epoch)
    # torch.save(net.state_dict(), './models/teachernet_' + str(epoch) + '_epoch.pt')

print('Finished Training')
writer.close()
