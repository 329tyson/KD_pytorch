import torch
import alexnet
import argparse
import torch.nn as nn
import torch.optim as optim
import math
import os
import cv2
import numpy as np
import os.path as osp
import torchvision.utils as vutils
import datetime
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from preprocess import *


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_aruments():
    parser = argparse.ArgumentParser(description="Pytorch RACNN")
    parser.add_argument("--dataset", type=str, default='cub')
    parser.add_argument("--batch", default='128', type=int)
    parser.add_argument("--annotation_train", type=str)
    parser.add_argument("--annotation_val", type=str)
    parser.add_argument("--data", type=str)
    parser.add_argument("--low_ratio", type=int, default=25)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("ten_batch_eval", action='store_true')
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--verbose", action = 'store_true')
    parser.add_argument("--pretrain_path", type=str)
    parser.add_argument("--gpu", type=str, default = '0')
    parser.add_argument("--feature", type=int, default=0)
    parser.add_argument("--decay", type=int, default=20)

    parser.set_defaults(ten_batch_eval=True)
    parser.set_defaults(verbose=True)

    return parser.parse_args()


def adjust_learning_rate(optimizer, epoch, args):
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        lr = lr * (0.5 ** (epoch // args.decay))
        param_group['lr'] = lr


def main():
    args = get_aruments()

    if args.feature :
        name = 'conv' + str(args.feature)
        model_name = 'SR_Pretrain_perceptual:' + str(args.feature) + '_lr:' + str(args.lr) + '_decay:' + str(args.decay) 

    # writer = SummaryWriter('_'.join(('runs/',datetime.datetime.now().strftime('%Y-%m-%d'), 'SR_Pretrain')))
    writer = SummaryWriter('_'.join(('runs/' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'), model_name)))
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

    print("===> Loading datasets")
    train_loader, eval_train_loader, eval_validation_loader, num_training, num_validation = generate_dataset(
        args.dataset,
        args.batch,
        args.annotation_train,
        args.annotation_val,
        args.data,
        args.low_ratio,
        args.ten_batch_eval,
        args.verbose,
        True)

    model = alexnet.SRLayer()
    mse_loss = nn.MSELoss()

    model = model.cuda()
    mse_loss = mse_loss.cuda()


    print("===> Setting Optimizer")
    optimizer = optim.SGD(
        [{'params': model.sconv1.weight, 'lr': 1.0 * args.lr},
         {'params': model.sconv1.bias, 'lr': 0.1 * args.lr},
         {'params': model.sconv2.weight, 'lr': 1.0 * args.lr},
         {'params': model.sconv2.bias, 'lr': 0.1 * args.lr},
         {'params': model.sconv3.weight, 'lr': 0.01 * args.lr},
         {'params': model.sconv3.bias, 'lr': 0.01 * args.lr}],
        momentum=0.9, weight_decay=0)

    alex = alexnet.AlexNet(0.5, 200, ['fc8'])
    load_weight(alex, args.pretrain_path)
    alex.cuda()
    alex.eval()
    model.train()

    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch, args)


        loss_value = 0
        show = 3
        count =0
        ldr= []
        sr = []
        hdr= []
        for x, x_low, _ in train_loader:
            x_low = x_low.cuda().float()
            x = x.cuda().float()
            sr_image = model(x_low)
            optimizer.zero_grad()

            t_output, t_features = alex(x)

            s_output, sr_features = alex(sr_image)

            l2loss = mse_loss(sr_image, x)
            # ploss = nn.KLDivLoss()(F.log_softmax(s_output, dim=1),
                                     # F.softmax(t_output, dim=1))    # teacher's hook is called in every loss.backward()
            # ploss = mse_loss(s_output, t_output)
            ploss = mse_loss(sr_features[name], t_features[name].detach())

            loss = ploss
            loss.backward()
            optimizer.step()
            if count < show :
                count += 1
                sr.append(sr_image[0])
                ldr.append(x_low[0])
                hdr.append(x[0])

            loss_value += loss.data.cpu()

        loss_value /= len(train_loader)

        torch.stack(sr, dim=0)
        torch.stack(ldr, dim=0)
        torch.stack(hdr, dim=0)

        sr = vutils.make_grid(sr, normalize=True, scale_each=True)
        ldr = vutils.make_grid(ldr, normalize=True, scale_each=True)
        hdr = vutils.make_grid(hdr, normalize=True, scale_each=True)

        writer.add_image('LDR', ldr, epoch + 1)
        writer.add_image('SR', sr, epoch + 1)
        writer.add_image('HDR', hdr, epoch + 1)
        writer.add_scalar('Perceptual Loss', loss_value, epoch + 1)

        print ("===> Epoch[{}/{}]: MSELOSS: {:3} PERCEPTUALLOSS: {:3}".format(epoch, args.epochs, ploss.item(), ploss.item()))

        if epoch % 10 == 0 and epoch != 0:
            print ("Save model (epoch:", epoch, ")")
            # torch.save(model.state_dict(), osp.join('./models/', 'sr_' + 'output_mse' +  str(epoch) + '.pth'))
            torch.save(model.state_dict(), osp.join('./models/', model_name + '_' +  str(epoch) + 'epoh.pth'))


if __name__ == "__main__":
    main()
