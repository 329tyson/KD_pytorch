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
from tensorboardX import SummaryWriter
from preprocess import *



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
    parser.add_argument("--lr", type=float)
    parser.add_argument("--verbose", action = 'store_true')

    parser.set_defaults(ten_batch_eval=True)
    parser.set_defaults(verbose=True)

    return parser.parse_args()


def adjust_learning_rate(optimizer, epoch, args):
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        lr = lr * (0.5 ** (epoch // 20))
        param_group['lr'] = lr


def main():
    args = get_aruments()
    writer = SummaryWriter()

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

    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        model.train()

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
            loss = mse_loss(x, sr_image)
            if count < show :
                count += 1
                sr.append(sr_image[0])
                ldr.append(x_low[0])
                hdr.append(x[0])

        torch.stack(sr, dim=0)
        torch.stack(ldr, dim=0)
        torch.stack(hdr, dim=0)

        sr = vutils.make_grid(sr, normalize=True, scale_each=True)
        ldr = vutils.make_grid(ldr, normalize=True, scale_each=True)
        hdr = vutils.make_grid(hdr, normalize=True, scale_each=True)

        writer.add_image('LDR', ldr, epoch + 1)
        writer.add_image('SR', sr, epoch + 1)
        writer.add_image('HDR', hdr, epoch + 1)

        print "===> Epoch[{}/{}]: Loss: {:3}".format(epoch, args.epochs, loss.item())

        model.eval()

        if args.save_model and epoch % 10 == 0 and epoch != 0:
            print "Save model (epoch:", epoch, ")"
            torch.save(model.state_dict(), osp.join('./models/', 'sr_' + version + '_' +  str(epoch) + '.pth'))


if __name__ == "__main__":
    main()
