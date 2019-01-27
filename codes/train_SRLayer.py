import torch
import alexnet
import argparse
import hdf5dataset
import torch.nn as nn
import torch.optim as optim
import math
import os
import cv2
import numpy as np
import os.path as osp
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataloader import ImageNetTestDataset
from dataloader import ImageNetDataset


def get_aruments():
    parser = argparse.ArgumentParser(description="Pytorch RACNN")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    parser.add_argument("--pretrained", default='', type=str, help="path to pretrained model (default:none)")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum, Default: 0.9")
    parser.add_argument("--base-lr", type=float, default=0.000000002, help="base learning rate")
    parser.add_argument("--weight-decay", type=float, default=0, help="weight decay")
    parser.add_argument("--gamma", type=float, default=0, help="gamma (default:0.5)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train for")
    parser.add_argument("--decay-period", type=int, default=20, help="Decay period to adjust lr(default:20)")
    parser.add_argument("--data-dir", type=str, default='./../data/ILSVRC2013_DET_val/', help="ILSVRC2013 dir")
    parser.add_argument("--data-test-dir", type=str, default='./../data/Test/', help="ILSVRC2013 test dir")
    parser.add_argument("--is-normalized", action="store_true", help="subtract mean?")
    parser.add_argument("--save-model", action="store_true")

    return parser.parse_args()


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


def adjust_learning_rate(optimizer, epoch, args):
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        lr = lr * (args.gamma ** (epoch // args.decay_period))
        param_group['lr'] = lr


def main():
    args = get_aruments()

    if args.is_normalized:
        version = str(args.batch_size) + '_norm_' + str(args.base_lr)
    else:
        version = str(args.batch_size) + '_' + str(args.base_lr)

    print version

    print("===> Loading datasets")
    train_set = hdf5dataset.DatasetFromHdf5("data/train.h5")
    training_data_loader = DataLoader(dataset=train_set, batch_size=128, shuffle=True)
    if args.is_normalized:
        train_loader = DataLoader(ImageNetDataset(args.data_dir), batch_size = args.batch_size, shuffle=True)
        test_loader = DataLoader(ImageNetTestDataset(args.data_test_dir), batch_size=1, shuffle=True)
    else:
        train_loader = DataLoader(ImageNetDataset(args.data_dir, mean=(0,0,0)), batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(ImageNetTestDataset(args.data_test_dir, mean=(0,0,0)), batch_size=1, shuffle=True)

    model = alexnet.SRLayer()
    # mse_loss = nn.MSELoss(size_average=False)
    mse_loss = nn.MSELoss()

    model = model.cuda()
    mse_loss = mse_loss.cuda()

    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading model '{}'".format(args.pretrained))
            weights = torch.load(args.pretrained)
            model.load_state_dict(weights['model'].state_dict())
        else:
            print("=> no model found at '{}'".format(args.pretrained))

    print("===> Setting Optimizer")
    optimizer = optim.SGD(
        [{'params': model.sconv1.weight, 'lr': 1.0 * args.base_lr},
         {'params': model.sconv1.bias, 'lr': 0.1 * args.base_lr},
         {'params': model.sconv2.weight, 'lr': 1.0 * args.base_lr},
         {'params': model.sconv2.bias, 'lr': 0.1 * args.base_lr},
         {'params': model.sconv3.weight, 'lr': 0.01 * args.base_lr},
         {'params': model.sconv3.bias, 'lr': 0.01 * args.base_lr}],
        momentum=args.momentum, weight_decay=args.weight_decay)
    print optimizer

    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        model.train()

        loss_value = 0

        """
        # load training image patch from h5df file
        for iteration, batch in enumerate(training_data_loader, 1):
            input, target = Variable(batch[0]), Variable(batch[1], requires_grad=False)
            # print input.shape
            # (128, 3, 51, 51)
            input = input.cuda()
            target = target.cuda()

            output = model(input)
            loss = mse_loss(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if iteration % 10 == 0:
            print("===> Epoch[{}]({}/{}): Loss: {:.10f}".format(epoch, iteration, len(training_data_loader),
                                                                loss.data[0]))
            loss_value += loss.data.cpu().numpy()

        print("===> Epoch[{}]: Loss: {:10f}".format(epoch, loss_value / len(training_data_loader)))
        """

        for i_iter in range(1000):
        # for i_iter, batch in enumerate(train_loader):
        #     images, low_images, _ = batch
            images, low_images, _ = next(iter(train_loader))
            bs, crops, c, h, w = images.shape

            images = images.view(-1, c, h, w)
            low_images = low_images.view(-1, c, h, w)

            # print images.shape
            # (?, 3, 51, 51)

            images = Variable(images, requires_grad=False).cuda()
            low_images = Variable(low_images).cuda()

            output = model(low_images)

            loss = mse_loss(output, images) / 2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i_iter % 10 == 0:
                print("===> Epoch[{}]({}/{}): Loss: {:.10f}".format(epoch, i_iter, len(train_loader),
                                                                    loss.data.cpu()))
                loss_value += loss.data.cpu().numpy()

        print("===> Epoch[{}]: Loss: {:10f}".format(epoch, loss_value / len(train_loader)))

        model.eval()

        avg_loss_value = 0
        avg_psnr = 0

        for j_iter, batch in enumerate(test_loader):
            images, low_images, _ = batch

            im_ori = np.array(images[0]).transpose((1, 2, 0))
            im_low = np.array(low_images[0]).transpose((1, 2, 0))

            images = Variable(images, requires_grad=False).cuda()
            low_images = Variable(low_images, requires_grad=False).cuda()

            output = model(low_images)
            loss = mse_loss(images, output) / 2

            im_sr = output[0].data.cpu().numpy().transpose((1, 2, 0))

            psnr_improvement = calculate_PSNR(im_ori, im_low, im_sr)

            avg_loss_value += loss.data.cpu().numpy()
            avg_psnr += psnr_improvement

        avg_loss_value /= len(test_loader)
        avg_psnr /= len(test_loader)

        print 'Test set: avg_loss_value = ', avg_loss_value, 'avg_psnr = ', avg_psnr

        if args.save_model and epoch % 10 == 0 and epoch != 0:
            print "Save model (epoch:", epoch, ")"
            torch.save(model.state_dict(), osp.join('./models/', 'sr_' + version + '_' +  str(epoch) + '.pth'))


if __name__ == "__main__":
    main()