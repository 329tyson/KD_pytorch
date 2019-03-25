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
from train import write_gradcam
glb_grad_at = {}


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
    parser.add_argument("--message", type=str, default='SR_Pretrain')

    parser.set_defaults(ten_batch_eval=True)
    parser.set_defaults(verbose=True)

    return parser.parse_args()


def adjust_learning_rate(optimizer, epoch, args):
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        lr = lr * (0.5 ** (epoch // args.decay))
        param_group['lr'] = lr

def save_grad(module, grad_in, grad_out):
    global glb_grad_at

    glb_grad_at[id(module)] = grad_out[0].detach()

def compute_gradCAM(feature, grad):
    w = F.adaptive_avg_pool2d(grad, 1)

    gcam = torch.mul(w, feature)
    gcam = torch.sum(gcam, dim = 1)

    gcam = torch.clamp(gcam, min=0.0)
    for gc in gcam:
        gc -= gc.min()
        gc /= gc.max()
    # gcam[gcam < 0.5] = 0.
    return gcam

def main():
    args = get_aruments()

    if args.feature :
        name = 'conv' + str(args.feature)
        model_name = 'SR_Pretrain_perceptual:' + str(args.feature) + '_lr:' + str(args.lr) + '_decay:' + str(args.decay)

    writer = SummaryWriter('_'.join(('runs/',datetime.datetime.now().strftime('%Y-%m-%d'), args.message)))
    # writer = SummaryWriter('_'.join(('runs/' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'), model_name)))
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

    net = alexnet.AlexNet(0.5, 200, ['fc8'])
    load_weight(net, args.pretrain_path)
    net.cuda()
    net.conv5.register_backward_hook(save_grad)
    net.eval()
    model.train()

    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch, args)
        loss_value = 0
        ldr= []
        sr = []
        hdr= []
        show = False
        for x, x_low, y, path in train_loader:
            x_low = x_low.cuda().float()
            x = x.cuda().float()
            y = y.cuda() - 1
            model.zero_grad()
            net.zero_grad()
            sr_image = model(x_low)
            optimizer.zero_grad()

            output, features = net(x)
            one_hot_y = torch.zeros(output.shape).float().cuda()
            for i in range(output.shape[0]):
                one_hot_y[i][y[i]] = 1.0

            output.backward(gradient = one_hot_y, retain_graph = True)

            s_output, sr_features = net(sr_image)
            # gradient-oriented SR
            gcams = compute_gradCAM(features['conv5'].detach(), glb_grad_at[id(net.conv5)])
            if show is False:
                for i in range(3):
                    hdr.append(x[i])
                    ldr.append(x_low[i])
                    sr.append(sr_image[i])
                gcam = gcams[:3]
                write_gradcam(gcam, sr, writer, epoch, mode ='sr')
                torch.stack(hdr, dim=0)
                torch.stack(ldr, dim=0)
                torch.stack(sr, dim=0)
                hdr = vutils.make_grid(hdr, normalize=True, scale_each=True)
                ldr = vutils.make_grid(ldr, normalize=True, scale_each=True)
                sr = vutils.make_grid(sr, normalize=True, scale_each=True)

                writer.add_image('HDR', hdr, epoch + 1)
                writer.add_image('LDR', ldr, epoch + 1)
                writer.add_image('SR', sr, epoch + 1)
                show =True
            loss = torch.sub(features['conv5'], sr_features['conv5'])
            loss = torch.mul(loss, loss)
            loss = torch.mul(loss, torch.unsqueeze(gcams, dim=1))
            loss = torch.mean(loss)
            loss += mse_loss(sr_image, x) * 0.5

            # loss = torch.mul(mse_loss(sr_image, x), torch.unsqueeze(res_gcams, dim=1))
            loss.backward()
            optimizer.step()

            loss_value += loss.data.cpu()

        loss_value /= len(train_loader)


        writer.add_scalar('Loss', loss_value, epoch + 1)

        print "===> Epoch[{}/{}]: MSELOSS: {:3} ".format(epoch, args.epochs, loss.item())

        if epoch % 10 == 0 and epoch != 0:
            print ("Save model (epoch:", epoch, ")")
            torch.save(model.state_dict(), osp.join('./models/', 'sr_' + 'output_mse' +  str(epoch) + '.pth'))
            # torch.save(model.state_dict(), osp.join('./models/', model_name + '_' +  str(epoch) + 'epoh.pth'))


if __name__ == "__main__":
    main()
