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
from tqdm import tqdm
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
    parser.add_argument("--gradcam", default=False, action = 'store_true')
    parser.add_argument("--l2", default=False, action= 'store_true')
    parser.add_argument("--image_norm", default=False, action= 'store_true')
    parser.add_argument("--regression_features",default=None )
    parser.add_argument("--weight", type=float, default=1.)

    parser.set_defaults(ten_batch_eval=True)
    parser.set_defaults(verbose=True)
    parser.set_defaults(gradcam=False)
    parser.set_defaults(l2 = False)
    parser.set_defaults(image_norm = False)

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

    writer = SummaryWriter('_'.join(('SR/',datetime.datetime.now().strftime('%Y-%m-%d'), args.message)))
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
        is_KD = True,
        image_norm = args.image_norm)

    gen = alexnet.SRLayer()
    dis = alexnet.Discriminator()
    mse_loss = nn.MSELoss()

    gen = gen.cuda()
    dis = dis.cuda()
    mse_loss = mse_loss.cuda()


    print("===> Setting Optimizer")
    # g_optimizer = optim.Adam(
        # [{'params': gen.sconv1.weight, 'lr': 1.0 * args.lr},
         # {'params': gen.sconv1.bias, 'lr': 0.1 * args.lr},
         # {'params': gen.sconv2.weight, 'lr': 1.0 * args.lr},
         # {'params': gen.sconv2.bias, 'lr': 0.1 * args.lr},
         # {'params': gen.sconv3.weight, 'lr': 0.01 * args.lr},
         # {'params': gen.sconv3.bias, 'lr': 0.001 * args.lr}])
    d_optimizer = optim.Adam(
        [{'params': dis.parameters(), 'lr':0.1 * args.lr}])
    g_optimizer = optim.Adam(
        [{'params': gen.parameters(), 'lr':0.1 * args.lr}])

    teacher = alexnet.AlexNet(0.5, 200, ['fc8'])
    load_weight(teacher, args.pretrain_path)
    teacher.cuda()

    # teacher.conv5.register_backward_hook(save_grad)
    teacher.eval()
    gen.train()
    dis.train()

    GLoss = AverageMeter()
    DLoss = AverageMeter()
    MSE = AverageMeter()
    PLoss = AverageMeter()

    print 'Feature regression for convs : {}'.format(args.regression_features)
    print 'regression convs length {}'.format(len(args.regression_features.split()))
    for epoch in range(args.epochs):
        adjust_learning_rate(g_optimizer, epoch, args)
        adjust_learning_rate(d_optimizer, epoch, args)
        GLoss.reset()
        DLoss.reset()
        MSE.reset()
        PLoss.reset()
        loss_value = 0
        G_loss_value = 0
        D_loss_value = 0
        d_loss = 0
        ldr= []
        sr = []
        hdr= []
        guide = []
        show = False
        train_iter = tqdm(enumerate(train_loader))
        for i, (x, x_low, y, path) in train_iter:
            x_low = x_low.cuda().float()
            x = x.cuda().float()
            y = y.cuda() - 1

            gen.zero_grad()
            teacher.zero_grad()
            dis.zero_grad()
            g_optimizer.zero_grad()
            d_optimizer.zero_grad()


            ###################################################
            # Update Generator                                #
            ###################################################
            for p in dis.parameters():
                p.requires_grad = False
            for p in gen.parameters():
                p.requires_grad = True

            sr_image, additive = gen(x_low)

            output, features = teacher(x)
            s_output, sr_features = teacher(sr_image)

            val_false = dis(sr_image)
            ploss = 0.
            for conv in args.regression_features.split():
                tconv = features['conv' + str(conv)].detach()
                sconv = sr_features['conv' + str(conv)].detach()
                # mseconv = torch.sub(tconv, sconv)
                # mseconv = torch.mul(mseconv, mseconv)
                # mseconv = torch.mean(torch.mul(mseconv, tconv))
                # ploss += torch.div(mseconv, len(args.regression_features.split()))
                ploss += torch.div(mse_loss(tconv, sconv), len(args.regression_features.split()))

            GAN_weight = args.weight
            d_loss_g = -torch.mean(torch.log(val_false)) * GAN_weight
            # d_loss_g = -torch.mean(torch.log(val_false)) * 0.001
            genloss = d_loss_g + ploss
            if args.l2:
                mseloss = mse_loss(sr_image, x)
                genloss += mseloss
            genloss.backward()

            ###################################################
            # Update Discriminator                            #
            ###################################################
            for p in dis.parameters():
                p.requires_grad = True

            for p in gen.parameters():
                p.requires_grad = False

            sr_image = sr_image.detach()
            val_true = dis(x)
            val_false = dis(sr_image)
            ones = torch.ones(4,1,1,1).cuda()
            d_loss_d = -torch.mean(torch.log(val_true)) * GAN_weight - torch.mean(torch.log(torch.sub(ones ,val_false))) * GAN_weight
            d_loss_d.backward()
            if show is False:
                for i in range(3):
                    hdr.append(x[i])
                    ldr.append(x_low[i])
                    sr.append(sr_image[i])
                    guide.append(additive[i])
                if args.gradcam:
                    gcam = gcams[:3]
                    write_gradcam(gcam, x, writer, epoch, mode ='sr')
                torch.stack(hdr, dim=0)
                torch.stack(ldr, dim=0)
                torch.stack(sr, dim=0)
                torch.stack(guide, dim=0)
                hdr = vutils.make_grid(hdr, normalize=True, scale_each=True)
                ldr = vutils.make_grid(ldr, normalize=True, scale_each=True)
                sr = vutils.make_grid(sr, normalize=True, scale_each=True)
                guide = vutils.make_grid(guide, normalize=True, scale_each=True)

                writer.add_image('HDR', hdr, epoch + 1)
                writer.add_image('LDR', ldr, epoch + 1)
                writer.add_image('SR', sr, epoch + 1)
                writer.add_image('GUIDANCE', guide, epoch + 1)
                show =True
            # Perceptual loss
            # grad_loss = torch.sub(features['conv5'], sr_features['conv5'])
            # grad_loss = torch.mul(grad_loss, grad_loss)
            # grad_loss = torch.mul(grad_loss, torch.unsqueeze(gcams, dim=1))
            # grad_loss = torch.mean(grad_loss)
            # loss = d_loss + grad_loss
            d_loss = d_loss_d + d_loss_g
            # loss += mseloss
            # print "===> Epoch[{}/{}]: GradCAMloss: {:3} G_loss: {:3} D_loss: {:3}".format(epoch, args.epochs, grad_loss.item(), d_loss_g.item(), d_loss_d.item())

            loss  = d_loss + ploss
            if loss == float('inf') or loss != loss:
                exit(1)
            g_optimizer.step()
            if d_loss_d.item() > 0.0001:
                d_optimizer.step()

            loss_value += loss.data.cpu()
            PLoss.update(ploss.item(), x.size(0))
            # MSE.update(mseloss.item(), x.size(0))
            GLoss.update(d_loss_g.item(), x.size(0))
            DLoss.update(d_loss_d.item(), x.size(0))
            # train_iter.set_description("===> Epoch[{}/{}]: GradCAMloss: {:3} G_loss: {:3} D_loss: {:3}".format(epoch, args.epochs, grad_loss.item(), d_loss_g.item(), d_loss_d.item()))
            # train_iter.set_description("===> Epoch[{}/{}]: G_loss: {:3} D_loss: {:3} MSEloss: {:3}".format(epoch, args.epochs,  d_loss_g.item(), d_loss_d.item(), mseloss.item()))
            if args.gradcam:
                train_iter.set_description("E[{}/{}]B[{}/{}]: G_loss: {:.5f} D_loss: {:.5f} GradCAM: {:.5f}".format(epoch, args.epochs,i,num_training//4,  d_loss_g.item(), d_loss_d.item(), ploss.item()))
            elif args.l2:
                train_iter.set_description("E[{}/{}]B[{}/{}]: G_loss: {:.5f} D_loss: {:.5f} Ploss: {:.5f} MSE: {:.5f}".format(epoch, args.epochs,i, num_training//4, d_loss_g.item(), d_loss_d.item(), ploss.item(), mseloss.item()))
            else:
                train_iter.set_description("E[{}/{}]B[{}/{}]: G_loss: {:.5f} D_loss: {:.5f} Ploss: {:.5f}".format(epoch, args.epochs,i, num_training//4,  d_loss_g.item(), d_loss_d.item(), ploss.item()))

        loss_value /= len(train_loader)


        writer.add_scalar('Loss', loss_value, epoch + 1)
        writer.add_scalar('GLoss', GLoss.avg, epoch + 1)
        writer.add_scalar('DLoss', DLoss.avg, epoch + 1)
        writer.add_scalar('PLoss', PLoss.avg, epoch + 1)

        # print "===> Epoch[{}/{}]: GradCAMloss: {:3} G_loss: {:3} D_loss: {:3}".format(epoch, args.epochs, grad_loss.item(), d_loss_g.item(), d_loss_d.item())
        # print "===> Epoch[{}/{}]: Average G_loss: {:3} Average D_loss: {:3} Average MSE : {:3}".format(epoch, args.epochs,  GLoss.avg, DLoss.avg, MSE.avg)
        print "Epoch[{}/{}]: Average G_loss: {:.5f} Average D_loss: {:.5f} Average Ploss : {:.5f}".format(epoch, args.epochs,  GLoss.avg, DLoss.avg, PLoss.avg)
        print '===========================================================\n'
        # print "===> Epoch[{}/{}]: GradCAMloss: {:3} DisLoss: {:3}".format(epoch, args.epochs, grad_loss.item(), d_loss.item())

        if epoch % 10 == 0 and epoch != 0:
            print ("Save gen (epoch:", epoch, ")")
            torch.save(gen.state_dict(), osp.join('./models/', 'SR_' + args.message + '_' +  str(epoch) + '.pth'))
            # torch.save(gen.state_dict(), osp.join('./models/', model_name + '_' +  str(epoch) + 'epoh.pth'))


if __name__ == "__main__":
    main()
