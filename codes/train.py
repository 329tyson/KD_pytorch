import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
import datetime
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.autograd import Variable

import cv2

from collections import OrderedDict
global glb_s_grad_at
global glb_c_grad_at

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
def isNaN(num):
    if num == float("inf"):
        return True
    return num != num

def bhatta_loss(output, target, prev=[], mode='numpy'):
    if mode == 'numpy':
        result_mul = output * target
        diff = result_mul - prev
        prev = result_mul
        result_abs = np.abs(result_mul)
        result_sqrt = np.sqrt(result_abs)
        result_sum = np.sum(result_sqrt, axis =(1, 2, 3))
        epsilon  = 0.00001
        result_log = np.log(result_sum + epsilon)
        out = -np.mean(result_log)
    else:
        out = -torch.log(torch.sum(torch.sqrt(torch.abs(torch.mul(output, target))), (1,2,3)))
        out = torch.mean(out)
    return out

def decay_lr(optimizer, epoch, init_lr, decay_period):
    lr = init_lr * (0.1 ** (epoch // decay_period))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.param_groups[7]['lr'] = lr * 10

def decay_lr_vgg(optimizer, epoch, init_lr, decay_period):
    lr = init_lr * (0.1 ** (epoch // decay_period))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def calculate_Gram_loss(s_feature, t_feature, norm_type, patch_num, style_weight, mse_loss, ratio):
    bn, c, h, w = t_feature.shape

    gram_losses = []

    # ratio = 4
    reduced_c = int(c / ratio)

    for i in range(patch_num):
        x_ = int(round((w * i / float(patch_num))))
        x_w = int(round((w * (i + 1) / float(patch_num))))

        for j in range(patch_num):
            y_ = int(round((h * j / float(patch_num))))
            y_h = int(round((h * (j + 1) / float(patch_num))))

            spatial_size = (y_h - y_) * (x_w - x_)

            t_vec = t_feature[:, :, y_: y_h, x_:x_w]    # shape: [bn, c, patch_h, patch_w]
            s_vec = s_feature[:, :, y_: y_h, x_:x_w]

            t_vec = t_vec.contiguous().view(bn, c, -1)  # shape: [bn, c, patch_h*w]
            s_vec = s_vec.contiguous().view(bn, c, -1)

            t_at = torch.sum(t_vec, 2)  # shape: [bn, c], channel attention map

            _, index = torch.sort(t_at, dim=1, descending=True) # shape: [bn, c]
            index, _ = torch.sort(index[:, :reduced_c], dim=1)    # shape: [bn, reduced_c]

            # indices = index.view(-1,1).repeat(1, spatial_size).view(t_vec.shape)
            # t_vec = torch.gather(t_vec, 1, indices)
            # s_vec = torch.gather(s_vec, 1, indices)
            # t_vec = t_vec[:,:c/ratio,:]
            # s_vec = s_vec[:,:c/ratio,:]

            t_vec = torch.cat([torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(t_vec, index)])
            s_vec = torch.cat([torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(s_vec, index)])

            if norm_type == 1:
                t_vec = t_vec.div(spatial_size)
                s_vec = s_vec.div(spatial_size)

            if norm_type == 2:
                t_vec = F.normalize(t_vec, p=2, dim=2)
                s_vec = F.normalize(s_vec, p=2, dim=2)

            t_Gram = torch.bmm(t_vec, t_vec.permute((0, 2, 1)))
            s_Gram = torch.bmm(s_vec, s_vec.permute((0, 2, 1)))

            if norm_type == 3 or norm_type == 4:
                t_Gram = t_Gram.div(spatial_size)
                s_Gram = s_Gram.div(spatial_size)

            gram_losses.append(mse_loss(s_Gram, t_Gram))

    if norm_type == 4:
        loss = style_weight * torch.mean(torch.stack(gram_losses)) / (reduced_c ** 2)

    else:
        loss = style_weight * torch.mean(torch.stack(gram_losses))

    return loss


def calculate_s_at(t_feature, s_feature, grad_at):
    bn, _, h, w = t_feature.shape
    t_at = torch.mean(torch.abs(t_feature), dim=1).view(bn, -1)
    # t_at = torch.sqrt(t_at)
    # t_at = t_at / torch.max(t_at, dim=1)[0].unsqueeze(1)
    print 'teacher at:', torch.min(t_at).data.cpu(), torch.mean(t_at).data.cpu(), torch.max(t_at).data.cpu()

    s_at = torch.mean(torch.abs(s_feature.detach()), dim=1).view(bn, -1)
    # s_at = torch.sqrt(s_at)
    # s_at = s_at / torch.max(s_at, dim=1)[0].unsqueeze(1)
    print 'student at:', torch.min(s_at).data.cpu(), torch.mean(s_at).data.cpu(), torch.max(s_at).data.cpu()

    r_at = torch.clamp(t_at - s_at, min=0.0).view(bn, h, w)
    r_at = torch.sqrt(r_at)
    at = r_at
    # print "r_at: ", torch.min(at).data.cpu(), torch.max(at).data.cpu(), torch.mean(at).data.cpu()
    
    # at = grad_at * t_at.view(bn, h, w) # torch.sqrt(grad_at * at.view(bn, h, w))
    # print 'at:', torch.min(at).data.cpu(), torch.mean(at).data.cpu(), torch.max(at).data.cpu()
    # at = F.adaptive_avg_pool2d(at.view(bn, 1, h, w), 5)
    # at = F.adaptive_avg_pool2d(at, h).view(bn, h, w)
    # print 'at:', torch.min(at).data.cpu(), torch.mean(at).data.cpu(), torch.max(at).data.cpu()
    
    # at = (at * t_at).view(bn, h, w)
    # at = torch.sqrt(at)

    return at


def calculate_attendedGram_loss(s_feature, t_feature, norm_type, style_weight, mse_loss, ratio):
    bn, c, h, w = t_feature.shape

    spatial_size = h * w

    # ratio = 10

    reduced_size = int(spatial_size / ratio)

    t_feature = t_feature.view(bn,c,-1)
    s_feature = s_feature.view(bn,c,-1)

    at = torch.sum(t_feature, dim=1)

    _, index= torch.sort(at, dim=1, descending=True) # [batch, h*w]
    index, _ = torch.sort(index[:, :reduced_size], dim=1)   # [batch, reduced_size]
    # index = index.detach()

    # indices = index.view(-1, 1).repeat(1, c).view(bn, -1, c).permute((0, 2, 1))
    # indices = indices.detach()
    # t_feature = torch.gather(t_feature, 2, indices)
    # s_feature = torch.gather(s_feature, 2, indices)
    # t_feature = t_feature[:,:,:(h*w)/ratio]
    # s_feature = s_feature[:,:,:(h*w)/ratio]

    t_feature = torch.cat([torch.index_select(a, 1, i).unsqueeze(0) for a, i in zip(t_feature, index)])
    s_feature = torch.cat([torch.index_select(a, 1, i).unsqueeze(0) for a, i in zip(s_feature, index)])

    t_Gram = torch.bmm(t_feature, t_feature.permute((0, 2, 1)))
    s_Gram = torch.bmm(s_feature, s_feature.permute((0, 2, 1)))

    loss = mse_loss(s_Gram, t_Gram)

    if norm_type == 3:
        loss = loss / (reduced_size ** 2)
    elif norm_type == 4:
        loss = loss / ((c*reduced_size) ** 2)

    loss *= style_weight

    return loss


def attendedFeature_loss(s_feature, t_feature, balance_weight, loss_fn, ratio, at):
    bn, c, h, w = t_feature.shape

    # print 'SR image shape : {}'.format(s_feature.shape)
    # print 'HR image shape : {}'.format(t_feature.shape)
    # print 'ATTENTION shape : {}'.format(at.shape)

    spatial_size = h * w
    reduced_size = int(spatial_size / ratio)

    # at = torch.sum(t_feature, dim=1)
    t_feature = t_feature.view(bn,c,-1)
    s_feature = s_feature.view(bn,c,-1)

    # FIXME: other mehod to calculate attention
    # at = torch.sum(t_feature, dim=1)
    # at = at.view(bn, -1)

    # Normalise to scale 1
    at = torch.div(at.view(bn, -1), torch.sum(at, dim =(1,2)).view(bn, 1))
    at = torch.mul(at, h * w)

    # _, index = torch.sort(at, dim=1, descending=True)
    # index, _ = torch.sort(index[:,:reduced_size], dim=1)

    # t_feature = torch.cat([torch.index_select(a, 1, i).unsqueeze(0) for a, i in zip(t_feature, index)])
    # s_feature = torch.cat([torch.index_select(a, 1, i).unsqueeze(0) for a, i in zip(s_feature, index)])

    # KD_loss = nn.KLDivLoss()(F.log_softmax(s_feature / 3, dim=1),
                                     # F.softmax(t_feature / 3, dim=1))    # teacher's hook is called in every loss.backward()
    # loss = torch.mul(KD_loss, 9)
    # loss = loss_fn(s_feature, t_feature)

    diff = torch.sub(t_feature, s_feature)
    diff = torch.mul(diff, diff)
    diff = torch.mul(diff, at.view(bn,1,-1))
    diff = torch.mean(diff)
    # loss *= balance_weight
    diff *= balance_weight

    if diff < 0 :
        import ipdb; ipdb.set_trace()
    return diff


def Feature_cs_at_loss(s_feature, t_feature, loss_fn, c_at, s_at, c, s):
    bn, c, h, w = s_feature.shape

    if s:
        s_at = (F.normalize(s_at.view(bn,-1), p=1, dim=1)).view(bn, h, w)
        print 's_at after norm : ', torch.min(s_at).data.cpu(), torch.mean(s_at.view(bn, -1)).data.cpu(), torch.max(s_at).data.cpu()
    if c:
        c_at = F.normalize(c_at, p=1, dim=1)
        print 'c_at after norm : ', torch.min(c_at).data.cpu(), torch.mean(c_at.view(bn, -1)).data.cpu(), torch.max(c_at).data.cpu()

    loss = loss_fn(s_feature, t_feature)
    loss = loss.view(bn, c, -1)

    if c:
        c_at = c_at.unsqueeze(2) # c_at = [bn, c, 1]
        loss = c_at * loss
    loss = torch.mean(loss, dim=1)
    loss = loss.view(bn, h, w)
    
    if s:
        loss = s_at * loss
    loss = torch.mean(loss)

    return loss


def channelAttendedFeature_loss(s_feature, t_feature, balance_weight, loss_fn, ratio, at):
    bn, c, h, w = t_feature.shape

    reduced_channel = int(c / ratio)

    _, index = torch.sort(at, dim=1, descending=True)
    index, _ = torch.sort(index[:, :reduced_channel], dim=1)

    t_feature = torch.cat([torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(t_feature, index)])
    s_feature = torch.cat([torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(s_feature, index)])

    loss = loss_fn(s_feature, t_feature)
    loss *= balance_weight

    return loss


def CAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps
    bz, nc, h, w = feature_conv.shape

    cam = weight_softmax[class_idx].unsqueeze(2).permute(0, 2, 1) * feature_conv.view(bz, nc, -1).permute(0, 2, 1)
    cam = cam.permute(0,2,1).view(bz, nc, h, w)
    cam = torch.sum(cam, dim=1)
    cam = cam.unsqueeze(1)

    return cam


def save_grad_at(module, grad_in, grad_out):
    global glb_s_grad_at
    global glb_c_grad_at
    # print('module hook')
    bn, c, h, w = grad_out[0].shape

    # Spatial Attention
    # grad_at = torch.sum(grad_out[0].detach(), dim=1)
    grad_at = torch.sum(torch.abs(grad_out[0].detach()), dim=1)
    # grad_at = torch.sum(torch.clamp(grad_out[0].detach(), min=0.0), dim=1)
    # grad_at = torch.clamp(torch.sum(grad_out[0].detach(), dim=1),min=0.0)

    glb_s_grad_at[id(module)] = grad_at

    # Channel Attention
    # grad_at = torch.mean(torch.abs(grad_out[0].detach()).view(bn, c, -1), dim=2)
    grad_at = torch.abs(torch.mean(grad_out[0].detach().view(bn, c, -1), dim=2))

    glb_c_grad_at[id(module)] = grad_at


def compute_gradCAM(feature, grad):
    bn, c, h, w= feature.shape
    # Assume : grad.shape = [bn,c]
    weight = grad
    weight = weight.unsqueeze(2)

    gradCAM = (weight * feature.view(bn, c, -1)).sum(dim=1)
    gradCAM = torch.clamp(gradCAM, min=0.0)

    gradCAM = gradCAM(bn, -1)
    gradCAM = gradCAM / torch.max(gradCAM, dim=1)[0].unsqueeze(1)
    gradCAM = gradCAM(bn, h, w)

    return gradCAM


def training(
    net,
    optimizer,
    init_lr,
    lr_decay,
    epochs,
    ten_crop,
    training_generator,
    eval_trainset_generator,
    eval_validationset_generator,
    num_training,
    num_validation,
    low_ratio,
    result_path,
    logger,
    vgg_gap,
    save
    ):
    lossfunction = nn.CrossEntropyLoss()
    if low_ratio != 0:
        modelName = '/teacher_LOW_{}x{}_'.format(str(low_ratio), str(low_ratio))
    else:
        modelName = '/teacher_HIGH_'
    print('modelName = ', result_path + modelName)
    for epoch in range(epochs):
        loss= 0.
        if not vgg_gap:
            decay_lr(optimizer, epoch, init_lr, lr_decay)
        else:
            decay_lr_vgg(optimizer, epoch, init_lr, lr_decay)
        net.train()
        for x, y in training_generator:
            # To CUDA tensors
            x = x.cuda().float()
            y = y.cuda() - 1

            # Calculate gradient && Backpropagate
            optimizer.zero_grad()

            # Network output
            output, _ = net(x)

            # Comment only when AlexNet returns only one val
            output = output[0]

            loss = lossfunction(output, y)
            loss.backward()
            optimizer.step()
        net.eval()
        # Test only 10, 20, 30... epochs
        if (epoch + 1) % 10 > 0 :
            logger.debug('[EPOCH{}][Training] loss : {}'.format(epoch+1,loss))
            continue
        hit_training = 0
        hit_validation = 0
        for x, y in eval_trainset_generator:
            # To CUDA tensors
            x = torch.squeeze(x)
            x = x.cuda().float()
            y -= 1

            # Network output
            output, _ = net(x)

            # Comment only when AlexNet returns only one val
            output = output[0]

            if ten_crop is True:
                prediction = torch.mean(output, dim=0)
                prediction = prediction.cpu().detach().numpy()

                if np.argmax(prediction) == y:
                    hit_training += 1
            else:
                _, prediction = torch.max(output, 1)
                prediction = prediction.cpu().detach().numpy()
                hit_training += (prediction == y.numpy()).sum()


        for x, y in eval_validationset_generator:
            # To CUDA tensors
            x = torch.squeeze(x)
            x = x.cuda().float()
            y -= 1

            # Network output
            output, _ = net(x)

            # Comment only when AlexNet returns only one val
            output = output[0]

            if ten_crop is True:
                prediction = torch.mean(output, dim=0)
                prediction = prediction.cpu().detach().numpy()

                if np.argmax(prediction) == y:
                    hit_validation += 1
            else:
                _, prediction = torch.max(output, 1)
                prediction = prediction.cpu().detach().numpy()
                hit_validation += (prediction == y.numpy()).sum()

        acc_training = float(hit_training) / num_training
        acc_validation = float(hit_validation) / num_validation
        logger.debug('Epoch : {}, training loss : {}'.format(epoch + 1, loss))
        logger.debug('    Training   set accuracy : {0:.2f}%, for {1:}/{2:}'
              .format(acc_training*100, hit_training, num_training))
        logger.debug('    Validation set accuracy : {0:.2f}%, for {1:}/{2:}\n'
              .format(acc_validation*100, hit_validation, num_validation))
        if acc_validation < 1. :
            logger.error('This combination seems not working, stop training')
            exit(1)
        if save:
            # torch.save(net.state_dict(), result_path + modelName + str(epoch + 1) + '_epoch_acc_' + str(acc_validation*100) +'.pt')
            torch.save(net.state_dict(),
                       result_path + modelName + str(epoch + 1) + '_epoch_acc_' + str(acc_validation * 100) + '.pt')


def training_KD(
    teacher_net,
    net,
    optimizer,
    temperature,
    init_lr,
    lr_decay,
    epochs,
    ten_crop,
    training_generator,
    eval_trainset_generator,
    eval_validationset_generator,
    num_training,
    num_validation,
    low_ratio,
    result_path,
    logger,
    vgg_gap,
    save
    ):
    lossfunction = nn.CrossEntropyLoss()
    writer = SummaryWriter()
    max_accuracy = 0.
    if low_ratio != 0:
        modelName = '/Student_LOW_{}x{}_'.format(str(low_ratio), str(low_ratio))
    else:
        print('are you serious ...?')

    teacher_net.eval()
    kdloss = AverageMeter()
    gtloss = AverageMeter()
    convloss = AverageMeter()
    prev = []
    bhlosses = []


    temperature2 = 5

    for epoch in range(epochs):
        loss= 0.
        if not vgg_gap:
            decay_lr(optimizer, epoch, init_lr, lr_decay)
        else:
            decay_lr_vgg(optimizer, epoch, init_lr, lr_decay)
        net.train()
        for x, x_low, y in training_generator:
            # To CUDA tensors
            x = x.cuda().float()
            x_low = x_low.cuda().float()
            y = y.cuda() - 1

            # teacher = teacher_net(x)
            teacher, t_feature = teacher_net(x)
            # t_feature = t_feature.detach()

            # Calculate gradient && Backpropagate
            optimizer.zero_grad()

            # Network output
            student, s_feature = net(x_low)

            # Calculate Region KD between CAM region of teacher & student

            t_convs = []
            s_convs = []

            for k,v in t_feature.items():
                t_convs.append(v)
            for k,v in s_feature.items():
                s_convs.append(v)

            # BH_loss = bhatta_loss(t_convs[0], s_convs[0], mode ='tensor')
            MSE_loss = 0


            KD_loss = nn.KLDivLoss()(F.log_softmax(student / temperature, dim=1),
                                     F.softmax(teacher / temperature, dim=1))

            KD_loss = torch.mul(KD_loss, temperature * temperature)

            GT_loss = lossfunction(student, y)


            # loss = KD_loss + GT_loss - BH_loss
            loss = KD_loss + GT_loss
            convloss.update(0)
            if isNaN(loss.item()) is True:
                logger.error("This combination failed due to the NaN|inf loss value")
                exit(1)

            kdloss.update(KD_loss.item(), x_low.size(0))
            gtloss.update(GT_loss.item(), x_low.size(0))

            # for i in range(len(t_convs)):
                # t_conv = t_convs[i].cpu().detach().numpy()
                # s_conv = s_convs[i].cpu().detach().numpy()
                # if len(prev) < i + 1:
                    # prev.append(np.zeros(t_conv.shape))
                    # bhlosses.append(bhatta_loss(t_conv, s_conv, prev[i]))
                # else:
                    # bhlosses[i] = bhatta_loss(t_conv, s_conv, prev[i])

            # logger.debug('[EPOCH{}][Training][KD loss : {:.3f}][GT_loss : {:.3f}][conv_l2_loss : {:.3f}]\n'
                         # '\t[CONV1 Distance : {}]'
                         # '\t[CONV2 Distance : {}]'
                         # '\t[CONV3 Distance : {}]'
                         # '\t[CONV4 Distance : {}]'
                         # '\t[CONV5 Distance : {}]'
                         # .format(epoch+1,kdloss.avg, gtloss.avg, convloss.avg,
                                 # bhlosses[0], bhlosses[1], bhlosses[2], bhlosses[3],bhlosses[4]))
            # logger.debug('\t[CONV1 Distance : {}]'
                         # '\t[CONV2 Distance : {}]'
                         # '\t[CONV5 Distance : {}]'
                         # '\t[CONV4 Distance : {}]'
                         # '\t[CONV5 Distance : {}]'
                         # .format(bhlosses[0], bhlosses[1], bhlosses[2], bhlosses[3],bhlosses[4]))

            loss.backward()
            optimizer.step()
        net.eval()

        if (epoch + 1) % 10 > 0 :
            # print('Epoch : {}, training loss : {}'.format(epoch + 1, loss))
            logger.debug('[EPOCH{}][Training][KD loss : {:.3f}][GT_loss : {:.3f}][MSE_LOSS : {:.3f}]'
                         .format(epoch+1,kdloss.avg, gtloss.avg, convloss.avg))
            # logger.debug('[EPOCH{}][Training][KD loss : {:.3f}][GT_loss : {:.3f}][conv_l2_loss : {:.3f}]\n'
                         # '\t[CONV1 Distance : {:.2f}]'
                         # '\t[CONV2 Distance : {:.2f}]'
                         # '\t[CONV2 Distance : {:.2f}]'
                         # '\t[CONV4 Distance : {:.2f}]'
                         # '\t[CONV5 Distance : {:.2f}]\n'
                         # .format(epoch+1,kdloss.avg, gtloss.avg, convloss.avg,
                                 # bhlosses[0], bhlosses[1], bhlosses[2], bhlosses[3],bhlosses[4]))
            writer.add_scalars('losses', {'KD_loss':kdloss.avg,
                                          'GT_loss':gtloss.avg,
                                          'MSE_loss':convloss.avg,
                                          }, epoch + 1)
            continue
        # Test
        hit_training = 0
        hit_validation = 0
        for x_low, y in eval_trainset_generator:
            # To CUDA tensors
            x_low = torch.squeeze(x_low)
            x_low = x_low.cuda().float()
            y -= 1

            # Network output
            output, _ = net(x_low)

            if ten_crop is True:
                prediction = torch.mean(output, dim=0)
                prediction = prediction.cpu().detach().numpy()

                if np.argmax(prediction) == y:
                    hit_training += 1
            else:
                _, prediction = torch.max(output, 1)
                prediction = prediction.cpu().detach().numpy()
                hit_training += (prediction == y.numpy()).sum()


        count_success = 0
        count_failure = 0
        count_show = 3
        success = []
        failure = []
        sr_success = []
        sr_failure = []
        for  x_low, y in eval_validationset_generator:
            # To CUDA tensors
            x_low = torch.squeeze(x_low)
            x_low = x_low.cuda().float()
            y -= 1

            # Network output
            output, _ = net(x_low)

            if ten_crop is True:
                prediction = torch.mean(output, dim=0)
                prediction = prediction.cpu().detach().numpy()

                # sr_image = vutils.make_grid(sr_x[0], normalize=True, scale_each=True)
                # image = vutils.make_grid(x_low[0], normalize=True, scale_each=True)

                if np.argmax(prediction) == y:
                    hit_validation += 1
                    if count_success > count_show :
                        continue
                    success.append(x_low[0])
                    # sr_success.append(sr_x[0])
                    count_success += 1
                elif count_failure < count_show + 1:
                    count_failure += 1
                    failure.append(x_low[0])
                    # sr_failure.append(sr_x[0])
            else:
                _, prediction = torch.max(output, 1)
                prediction = prediction.cpu().detach().numpy()
                hit_validation += (prediction == y.numpy()).sum()

        if ten_crop is True:
            torch.stack(success, dim=0)
            torch.stack(failure, dim=0)
            # torch.stack(sr_success, dim=0)
            # torch.stack(sr_failure, dim=0)
            success = vutils.make_grid(success, normalize=True, scale_each=True)
            failure = vutils.make_grid(failure, normalize=True, scale_each=True)
            # sr_success = vutils.make_grid(sr_success, normalize=True, scale_each=True)
            # sr_failure = vutils.make_grid(sr_failure, normalize=True, scale_each=True)

            writer.add_image('Success', success, epoch + 1)
            # writer.add_image('SR_Success', sr_success, epoch + 1)
            writer.add_image('Failure', failure, epoch + 1)
            # writer.add_image('SR_Failure', sr_failure, epoch + 1)
        # Trace
        acc_training = float(hit_training) / num_training
        acc_validation = float(hit_validation) / num_validation
        # logger.debug('[EPOCH{}][Training][KD loss : {:.3f}][GT_loss : {:.3f}]\n'
                     # '\t[CONV1 BHLOSS : {:.3f}]'
                     # '\t[CONV2 BHLOSS : {:.3f}]'
                     # '\t[CONV3 BHLOSS : {:.3f}]'
                     # '\t[CONV4 BHLOSS : {:.3f}]'
                     # '\t[CONV5 BHLOSS : {:.3f}]'
                     # .format(epoch+1,kdloss.avg, gtloss.avg,
                             # bhlosses[0], bhlosses[1], bhlosses[2], bhlosses[3],bhlosses[4]))
        logger.debug('[EPOCH{}][Training][KD loss : {:.3f}][GT_loss : {:.3f}][MSE_loss : {:.3f}]'
                     .format(epoch+1,kdloss.avg, gtloss.avg, convloss.avg))
        # logger.debug('Epoch : {}, training loss : {}'.format(epoch + 1, loss))
        logger.debug('    Training   set accuracy : {0:.2f}%, for {1:}/{2:}'
              .format(acc_training*100, hit_training, num_training))
        logger.debug('    Validation set accuracy : {0:.2f}%, for {1:}/{2:}\n'
              .format(acc_validation*100, hit_validation, num_validation))
        if max_accuracy < acc_validation * 100 : max_accuracy = acc_validation
        if save:
            torch.save(net.state_dict(), result_path + modelName + str(epoch + 1) + '_epoch_acc_' + str(acc_validation* 100) + '.pt')
    logger.debug('Finished Training\n')
    logger.debug('MAX_ACCURACY : {:.2f}'.format(max_accuracy * 100))


def training_Gram_KD(
    teacher_net,
    net,
    optimizer,
    temperature,
    init_lr,
    lr_decay,
    epochs,
    ten_crop,
    training_generator,
    eval_trainset_generator,
    eval_validationset_generator,
    num_training,
    num_validation,
    low_ratio,
    result_path,
    logger,
    style_weight,
    norm_type,
    patch_num,
    gram_features,
    hint,
    at_enabled,
    at_ratio,
    save,
    c,
    s
    ):
    global glb_s_grad_at
    global glb_c_grad_at
    glb_s_grad_at = OrderedDict()
    glb_c_grad_at = OrderedDict()


    ce_loss = nn.CrossEntropyLoss()
    # mse_loss = nn.MSELoss()
    mse_loss = nn.MSELoss(reduce=False)
    max_accuracy = 0.

    kdloss = AverageMeter()
    gtloss = AverageMeter()
    convloss = AverageMeter()

    if low_ratio != 0:
        modelName = '/Student_LOW_{}x{}_'.format(str(low_ratio), str(low_ratio))
    else:
        print('are you serious ...?')

    teacher_net.eval()

    if hint:
        print('1st stage Training using Gram loss')
        for epoch in range(40):
            loss = 0.
            decay_lr(optimizer, epoch, init_lr, lr_decay)
            net.train()

            for x, x_low, y in training_generator:
                x = x.cuda().float()
                x_low = x_low.cuda().float()
                y = y.cuda() - 1

                _, t_features = teacher_net(x)

                t_conv1 = t_features['conv1'].detach()
                t_conv2 = t_features['conv2'].detach()
                t_conv3 = t_features['conv3'].detach()
                t_conv4 = t_features['conv4'].detach()
                t_conv5 = t_features['conv5'].detach()

                optimizer.zero_grad()

                _, s_features = net(x_low)

                s_conv1 = s_features['conv1']
                s_conv2 = s_features['conv2']
                s_conv3 = s_features['conv3']
                s_conv4 = s_features['conv4']
                s_conv5 = s_features['conv5']

                loss = []

                # feature regression method
                if 1 in gram_features:
                    loss.append(mse_loss(s_conv1, t_conv1) * style_weight)
                if 2 in gram_features:
                    loss.append(mse_loss(s_conv2, t_conv2) * style_weight)
                if 3 in gram_features:
                    loss.append(mse_loss(s_conv3, t_conv3) * style_weight)
                if 4 in gram_features:
                    loss.append(mse_loss(s_conv4, t_conv4) * style_weight)
                if 5 in gram_features:
                    loss.append(mse_loss(s_conv5, t_conv5) * style_weight)

                # print loss

                loss = torch.mean(torch.stack(loss))

                # print loss.data.cpu()

                if loss == float('inf') or loss != loss:
                    logger.error('Loss is infinity, stop!')
                    return

                loss.backward()
                optimizer.step()
            print('In 1st stage, epoch : {}, total loss : {}'.format(
                    epoch, loss.data.cpu()))

    # To calculate and save gradient attetion, register backward_hook
    """
    teacher_net.conv1.register_backward_hook(save_grad_at)
    teacher_net.conv2.register_backward_hook(save_grad_at)
    teacher_net.conv3.register_backward_hook(save_grad_at)
    teacher_net.conv4.register_backward_hook(save_grad_at)
    teacher_net.conv5.register_backward_hook(save_grad_at)
    """
    teacher_net.pool1.register_backward_hook(save_grad_at)
    teacher_net.pool2.register_backward_hook(save_grad_at)
    teacher_net.relu3.register_backward_hook(save_grad_at)
    teacher_net.relu4.register_backward_hook(save_grad_at)
    teacher_net.pool5.register_backward_hook(save_grad_at)

    """
    net.pool1.register_backward_hook(save_grad_at)
    net.pool2.register_backward_hook(save_grad_at)
    net.relu3.register_backward_hook(save_grad_at)
    net.relu4.register_backward_hook(save_grad_at)
    net.pool5.register_backward_hook(save_grad_at)
    """

    for epoch in range(epochs):
        loss= 0.
        decay_lr(optimizer, epoch, init_lr, lr_decay)
        net.train()
        for x, x_low, y in training_generator:
            # To CUDA tensors
            x = x.cuda().float()
            x = Variable(x, requires_grad = True)
            x_low = x_low.cuda().float()

            y = y.cuda() - 1

            teacher, t_features = teacher_net(x)

            one_hot_y = torch.zeros(teacher.shape).float().cuda()
            for i in range(teacher.shape[0]):
                one_hot_y[i][y[i]] = 1.0

            teacher_net.zero_grad()
            teacher.backward(gradient=one_hot_y, retain_graph=True)

            t_conv1 = t_features['conv1'].detach()
            t_conv2 = t_features['conv2'].detach()
            t_conv3 = t_features['conv3'].detach()
            t_conv4 = t_features['conv4'].detach()
            t_conv5 = t_features['conv5'].detach()

            # # Calculate gradient && Backpropagate
            # optimizer.zero_grad()

            # Network output
            student, s_features = net(x_low)

            net.zero_grad()
            student.backward(gradient=one_hot_y, retain_graph=True)

            # Calculate gradient && Backpropagate
            optimizer.zero_grad()

            s_conv1 = s_features['conv1']
            s_conv2 = s_features['conv2']
            s_conv3 = s_features['conv3']
            s_conv4 = s_features['conv4']
            s_conv5 = s_features['conv5']

            KD_loss = nn.KLDivLoss()(F.log_softmax(student / temperature, dim=1),
                                  # F.softmax(teacher / temperature, dim=1))    # teacher's hook is called in every loss.backward()
                                  F.softmax(teacher.detach() / temperature, dim=1))

            KD_loss = torch.mul(KD_loss, temperature * temperature)
            # KD_loss = .0
            
            GT_loss = ce_loss(student, y)

            GRAM_loss = .0

            # feature regression with attention
            if hint == False:
                if str(1) in gram_features:
                    # GRAM_loss += mse_loss(s_conv1, t_conv1)
                    c_at = glb_c_grad_at[id(teacher_net.pool1)]
                    s_at = calculate_s_at(t_conv1.detach(), s_conv1.detach(), glb_s_grad_at[id(teacher_net.pool1)])
                    GRAM_loss += Feature_cs_at_loss(s_conv1, t_conv1, mse_loss, c_at, s_at, c, s)
                if str(2) in gram_features:
                    # GRAM_loss += mse_loss(s_conv2, t_conv2)
                    c_at = glb_c_grad_at[id(teacher_net.pool2)]
                    s_at = calculate_s_at(t_conv2.detach(), s_conv2.detach(), glb_s_grad_at[id(teacher_net.pool2)])
                    GRAM_loss += Feature_cs_at_loss(s_conv2, t_conv2, mse_loss, c_at, s_at, c, s)
                if str(3) in gram_features:
                    # GRAM_loss += mse_loss(s_conv3, t_conv3)
                    c_at = glb_c_grad_at[id(teacher_net.relu3)]
                    s_at = calculate_s_at(t_conv3.detach(), s_conv3.detach(), glb_s_grad_at[id(teacher_net.relu3)])
                    GRAM_loss += Feature_cs_at_loss(s_conv3, t_conv3, mse_loss, c_at, s_at, c, s)
                if str(4) in gram_features:
                    # GRAM_loss += mse_loss(s_conv4, t_conv4)
                    c_at = glb_c_grad_at[id(teacher_net.relu4)]
                    s_at = calculate_s_at(t_conv4.detach(), s_conv4.detach(), glb_s_grad_at[id(teacher_net.relu4)])
                    GRAM_loss += Feature_cs_at_loss(s_conv4, t_conv4, mse_loss, c_at, s_at, c, s)
                if str(5) in gram_features:
                    # GRAM_loss += mse_loss(s_conv5, t_conv5)
                    c_at = glb_c_grad_at[id(teacher_net.pool5)]
                    s_at = calculate_s_at(t_conv5.detach(), s_conv5.detach(), glb_s_grad_at[id(teacher_net.pool5)])
                    GRAM_loss += Feature_cs_at_loss(s_conv5, t_conv5, mse_loss, c_at, s_at, c, s)

                GRAM_loss *= style_weight

            loss = KD_loss + GT_loss + GRAM_loss
            kdloss.update(KD_loss.item(), x_low.size(0))
            gtloss.update(GT_loss.item(), x_low.size(0))
            convloss.update(GRAM_loss.item(), x_low.size(0))

            if loss == float('inf') or loss != loss:
                logger.error('Loss is infinity, stop!')
                return

            # if hint == False:
            #     print KD_loss.data.cpu(), GT_loss.data.cpu(), GRAM_loss.data.cpu()
            # else:
            #     print KD_loss.data.cpu(), GT_loss.data.cpu()

            loss.backward()
            optimizer.step()
        net.eval()

        if (epoch + 1) % 10 > 0 :
            logger.debug('[EPOCH{}][Training][KD loss : {:.3f}][GT_loss : {:.3f}][MSE_loss : {:.3f}]'
                     .format(epoch+1,kdloss.avg, gtloss.avg, convloss.avg))
            continue
        # Test
        hit_training = 0
        hit_validation = 0
        for x_low, y in eval_trainset_generator:
            # To CUDA tensors
            x_low = torch.squeeze(x_low)
            x_low = x_low.cuda().float()
            y -= 1

            # Network output
            output, _ = net(x_low)

            if ten_crop is True:
                prediction = torch.mean(output, dim=0)
                prediction = prediction.cpu().detach().numpy()

                if np.argmax(prediction) == y:
                    hit_training += 1
            else:
                _, prediction = torch.max(output, 1)
                prediction = prediction.cpu().detach().numpy()
                hit_training += (prediction == y.numpy()).sum()


        for  x_low, y in eval_validationset_generator:
            # To CUDA tensors
            x_low = torch.squeeze(x_low)
            x_low = x_low.cuda().float()
            y -= 1

            # Network output
            output, _ = net(x_low)

            if ten_crop is True:
                prediction = torch.mean(output, dim=0)
                prediction = prediction.cpu().detach().numpy()

                if np.argmax(prediction) == y:
                    hit_validation += 1
            else:
                _, prediction = torch.max(output, 1)
                prediction = prediction.cpu().detach().numpy()
                hit_validation += (prediction == y.numpy()).sum()


        # Trace
        acc_training = float(hit_training) / num_training
        acc_validation = float(hit_validation) / num_validation
        logger.debug('[EPOCH{}][Training][KD loss : {:.3f}][GT_loss : {:.3f}][MSE_loss : {:.3f}]'
                     .format(epoch+1,kdloss.avg, gtloss.avg, convloss.avg))
        # logger.debug('Epoch : {}, training loss : {}'.format(epoch + 1, loss))
        logger.debug('    Training   set accuracy : {0:.2f}%, for {1:}/{2:}'
              .format(acc_training*100, hit_training, num_training))
        logger.debug('    Validation set accuracy : {0:.2f}%, for {1:}/{2:}\n'
              .format(acc_validation*100, hit_validation, num_validation))

        if max_accuracy < acc_validation: max_accuracy = acc_validation
        if acc_validation < 0.01 :
            logger.error('This combination seems not working, stop training')
            exit(1)

        if save:
            torch.save(net.state_dict(), result_path + modelName + str(epoch + 1) + '_epoch_acc_' + str(acc_validation* 100) + '.pt')
    logger.debug('Finished Training\n')
    logger.debug('MAX_ACCURACY : {:.2f}'.format(max_accuracy * 100))

def training_attention_SR(
    teacher_net,
    net,
    optimizer,
    temperature,
    init_lr,
    lr_decay,
    epochs,
    ten_crop,
    training_generator,
    eval_trainset_generator,
    eval_validationset_generator,
    num_training,
    num_validation,
    low_ratio,
    result_path,
    logger,
    attention_weight,
    norm_type,
    patch_num,
    gram_features,
    at_enabled,
    at_ratio,
    save,
    description
    ):
    global glb_grad_at
    glb_grad_at = OrderedDict()

    teacher_net.conv5.register_backward_hook(save_grad_at)
    writer = SummaryWriter('_'.join(('runs/',datetime.datetime.now().strftime('%Y-%m-%d'), description)))
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    max_accuracy = 0.

    gtloss = AverageMeter()
    srloss = AverageMeter()
    kdloss = AverageMeter()

    if low_ratio != 0:
        modelName = '/Student_LOW_{}x{}_'.format(str(low_ratio), str(low_ratio))
    else:
        print('are you serious ...?')


    for epoch in range(epochs):
        loss= 0.
        # decay_lr(optimizer, epoch, init_lr, lr_decay)
        net.train()
        for x, x_low, y in training_generator:
            # training_bar.set_description('TRAINING EPOCH[{}/{}]'.format(i, str(46)))
            # To CUDA tensors
            x = x.cuda().float()
            x_low = x_low.cuda().float()

            y = y.cuda() - 1

            # Calculate gradient && Backpropagate
            optimizer.zero_grad()

            # Network output
            teacher, t_features = teacher_net(x)
            sr_image, output = net(x_low)
            output, s_features = output

            one_hot_y = torch.zeros(output.shape).float().cuda()
            for i in range(output.shape[0]):
                one_hot_y[i][y[i]] = 1.0

            teacher.backward(gradient = one_hot_y, retain_graph = True)


            # SR_loss = attendedFeature_loss(sr_image, x, attention_weight, mse_loss, at_ratio, glb_grad_at[id(net.srLayer)])
            SR_loss = attendedFeature_loss(s_features['conv5'], t_features['conv5'], 1, mse_loss, 2, glb_grad_at[id(teacher_net.conv5)])
            # SR_loss = 0

            # SR_loss.backward(gradient=one_hot_y)
            GT_loss = ce_loss(output, y)
            KD_loss = nn.KLDivLoss()(F.log_softmax(output / 3, dim=1),
                                     F.softmax(teacher / 3, dim=1))    # teacher's hook is called in every loss.backward()

            loss = GT_loss + KD_loss + SR_loss
            optimizer.zero_grad()
            loss.backward()

            gtloss.update(GT_loss.item(), x_low.size(0))
            srloss.update(SR_loss, x_low.size(0))
            kdloss.update(KD_loss.item(), x_low.size(0))

            if SR_loss == float('inf') or SR_loss != SR_loss:
                logger.error('SR_loss value : {}'.format(SR_loss.item()))
                logger.error('Loss is infinity, stop!')
                exit(1)

            optimizer.step()
        writer.add_scalars('losses', {'CE': gtloss.avg,
                                      'RECON': srloss.avg,
                                      'KD': kdloss.avg}, epoch + 1)
        net.eval()

        if (epoch + 1) % 10 > 0 :
            logger.debug('[EPOCH{}][Training][GT_LOSS : {:.3f}][RECON_LOSS : {:.3f}][KD_LOSS : {:.3f}]'
                     .format(epoch+1, gtloss.avg, srloss.avg, kdloss.avg))
            continue
        # Test
        hit_training = 0
        hit_validation = 0
        eval_training_bar = tqdm(eval_trainset_generator)
        eval_validation_bar = tqdm(eval_validationset_generator)
        count_success = 0
        count_failure = 0
        count_show = 3
        success = []
        failure = []
        sr_success = []
        sr_failure = []
        for i,(x_low, y) in enumerate(eval_training_bar):
            eval_training_bar.set_description('TESTING TRAINING SET, PROCESSING BATCH[{}/{}]'.format(i, str(num_training)))
            # To CUDA tensors
            x_low = torch.squeeze(x_low)
            x_low = x_low.cuda().float()
            y -= 1

            # Network output
            sr_image, output = net(x_low)
            output, features = output

            if ten_crop is True:
                prediction = torch.mean(output, dim=0)
                prediction = prediction.cpu().detach().numpy()

                if np.argmax(prediction) == y:
                    hit_training += 1
            else:
                _, prediction = torch.max(output, 1)
                prediction = prediction.cpu().detach().numpy()
                hit_training += (prediction == y.numpy()).sum()


        for i,(x_low, y) in enumerate(eval_validation_bar):
            eval_validation_bar.set_description('TESTING TEST SET, PROCESSING BATCH[{}/{}]'.format(i, str(num_validation)))
            # To CUDA tensors
            x_low = torch.squeeze(x_low)
            x_low = x_low.cuda().float()
            y -= 1

            # Network output
            sr_image, output = net(x_low)
            output, features = output

            if ten_crop is True:
                prediction = torch.mean(output, dim=0)
                prediction = prediction.cpu().detach().numpy()

                if np.argmax(prediction) == y:
                    hit_validation += 1
                    if count_success > count_show :
                        continue
                    success.append(x_low[0])
                    sr_success.append(sr_image[0])
                    count_success += 1
                elif count_failure < count_show + 1:
                    count_failure += 1
                    failure.append(x_low[0])
                    sr_failure.append(sr_image[0])
            else:
                _, prediction = torch.max(output, 1)
                prediction = prediction.cpu().detach().numpy()
                hit_validation += (prediction == y.numpy()).sum()

        torch.stack(success, dim=0)
        torch.stack(failure, dim=0)
        torch.stack(sr_success, dim=0)
        torch.stack(sr_failure, dim=0)
        success = vutils.make_grid(success, normalize=True, scale_each=True)
        failure = vutils.make_grid(failure, normalize=True, scale_each=True)
        sr_success = vutils.make_grid(sr_success, normalize=True, scale_each=True)
        sr_failure = vutils.make_grid(sr_failure, normalize=True, scale_each=True)

        writer.add_image('Success', success, epoch + 1)
        writer.add_image('SR_Success', sr_success, epoch + 1)
        writer.add_image('Failure', failure, epoch + 1)
        writer.add_image('SR_Failure', sr_failure, epoch + 1)

        # Trace
        acc_training = float(hit_training) / num_training
        acc_validation = float(hit_validation) / num_validation
        logger.debug('[EPOCH{}][Training][GT_LOSS : {:.3f}][RECON_LOSS : {:.3f}]'
                     .format(epoch+1, gtloss.avg, srloss.avg))
        # logger.debug('Epoch : {}, training loss : {}'.format(epoch + 1, loss))
        logger.debug('    Training   set accuracy : {0:.2f}%, for {1:}/{2:}'
              .format(acc_training*100, hit_training, num_training))
        logger.debug('    Validation set accuracy : {0:.2f}%, for {1:}/{2:}\n'
              .format(acc_validation*100, hit_validation, num_validation))

        if max_accuracy < acc_validation: max_accuracy = acc_validation
        if acc_validation < 0.01 :
            logger.error('This combination seems not working, stop training')
            exit(1)

        if save:
            torch.save(net.state_dict(), result_path + modelName + str(epoch + 1) + '_epoch_acc_' + str(acc_validation* 100) + '.pt')
    logger.debug('Finished Training\n')
    logger.debug('MAX_ACCURACY : {:.2f}'.format(max_accuracy * 100))
