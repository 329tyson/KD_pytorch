import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
import datetime
import cv2
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from logger import Logger
from logger import display_function_stack
from logger import AverageMeter

from collections import OrderedDict
glb_grad = {}
glb_t_grad = {}
glb_s_grad = {}

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

    # for lr decay in SHAREDALEXNET training
    optimizer.param_groups[4]['lr'] = lr * 10

    # for lr decay in ALEXNET training
    # optimizer.param_groups[7]['lr'] = lr * 10

def adjust_learning_rate(optimizer, epoch):
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        lr = lr * (0.1 ** (epoch // 300))
        param_group['lr'] = lr

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

    spatial_size = h * w
    reduced_size = int(spatial_size / ratio)

    # at = torch.sum(t_feature, dim=1)
    t_feature = t_feature.view(bn,c,-1)
    s_feature = s_feature.view(bn,c,-1)

    # Normalise to scale 1
    at = torch.div(at.view(bn, -1), torch.sum(at, dim =(1,2)).view(bn, 1))
    at = torch.mul(at, h * w)

    diff = torch.sub(t_feature, s_feature)
    diff = torch.mul(diff, diff)
    diff = torch.mul(diff, at.view(bn,1,-1))
    diff = torch.mean(diff)
    # loss *= balance_weight
    diff *= balance_weight

    if diff < 0 :
        import ipdb; ipdb.set_trace()
    return diff


def CAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps
    bz, nc, h, w = feature_conv.shape

    cam = weight_softmax[class_idx].unsqueeze(2).permute(0, 2, 1) * feature_conv.view(bz, nc, -1).permute(0, 2, 1)
    cam = cam.permute(0,2,1).view(bz, nc, h, w)
    cam = torch.sum(cam, dim=1)
    cam = cam.unsqueeze(1)

    return cam


def save_grad_s(module, grad_in, grad_out):
    global glb_s_grad
    glb_s_grad[id(module)] = grad_out[0].detach()

def save_grad_t(module, grad_in, grad_out):
    global glb_t_grad
    glb_t_grad[id(module)] = grad_out[0].detach()

def save_grad(module, grad_in, grad_out):
    global glb_grad
    glb_grad[id(module)] = grad_out[0].detach()
@display_function_stack
def shared_training(
    net,
    optim_hr,
    optim_lr,
    ten_crop,
    training_generator,
    eval_trainset_generator,
    eval_validationset_generator,
    num_training,
    num_validation,
    result_path,
    epochs,
    low_ratio,
    init_lr,
    lr_decay,
    logger = None
    ):
    print '\n\n\n\n'
    celoss = nn.CrossEntropyLoss()
    max_accuracy = 0.
    hr_loss = AverageMeter('hr_loss')
    lr_loss = AverageMeter('lr_loss')
    acc_validation = AverageMeter('v_acc')
    net.train()
    for epoch in range(epochs):
        hr_loss.reset()
        lr_loss.reset()
        # decay_lr(optim_total, epoch, init_lr, lr_decay)
        decay_lr(optim_hr, epoch, init_lr, lr_decay)
        decay_lr(optim_lr, epoch, init_lr, lr_decay)
        training_pbar = tqdm(enumerate(training_generator), desc='Training ', bar_format='{desc:<5} [R {rate_fmt}]', leave=False)

        for i, (x, x_low, y) in training_pbar:
            optim_hr.zero_grad()
            optim_lr.zero_grad()

            x = x.cuda().float()
            x_low = x_low.cuda().float()
            y = y.cuda() - 1

            hr_out, lr_out, hr_features, lr_features = net(x_low, x)
            gt_hr_loss = celoss(hr_out, y)
            gt_lr_loss = celoss(lr_out, y)

            hr_loss.update(gt_hr_loss.item(), x.size(0))
            lr_loss.update(gt_lr_loss.item(), x_low.size(0))

            loss = gt_hr_loss + gt_lr_loss
            loss.backward()
            training_pbar.set_description('TRAINING, E[{}/{}]B[{}/{}][HIGH_GT : {:.3f}][LOW_GT : {:.3f}]'.format(epoch + 1, str(epochs), i + 1, str(int(num_training / 128)) ,hr_loss.avg, lr_loss.avg ))
            training_pbar.update()
            training_pbar.refresh()
            # from time import sleep; sleep(1)

            optim_hr.step()
            optim_lr.step()


        logger.iteration(hr_loss, lr_loss, E=str(epoch+1) + '/' + str(epochs))
        if (epoch + 1) % 10 > 0:
            continue
        hit_training = 0
        hit_validation = 0
        test_validation_pbar = tqdm(eval_validationset_generator, desc='Validation', bar_format='{desc:<5} [R {rate_fmt}]', leave=False)
        # for i, (x_low, y) in enumerate(test_training_pbar): # To CUDA tensors
            # x_low = torch.squeeze(x_low)
            # x_low = x_low.cuda().float()
            # y -= 1

            # # Network output
            # _, output ,_, _= net(x_low, x)

            # if ten_crop is True:
                # prediction = torch.mean(output, dim=0)
                # prediction = prediction.cpu().detach().numpy()

                # if np.argmax(prediction) == y:
                    # hit_training += 1
            # else:
                # _, prediction = torch.max(output, 1)
                # prediction = prediction.cpu().detach().numpy()
                # hit_training += (prediction == y.numpy()).sum()
                # test_training_pbar.set_description('TESTING TRAINING SET, PROCESSING BATCH[{}/{}][ACC:{}]'.format(i + 1, str(num_training)))
        acc_validation.reset()
        for  i, (x_low, y) in enumerate(test_validation_pbar):
            test_validation_pbar.set_description('TRAINING, E[{}/{}]B[{}/{}]'.format(epoch + 1, str(epochs), i + 1, str(num_validation)))
            test_validation_pbar.update()
            test_validation_pbar.refresh()
            # To CUDA tensors
            x_low = torch.squeeze(x_low)
            x_low = x_low.cuda().float()
            y -= 1

            # Network output
            _, output, _, _ = net(x_low, x)

            if ten_crop is True:
                prediction = torch.mean(output, dim=0)
                prediction = prediction.cpu().detach().numpy()

                if np.argmax(prediction) == y:
                    hit_validation += 1
            else:
                _, prediction = torch.max(output, 1)
                prediction = prediction.cpu().detach().numpy()
                hit_validation += (prediction == y.numpy()).sum()
        acc_validation.update(float(hit_validation)/float(num_validation), 1)
        if max_accuracy < acc_validation.avg * 100 : max_accuracy = acc_validation
        logger.iteration(acc_validation, E=str(epoch+1) + '/' + str(epochs))
    logger.message('Finished Training\n')
    logger.message('MAX_ACCURACY : {:.2f}'.format(max_accuracy * 100))
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
        # for x, y in training_generator:
        for x, y in training_generator:
            # To CUDA tensors
            x = x.cuda().float()
            y = y.cuda() - 1

            # Calculate gradient && Backpropagate
            optimizer.zero_grad()

            # Network output
            output, features = net(x)

            loss = lossfunction(output, y)
            loss.backward()
            optimizer.step()
        net.eval()
        # Test only 10, 20, 30... epochs
        if (epoch + 1) % 10 > 0 :
            logger.message('[EPOCH{}][Training] loss : {}'.format(epoch+1,loss))
            continue
        hit_training = 0
        hit_validation = 0
        eval_training_pbar = tqdm(eval_trainset_generator)
        eval_validation_pbar = tqdm(eval_validationset_generator)
        for x, y in eval_training_pbar:
            # To CUDA tensors
            x = torch.squeeze(x)
            x = x.cuda().float()
            y -= 1

            # Network output
            output, features = net(x)

            # Comment only when AlexNet returns only one val
            # output = output[0]

            if ten_crop is True:
                prediction = torch.mean(output, dim=0)
                prediction = prediction.cpu().detach().numpy()

                if np.argmax(prediction) == y:
                    hit_training += 1
            else:
                _, prediction = torch.max(output, 1)
                prediction = prediction.cpu().detach().numpy()
                hit_training += (prediction == y.numpy()).sum()


        for x, y in eval_validation_pbar:
            # To CUDA tensors
            x = torch.squeeze(x)
            x = x.cuda().float()
            y -= 1

            # Network output
            output, _ = net(x)

            # Comment only when AlexNet returns only one val
            # output = output[0]

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
        logger.message('Epoch : {}, training loss : {}'.format(epoch + 1, loss))
        logger.message('    Training   set accuracy : {0:.2f}%, for {1:}/{2:}'
              .format(acc_training*100, hit_training, num_training))
        logger.message('    Validation set accuracy : {0:.2f}%, for {1:}/{2:}\n'
              .format(acc_validation*100, hit_validation, num_validation))
        # if acc_validation < 1. :
            # logger.error('This combination seems not working, stop training')
            # exit(1)
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
    save,
    shared
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
    gtloss_high = AverageMeter()
    gtloss_small = AverageMeter()


    temperature2 = 5

    for epoch in range(epochs):
        loss= 0.
        if not vgg_gap:
            decay_lr(optimizer, epoch, init_lr, lr_decay)
        else:
            decay_lr_vgg(optimizer, epoch, init_lr, lr_decay)
        net.train()
        kdloss.reset()
        gtloss.reset()
        gtloss_high.reset()
        gtloss_small.reset()
        # for x, x_low, y in training_generator:
        for x, x_low, x_small, y in training_generator:
            # To CUDA tensors
            x = x.cuda().float()
            x_low = x_low.cuda().float()
            x_small = x_small.cuda().float()
            y = y.cuda() - 1

            # teacher = teacher_net(x)
            teacher, t_feature = teacher_net(x)
            # t_feature = t_feature.detach()

            optimizer.zero_grad()

            # Network output
            student, s_feature = net(x_low)


            # Calculate Region KD between CAM region of teacher & student

            KD_loss = nn.KLDivLoss()(F.log_softmax(student / temperature, dim=1),
                                     F.softmax(teacher / temperature, dim=1))

            KD_loss = torch.mul(KD_loss, temperature * temperature)

            GT_loss = lossfunction(student, y)


            if epoch < lr_decay :
                # comment if you only train LR images
                student_high, s_features_high = net(x)
                GT_loss_high = lossfunction(student_high, y)
                student_small, _ = net(x_small)
                GT_loss_small = lossfunction(student_small, y)
                gtloss_high.update(GT_loss_high.item(), x_low.size(0))
                gtloss_small.update(GT_loss_small.item(), x_low.size(0))
            else :
                GT_loss_high = 0
                GT_loss_small = 0
                gtloss_high.update(0, x_low.size(0))
                gtloss_small.update(0, x_low.size(0))
            # loss = KD_loss + GT_loss - BH_loss
            loss = KD_loss + GT_loss + GT_loss_high + GT_loss_small
            # loss = KD_loss + GT_loss
            if isNaN(loss.item()) is True:
                logger.error("This combination failed due to the NaN|inf loss value")
                exit(1)

            kdloss.update(KD_loss.item(), x_low.size(0))
            gtloss.update(GT_loss.item(), x_low.size(0))

            loss.backward()
            optimizer.step()
        net.eval()

        if (epoch + 1) % 10 > 0 :
            # print('Epoch : {}, training loss : {}'.format(epoch + 1, loss))
            logger.debug('[EPOCH{}][Training][KD loss : {:.3f}][GT_loss : {:.3f}][GT_loss_high : {:.3f}][GT_loss_small : {:.3f}]'
                         .format(epoch+1,kdloss.avg, gtloss.avg, gtloss_high.avg,gtloss_small.avg))
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
                                          'GT_high':gtloss_high.avg,
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
        logger.debug('[EPOCH{}][Training][KD loss : {:.3f}][GT_loss : {:.3f}][GT_loss_high : {:.3f}][GT_loss_small : {:.3f}]'
                         .format(epoch+1,kdloss.avg, gtloss.avg, gtloss_high.avg,gtloss_small.avg))
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

def getGcam(feature, grad):
    w = F.adaptive_avg_pool2d(grad, 1)
    # print 'weight shape after avgpool {}'.format(w.shape)
    # gcam = torch.sum(grad, dim = (2,3), keepdim = True)
    gcam = torch.mul(w, feature)
    gcam = torch.sum(gcam, dim = 1)

    gcam = torch.clamp(gcam, min = 0.0)
    for gc in gcam:
        gc -= gc.min()
        gc /= gc.max()

    # gcam[gcam<0.5] = 0.
    return gcam


def write_gradcam(gcam, image, writer,epoch, mode = 't'):
    """
        gcam = gradcam of 3 images
        image = 3 images
        writer = summarywriter
        epoch = epochs
    """
    # write three images to tensorboard
    raw_imgs = []
    gcams = []
    h,w = (227,227)

    for im, gc in zip(image, gcam):
        gc = gc.detach().cpu().numpy()
        im = im.detach().permute(1,2,0).cpu().numpy()
        im += np.array([123.68, 116.779, 103.939])
        im[im < 0] = 0
        im[im > 255.] = 255.
        gc = cv2.resize(gc, (w, h))
        gc = gc * 255.0
        gc = cv2.applyColorMap(np.uint8(gc), cv2.COLORMAP_JET)
        gc = gc.astype(np.float) + im.astype(np.float)
        gc = (gc / gc.max()) * 255.0
        b, g, r = cv2.split(im)
        im = cv2.merge([r,g,b])
        raw_imgs.append(torch.from_numpy(im))
        b, g, r = cv2.split(gc)
        gc = cv2.merge([r,g,b])
        gcams.append(torch.from_numpy(gc))
    gcams = [gc.permute(2,0,1) for gc in gcams]
    gcams = torch.stack(gcams, dim=0)
    gcams = vutils.make_grid(gcams, normalize=True, scale_each=True)
    raw_imgs = [im.permute(2,0,1) for im in raw_imgs]
    raw_imgs = torch.stack(raw_imgs, dim=0)
    raw_imgs = vutils.make_grid(raw_imgs, normalize=True, scale_each=True)
    if mode == 't':
        writer.add_image('raw_imgs', raw_imgs, epoch + 1)
        writer.add_image('t_Gradcams', gcams, epoch + 1)
    elif mode == 's' :
        writer.add_image('s_Gradcams', gcams, epoch + 1)
    else:
        writer.add_image('raw_imgs', raw_imgs, epoch + 1)
        writer.add_image('sr_gradcams', gcams, epoch + 1)


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
    description
    ):
    global glb_grad_at
    writer = SummaryWriter('_'.join(('runs/',datetime.datetime.now().strftime('%Y-%m-%d'), description)))
    glb_grad_at = OrderedDict()

    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    max_accuracy = 0.

    kdloss = AverageMeter()
    gtloss = AverageMeter()
    convloss = AverageMeter()

    if low_ratio != 0:
        modelName = '/Student_LOW_{}x{}_'.format(str(low_ratio), str(low_ratio))
    else:
        print('are you serious ...?')

    teacher_net.eval()

    for epoch in range(epochs):
        show = False
        loss= 0.
        decay_lr(optimizer, epoch, init_lr, lr_decay)
        kdloss.reset()
        gtloss.reset()
        convloss.reset()
        net.train()
        for x, x_low, y in training_generator:
            # To CUDA tensors
            x = x.cuda().float()
            x_low = x_low.cuda().float()

            y = y.cuda() - 1

            optimizer.zero_grad()
            net.zero_grad()

            teacher, t_features = teacher_net(x)

            t_conv5 = t_features['conv5']

            # Network output
            student, s_features = net(x_low)

            s_conv5 = s_features['conv5']

            KD_loss = nn.KLDivLoss()(F.log_softmax(student / temperature, dim=1),
                                  F.softmax(teacher / temperature, dim=1))    # teacher's hook is called in every loss.backward()
                                  # F.softmax(teacher.detach() / temperature, dim=1))

            KD_loss = torch.mul(KD_loss, temperature * temperature)

            student_high, s_features_high = net(x)

            if epoch < lr_decay:
                GT_loss = ce_loss(student, y) + ce_loss(student_high, y)
            else:
                GT_loss = ce_loss(student, y)

            featureMSE = mse_loss(s_conv5, t_conv5)

            loss = KD_loss + GT_loss + featureMSE
            kdloss.update(KD_loss.item(), x_low.size(0))
            gtloss.update(GT_loss.item(), x_low.size(0))
            convloss.update(featureMSE.item(), x_low.size(0))

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
            logger.debug('[EPOCH{}][Training][KD loss : {:.3f}][GT_loss : {:.3f}][Feature_loss : {:.3f}]'
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
        logger.debug('[EPOCH{}][Training][KD loss : {:.3f}][GT_loss : {:.3f}][Feature_loss : {:.3f}]'
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

    # teacher_net.conv5.register_backward_hook(save_grad_at)
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
        adjust_learning_rate(optimizer, epoch)
        net.train()
        gtloss.reset()
        srloss.reset()
        kdloss.reset()
        for x, x_low, y, path in training_generator:
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

            # one_hot_y = torch.zeros(output.shape).float().cuda()
            # for i in range(output.shape[0]):
                # one_hot_y[i][y[i]] = 1.0

            # teacher.backward(gradient = one_hot_y, retain_graph = True)


            # SR_loss = attendedFeature_loss(sr_image, x, attention_weight, mse_loss, at_ratio, glb_grad_at[id(net.srLayer)])
            # SR_loss = attendedFeature_loss(s_features['conv5'], t_features['conv5'].detach(), 1, mse_loss, 2, glb_grad_at[id(teacher_net.conv5)])
            # SR_loss = mse_loss(s_features['conv5'], t_features['conv5'])

            # SR_loss.backward(gradient=one_hot_y)
            GT_loss = ce_loss(output, y)
            KD_loss = nn.KLDivLoss()(F.log_softmax(output / 3, dim=1),
                                     F.softmax(teacher / 3, dim=1))    # teacher's hook is called in every loss.backward()
            KD_loss = torch.mul(KD_loss, 9)

            # loss = GT_loss + KD_loss + SR_loss
            SR_loss = 0.
            loss = GT_loss + KD_loss
            optimizer.zero_grad()
            loss.backward()

            gtloss.update(GT_loss.item(), x_low.size(0))
            # srloss.update(SR_loss.item(), x_low.size(0))
            srloss.update(0, x_low.size(0))
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
            x_low = torch.squeeze(x_low)
            x_low = x_low.cuda().float()
            y -= 1
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
