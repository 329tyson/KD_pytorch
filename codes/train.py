import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as vutils
import datetime
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.autograd import Variable

import cv2

from collections import OrderedDict
global glb_s_grad_at
global glb_c_grad_at
global glb_elem_grad_at
global glb_grad

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

class HingeLoss(nn.Module):
    def __init__(self, margin):
        super(HingeLoss, self).__init__()
        self.margin = margin

    def forward(self, input, target):
        hinge_loss = self.margin - torch.mul(input, target)
        hinge_loss[hinge_loss < 0] = 0
        return torch.mean(hinge_loss)


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
    optimizer.param_groups[-1]['lr'] = lr * 10  # Assume : last element is fc8


def decay_lr_fc8(optimizer, epoch, init_lr, decay_period):
    lr = init_lr * (0.1 ** (epoch // decay_period))
    optimizer.param_groups[0]['lr'] = lr * 10


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


def attention_gram_loss(s_feature, t_feature, at, mse_loss, spatial):
    bn, c, h, w= t_feature.shape

    t_vec = t_feature.view(bn, c, -1)
    s_vec = s_feature.view(bn, c, -1)

    if spatial:
        at = F.normalize(at.view(bn,-1), p=1, dim=1)    # shape: [bn, h*w]
        at = at / torch.max(at, dim=1)[0].unsqueeze(1)

        at = at.unsqueeze(1)
        at = at.expand(-1, c, -1) # shape : [bn, c, h*w]
        # print at.shape, torch.min(at).data.cpu(), torch.mean(at).data.cpu(), torch.max(at).data.cpu()

        t_vec = t_vec * at
        s_vec = s_vec * at

    t_gram = torch.bmm(t_vec, t_vec.permute((0,2,1)))
    s_gram = torch.bmm(s_vec, s_vec.permute((0,2,1)))

    t_gram = t_gram.div(h*w*c)
    s_gram = s_gram.div(h*w*c)

    loss = mse_loss(s_gram, t_gram)
    return loss


def calculate_s_at(t_feature, s_feature, grad_at):
    bn, _, h, w = t_feature.shape
    # t_at = torch.mean(torch.abs(t_feature), dim=1).view(bn, -1) # abs is meaningless, if relu is applied 
    # t_at = torch.sqrt(t_at)
    # t_at = t_at / torch.max(t_at, dim=1)[0].unsqueeze(1)
    # print 'teacher at:', torch.min(t_at).data.cpu(), torch.mean(t_at).data.cpu(), torch.max(t_at).data.cpu()

    # s_at = torch.mean(torch.abs(s_feature.detach()), dim=1).view(bn, -1)
    # s_at = torch.sqrt(s_at)
    # s_at = s_at / torch.max(s_at, dim=1)[0].unsqueeze(1)
    # print 'student at:', torch.min(s_at).data.cpu(), torch.mean(s_at).data.cpu(), torch.max(s_at).data.cpu()

    # r_at = torch.clamp(t_at - s_at, min=0.0).view(bn, h, w)
    # r_at = torch.sqrt(r_at)
    # at = r_at
    # print "r_at: ", torch.min(at).data.cpu(), torch.max(at).data.cpu(), torch.mean(at).data.cpu()
    
    # at = grad_at * t_at.view(bn, h, w) # torch.sqrt(grad_at * at.view(bn, h, w))
    # print 'at:', torch.min(at).data.cpu(), torch.mean(at).data.cpu(), torch.max(at).data.cpu()
    # at = F.adaptive_avg_pool2d(at.view(bn, 1, h, w), 5)
    # at = F.adaptive_avg_pool2d(at, h).view(bn, h, w)
    # print 'at:', torch.min(at).data.cpu(), torch.mean(at).data.cpu(), torch.max(at).data.cpu()
    
    # at = (at * t_at).view(bn, h, w)
    # at = torch.sqrt(at)

    at = grad_at

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


def Feature_cs_at_loss(s_feature, t_feature, loss_fn, c_at, s_at, channel, spatial):
    bn, c, h, w = s_feature.shape

    if spatial:
        s_at = F.normalize(s_at.view(bn,-1), p=1, dim=1)    # shape: [bn, h*w]
        s_at = s_at / torch.max(s_at, dim=1)[0].unsqueeze(1)
        s_at = s_at.view(bn, h, w)
        # print 's_at after norm : ', torch.min(s_at).data.cpu(), torch.mean(s_at.view(bn, -1)).data.cpu(), torch.max(s_at).data.cpu()
    if channel:
        c_at = F.normalize(c_at, p=1, dim=1)    # shape: [bn, c]
        c_at = c_at / torch.max(c_at, dim=1)[0].unsqueeze(1)
        # print 'c_at after norm : ', torch.min(c_at).data.cpu(), torch.mean(c_at.view(bn, -1)).data.cpu(), torch.max(c_at).data.cpu()

    loss = loss_fn(s_feature, t_feature)
    loss = loss.view(bn, c, -1)

    if channel:
        c_at = c_at.unsqueeze(2) # c_at = [bn, c, 1]
        loss = c_at * loss
    loss = torch.mean(loss, dim=1)
    loss = loss.view(bn, h, w)
    
    if spatial:
        loss = s_at * loss
    loss = torch.mean(loss)

    return loss

def Feature_elem_at_loss(s_feature, t_feature, loss_fn, at):
    bn, c, h, w = s_feature.shape

    print 'elem at : ', torch.min(at).data.cpu(), torch.mean(at).data.cpu(), torch.max(at).data.cpu()
    at = at.view(bn, -1)
    at = at / torch.max(at, dim=1)[0].unsqueeze(1)
    at = at.view(bn, c, h, w)
    print 'elem at after norm: ', torch.min(at).data.cpu(), torch.mean(at).data.cpu(), torch.max(at).data.cpu()

    loss = loss_fn(s_feature, t_feature)
    loss = loss * at
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
"""
    # Channel Attention
    grad_at = torch.mean(torch.abs(grad_out[0].detach()).view(bn, c, -1), dim=2)
    # grad_at = torch.abs(torch.mean(grad_out[0].detach().view(bn, c, -1), dim=2))

    glb_c_grad_at[id(module)] = grad_at
"""

"""
def save_grad_at(module, grad_in, grad_out):
    global glb_elem_grad_at
    bn, c, h, w = grad_out[0].shape

    glb_elem_grad_at[id(module)] = torch.abs(grad_out[0].detach())
"""

def save_grad(module, grad_in, grad_out):
    global glb_grad
    glb_grad[id(module)] = grad_out[0].detach()


def reset_zero_grad(module, grad_in, grad_out):
    # Assume : this hook function is registered in Conv filter
    # So, grad_in[0] : input jacobian, grad_in[1] : filter gradient, grad_in[2] : bias gradient
    if grad_in[0] is not None:
        for i in range(len(grad_in[0])):
            grad_in[0][i] = 0.0

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
    save,
    is_writer
    ):
    lossfunction = nn.CrossEntropyLoss()
    max_accuracy = 0.0

    if low_ratio != 0:
        model_name = 'Teacher_LOW_{}x{}_'.format(str(low_ratio), str(low_ratio)) + '_lr:' \
                     + str(init_lr) + '_decay:' + str(lr_decay)
    else:
        model_name = 'Teacher_HIGH' + '_lr:' + str(init_lr) + '_decay:' + str(lr_decay)

    if any(net.residuals):
        model_name = model_name + '_resAdapter' + str(net.residual_layer_str)

    writer = SummaryWriter('_'.join(('runs/' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'), model_name)))
    model_name = '/' + model_name
    print('modelName = ', result_path + model_name)

    for epoch in range(epochs):
        loss = 0.
        loss_value = 0

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


            loss = lossfunction(output, y)
            loss.backward()
            optimizer.step()

            loss_value += loss.data.cpu()

        loss_value /= len(training_generator)
        writer.add_scalars('losses', {'GT_loss': loss_value, }, epoch + 1)

        net.eval()

        if (epoch + 1) % 10:
            logger.debug('[EPOCH{}][Training] loss : {}'.format(epoch+1,loss))

        else:   # Test only 10, 20, 30... epochs
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

            for x, y in eval_validationset_generator:
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
            logger.debug('Epoch : {}, training loss : {}'.format(epoch + 1, loss))
            logger.debug('    Training   set accuracy : {0:.2f}%, for {1:}/{2:}'
                         .format(acc_training*100, hit_training, num_training))
            logger.debug('    Validation set accuracy : {0:.2f}%, for {1:}/{2:}\n'
                         .format(acc_validation*100, hit_validation, num_validation))

            if max_accuracy <= acc_validation:
                max_accuracy = acc_validation

                if save:
                    torch.save(net.state_dict(),
                               result_path + model_name + '_epoch' + str(epoch + 1) + '_acc' + str(round(acc_validation * 100, 4)) + '.pt')

            writer.add_scalars('accuracy', {'training_acc': acc_training, 'val_acc': acc_validation, }, epoch + 1)

    logger.debug('Finished Training\n')
    logger.debug('MAX_ACCURACY : {:.2f}'.format(max_accuracy * 100))


def training_adapter(
    teacher_net,
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
    weight,
    at_ratio,
    vgg_gap,
    save,
    is_writer,
    adapter_features
    ):
    global glb_grad
    global glb_s_grad_at
    glb_grad = OrderedDict()
    glb_s_grad_at = OrderedDict()

    lossfunction = nn.CrossEntropyLoss()
    hinge = HingeLoss(at_ratio)
    mse_loss = nn.MSELoss()
    max_accuracy = 0.0

    if low_ratio != 0:
        model_name = 'Teacher_LOW_{}x{}_'.format(str(low_ratio), str(low_ratio)) + '_lr:' \
                     + str(init_lr) + '_decay:' + str(lr_decay)
    else:
        model_name = 'Teacher_HIGH' + '_lr:' + str(init_lr) + '_decay:' + str(lr_decay)

    if any(net.residuals):
        model_name = model_name + '_resAdapter' + str(net.residual_layer_str)

    if is_writer:
        writer = SummaryWriter('_'.join(('runs/' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'), model_name)))
    model_name = '/' + model_name
    print('modelName = ', result_path + model_name)

    celoss = AverageMeter()
    adapterloss = AverageMeter()

    if str(1) in adapter_features:
        ratio1 = AverageMeter()
    if str(2) in adapter_features:
        ratio2 = AverageMeter()
    if str(3) in adapter_features:
        ratio3 = AverageMeter()
    if str(4) in adapter_features:
        ratio4 = AverageMeter()
    if str(5) in adapter_features:
        ratio5 = AverageMeter()

    teacher_net.eval()

    """
    if str(1) in adapter_features:
        net.conv1.register_backward_hook(save_grad)
        net.res_adapter1.register_backward_hook(save_grad)
    if str(2) in adapter_features:
        net.res_adapter2.register_backward_hook(save_grad)
    if str(3) in adapter_features:
        net.res_adapter3.register_backward_hook(save_grad)
    if str(4) in adapter_features:
        net.res_adapter4.register_backward_hook(save_grad)
    if str(5) in adapter_features:
        net.res_adapter5.register_backward_hook(save_grad)
    # net.fc6.register_backward_hook(save_grad)
    # net.fc7.register_backward_hook(save_grad)
    # net.fc8.register_backward_hook(save_grad)
    """

    if str(1) in adapter_features:
        net.res_adapter1.register_backward_hook(reset_zero_grad)
        net.at1.register_backward_hook(reset_zero_grad)
    if str(2) in adapter_features:
        net.res_adapter2.register_backward_hook(reset_zero_grad)
        net.at2.register_backward_hook(reset_zero_grad)
    if str(3) in adapter_features:
        net.res_adapter3.register_backward_hook(reset_zero_grad)
        net.at3.register_backward_hook(reset_zero_grad)
    if str(4) in adapter_features:
        net.res_adapter4.register_backward_hook(reset_zero_grad)
        net.at4.register_backward_hook(reset_zero_grad)
    if str(5) in adapter_features:
        net.res_adapter5.register_backward_hook(reset_zero_grad)
        net.at5.register_backward_hook(reset_zero_grad)

    relu = nn.ReLU()

    for epoch in range(epochs):

        celoss.reset()
        adapterloss.reset()

        if str(1) in adapter_features:
            ratio1.reset()
        if str(2) in adapter_features:
            ratio2.reset()
        if str(3) in adapter_features:
            ratio3.reset()
        if str(4) in adapter_features:
            ratio4.reset()
        if str(5) in adapter_features:
            ratio5.reset()

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

            # Calculate gradient && Backpropagate
            optimizer.zero_grad()

            # Network output
            output, s_feature = net(x_low)
            _, t_feature = teacher_net(x)
            """
            teacher, t_feature = teacher_net(x)

            one_hot_y = torch.zeros(teacher.shape).float().cuda ()
            for i in range(teacher.shape[0]):
                one_hot_y[i][y[i]] = 1.0

            teacher_net.zero_grad()
            teacher.backward(gradient=one_hot_y, retain_graph=True)
            """

            adapter_loss = 0.0
            act_loss = 0.0

            if str(1) in adapter_features:
                s_conv1 = s_feature['conv1']
                t_conv1 = t_feature['conv1'].detach()
                s_res1 = s_feature['res1']
                mask1 = s_feature['mask1'][:,1,:,:].unsqueeze(1)

                acts_ratio1 = torch.mean(mask1.view(mask1.shape[0], -1), dim=1)
                # act_loss += torch.mean(torch.pow(0.5 - acts_ratio1, 2))
                act_loss += hinge(acts_ratio1, 1)

                mask1 = mask1.detach()
                adapter_loss += mse_loss(mask1 * t_conv1, relu(mask1 * s_conv1 + s_res1))

                # adapter_loss += mse_loss(s_conv1, t_conv1)
                ratio1.update(torch.mean(acts_ratio1).item(), x_low.size(0))

                """
                s_conv1 = s_feature['conv1'].detach()
                t_conv1 = t_feature['conv1'].detach()
                s_res1 = s_feature['res1']
                adapter_loss += mse_loss(t_conv1, relu(s_res1 + s_conv1))
                """
            if str(2) in adapter_features:
                s_conv2 = s_feature['conv2']
                t_conv2 = t_feature['conv2'].detach()
                s_res2 = s_feature['res2']
                mask2 = s_feature['mask2'][:,1,:,:].unsqueeze(1)

                acts_ratio2 = torch.mean(mask2.view(mask2.shape[0], -1), dim=1)
                # act_loss += torch.mean(torch.pow(0.5 - acts_ratio2, 2))
                act_loss += hinge(acts_ratio2, 1)

                mask2 = mask2.detach()
                adapter_loss += mse_loss(mask2 * t_conv2, relu(mask2 * s_conv2 + s_res2))

                # adapter_loss += mse_loss(s_conv2, t_conv2)
                ratio2.update(torch.mean(acts_ratio2).item(), x_low.size(0))

                """
                s_conv2 = s_feature['conv2'].detach()
                t_conv2 = t_feature['conv2'].detach()
                s_res2 = s_feature['res2']
                adapter_loss += mse_loss(t_conv2, relu(s_res2 + s_conv2))
                """
            if str(3) in adapter_features:
                s_conv3 = s_feature['conv3']
                t_conv3 = t_feature['conv3'].detach()
                s_res3 = s_feature['res3']
                mask3 = s_feature['mask3'][:,1,:,:].unsqueeze(1)

                acts_ratio3 = torch.mean(mask3.view(mask3.shape[0], -1), dim=1)
                # act_loss += torch.mean(torch.pow(0.5 - acts_ratio3, 2))
                act_loss += hinge(acts_ratio3, 1)

                mask3 = mask3.detach()
                adapter_loss += mse_loss(mask3 * t_conv3, relu(mask3 * s_conv3 + s_res3))
                # adapter_loss += mse_loss(s_conv3, t_conv3)
                ratio3.update(torch.mean(acts_ratio3).item(), x_low.size(0))

                """
                s_conv3 = s_feature['conv3'].detach()
                t_conv3 = t_feature['conv3'].detach()
                s_res3 = s_feature['res3']
                adapter_loss += mse_loss(t_conv3, relu(s_res3 + s_conv3))
                """
            if str(4) in adapter_features:
                s_conv4 = s_feature['conv4']
                t_conv4 = t_feature['conv4'].detach()
                s_res4 = s_feature['res4']
                mask4 = s_feature['mask4'][:,1,:,:].unsqueeze(1)

                acts_ratio4 = torch.mean(mask4.view(mask4.shape[0], -1), dim=1)
                # act_loss += torch.mean(torch.pow(0.5 - acts_ratio4, 2))
                act_loss += hinge(acts_ratio4, 1)

                mask4 = mask4.detach()
                adapter_loss += mse_loss(mask4 * t_conv4, relu(mask4 * s_conv4 + s_res4))
                # adapter_loss += mse_loss(s_conv4, t_conv4)
                ratio4.update(torch.mean(acts_ratio4).item(), x_low.size(0))

                """
                s_conv4 = s_feature['conv4'].detach()
                t_conv4 = t_feature['conv4'].detach()
                s_res4 = s_feature['res4']
                adapter_loss += mse_loss(t_conv4, relu(s_res4 + s_conv4))
                """
            if str(5) in adapter_features:
                s_conv5 = s_feature['conv5']
                t_conv5 = t_feature['conv5'].detach()
                s_res5 = s_feature['res5']
                mask5 = s_feature['mask5'][:,1,:,:].unsqueeze(1)

                acts_ratio5 = torch.mean(mask5.view(mask5.shape[0], -1), dim=1)
                # act_loss += torch.mean(torch.pow(0.5 - acts_ratio5, 2))
                act_loss += hinge(acts_ratio5, 1)

                mask5 = mask5.detach()
                adapter_loss += mse_loss(mask5 * t_conv5, relu(mask5 * s_conv5 + s_res5))

                # adapter_loss += mse_loss(s_conv5, t_conv5)
                ratio5.update(torch.mean(acts_ratio5).item(), x_low.size(0))

                """
                s_conv5 = s_feature['conv5'].detach()
                t_conv5 = t_feature['conv5'].detach()
                s_res5 = s_feature['res5']
                adapter_loss += mse_loss(t_conv5, relu(s_res5 + s_conv5))
                """

            ce_loss = lossfunction(output, y)
            # FIXME: act_loss weight should be fixed!
            loss = ce_loss + adapter_loss * weight + act_loss * 4

            celoss.update(ce_loss.item(), x_low.size(0))
            adapterloss.update(adapter_loss.item(), x_low.size(0))

            loss.backward()
            optimizer.step()

            """
            print 'loss : ', loss.item()
            print '\n[ activation value ]'
            if str(1) in adapter_features:
                print 't_conv1: ', torch.min(t_feature['conv1']).item(), torch.mean(t_feature['conv1']).item(), torch.max(t_feature['conv1']).item()
                print 's_conv1: ', torch.min(s_conv1).item(), torch.mean(s_conv1).item(), torch.max(s_conv1).item()
                print 'conv1(t - s): ', torch.min(t_conv1 - s_conv1).item(), torch.mean(t_conv1 - s_conv1).item(), torch.max(t_conv1 - s_conv1).item()
            if str(2) in adapter_features:
                print 's_conv2: ', torch.min(s_conv2).item(), torch.mean(s_conv2).item(), torch.max(s_conv2).item()
                print 'conv2(t - s): ', torch.min(t_conv2 - s_conv2).item(), torch.mean(t_conv2 - s_conv2).item(), torch.max(t_conv2 - s_conv2).item()
            if str(3) in adapter_features:
                print 's_conv3: ', torch.min(s_conv3).item(), torch.mean(s_conv3).item(), torch.max(s_conv3).item()
                print 'conv3(t - s): ', torch.min(t_conv3 - s_conv3).item(), torch.mean(t_conv3 - s_conv3).item(), torch.max(t_conv3 - s_conv3).item()
            if str(4) in adapter_features:
                print 's_conv4: ', torch.min(s_conv4).item(), torch.mean(s_conv4).item(), torch.max(s_conv4).item()
                print 'conv4(t - s): ', torch.min(t_conv4 - s_conv4).item(), torch.mean(t_conv4 - s_conv4).item(), torch.max(t_conv4 - s_conv4).item()
            if str(5) in adapter_features:
                print 's_conv5: ', torch.min(s_conv5).item(), torch.mean(s_conv5).item(), torch.max(s_conv5).item()
                print 'conv5(t - s): ', torch.min(t_conv5 - s_conv5).item(), torch.mean(t_conv5 - s_conv5).item(), torch.max(t_conv5 - s_conv5).item()
            fc6 = s_feature['fc6']
            fc7 = s_feature['fc7']
            fc8 = y
            print 'fc6: ', torch.min(fc6).item(), torch.mean(fc6).item(), torch.max(fc6).item()
            print 'fc7: ', torch.min(fc7).item(), torch.mean(fc7).item(), torch.max(fc7).item()
            print 'fc8: ', torch.min(fc8).item(), torch.max(fc8).item()

            print '\n[ gradient value ]'
            if str(1) in adapter_features:
                print 'res1.grad: ', torch.min(glb_grad[id(net.res_adapter1)]).item(), torch.mean(glb_grad[id(net.res_adapter1)]).item(), torch.max(glb_grad[id(net.res_adapter1)]).item()
                print 'conv1.grad: ', torch.min(glb_grad[id(net.conv1)]).item(), torch.mean(glb_grad[id(net.conv1)]).item(), torch.max(glb_grad[id(net.conv1)]).item()
            if str(2) in adapter_features:
                print 'res2.grad: ', torch.min(glb_grad[id(net.res_adapter2)]).item(), torch.mean(glb_grad[id(net.res_adapter2)]).item(), torch.max(glb_grad[id(net.res_adapter2)]).item()
            if str(3) in adapter_features:
                print 'res3.grad: ', torch.min(glb_grad[id(net.res_adapter3)]).item(), torch.mean(glb_grad[id(net.res_adapter3)]).item(), torch.max(glb_grad[id(net.res_adapter3)]).item()
            if str(4) in adapter_features:
                print 'res4.grad: ', torch.min(glb_grad[id(net.res_adapter4)]).item(), torch.mean(glb_grad[id(net.res_adapter4)]).item(), torch.max(glb_grad[id(net.res_adapter4)]).item()
            if str(5) in adapter_features:
                print 'res5.grad: ', torch.min(glb_grad[id(net.res_adapter5)]).item(), torch.mean(glb_grad[id(net.res_adapter5)]).item(), torch.max(glb_grad[id(net.res_adapter5)]).item()

            # print 'fc6.grad: ', torch.min(glb_grad[id(net.fc6)]).item(), torch.mean(glb_grad[id(net.fc6)]).item(), torch.max(glb_grad[id(net.fc6)]).item()
            # print 'fc7.grad: ', torch.min(glb_grad[id(net.fc7)]).item(), torch.mean(glb_grad[id(net.fc7)]).item(), torch.max(glb_grad[id(net.fc7)]).item()
            # print 'fc8.grad: ', torch.min(glb_grad[id(net.fc8)]).item(), torch.max(glb_grad[id(net.fc8)]).item()
            
            print '[ weight value ]'
            if str(1) in adapter_features:
                print 'res1_1.w: ', torch.min(net.res_adapter1.conv1.weight).item(), torch.mean(net.res_adapter1.conv1.weight).item(), torch.max(net.res_adapter1.conv1.weight).item()
                print 'res1_2.w: ', torch.min(net.res_adapter1.conv2.weight).item(), torch.mean(net.res_adapter1.conv2.weight).item(), torch.max(net.res_adapter1.conv2.weight).item()

            if str(2) in adapter_features:
                print 'res2_1.w: ', torch.min(net.res_adapter2.conv1.weight).item(), torch.mean(net.res_adapter2.conv1.weight).item(), torch.max(net.res_adapter2.conv1.weight).item()
                print 'res2_2.w: ', torch.min(net.res_adapter2.conv2.weight).item(), torch.mean(net.res_adapter2.conv2.weight).item(), torch.max(net.res_adapter2.conv2.weight).item()
            
            if str(3) in adapter_features:
                print 'res3_1.w: ', torch.min(net.res_adapter3.conv1.weight).item(), torch.mean(net.res_adapter3.conv1.weight).item(), torch.max(net.res_adapter3.conv1.weight).item()
                print 'res3_2.w: ', torch.min(net.res_adapter3.conv2.weight).item(), torch.mean(net.res_adapter3.conv2.weight).item(), torch.max(net.res_adapter3.conv2.weight).item()
            
            if str(4) in adapter_features:
                print 'res4_1.w: ', torch.min(net.res_adapter4.conv1.weight).item(), torch.mean(net.res_adapter4.conv1.weight).item(), torch.max(net.res_adapter4.conv1.weight).item()
                print 'res4_2.w: ', torch.min(net.res_adapter4.conv2.weight).item(), torch.mean(net.res_adapter4.conv2.weight).item(), torch.max(net.res_adapter4.conv2.weight).item()
            
            if str(5) in adapter_features:
                print 'res5_1.w: ', torch.min(net.res_adapter5.conv1.weight).item(), torch.mean(net.res_adapter5.conv1.weight).item(), torch.max(net.res_adapter5.conv1.weight).item()
                print 'res5_2.w: ', torch.min(net.res_adapter5.conv2.weight).item(), torch.mean(net.res_adapter5.conv2.weight).item(), torch.max(net.res_adapter5.conv2.weight).item()
            
            # print 'fc6.w: ', torch.min(net.fc6.weight).item(), torch.mean(net.fc6.weight).item(), torch.max(net.fc6.weight).item()
            # print 'fc7.w: ', torch.min(net.fc7.weight).item(), torch.mean(net.fc7.weight).item(), torch.max(net.fc7.weight).item()
            # print 'fc8.w: ', torch.min(net.fc8.weight).item(), torch.mean(net.fc8.weight).item(), torch.max(net.fc8.weight).item()
            """
            # print 'at2.w: ', torch.min(net.at2.weight).item(), torch.mean(net.at2.weight).item(), torch.max(net.at2.weight).item()
        
        if is_writer:    
            writer.add_scalars('losses', {'CE_loss': celoss.avg,
                                          'Adapter_loss': adapterloss.avg, }, epoch + 1)

        net.eval()

        if (epoch + 1) % 10:
            ratios = []
            if str(1) in adapter_features:
                ratios.append(round(ratio1.avg, 2))
            if str(2) in adapter_features:
                ratios.append(round(ratio2.avg, 2))
            if str(3) in adapter_features:
                ratios.append(round(ratio3.avg, 2))
            if str(4) in adapter_features:
                ratios.append(round(ratio4.avg, 2))
            if str(5) in adapter_features:
                ratios.append(round(ratio5.avg, 2))

            logger.debug('[EPOCH{}][Training][CE loss : {:.3f}][Adapter loss : {:.3f}][ratios : {}]'
                         .format(epoch+1, celoss.avg, adapterloss.avg, ratios))

        else:   # Test only 10, 20, 30... epochs
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

            for x, y in eval_validationset_generator:
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
            logger.debug('[EPOCH{}][Training][CE loss : {:.3f}][Adapter loss : {:.3f}]'
                         .format(epoch+1, celoss.avg, adapterloss.avg))
            logger.debug('    Training   set accuracy : {0:.2f}%, for {1:}/{2:}'
                         .format(acc_training*100, hit_training, num_training))
            logger.debug('    Validation set accuracy : {0:.2f}%, for {1:}/{2:}\n'
                         .format(acc_validation*100, hit_validation, num_validation))

            if max_accuracy <= acc_validation:
                max_accuracy = acc_validation

                if save:
                    torch.save(net.state_dict(),
                               result_path + model_name + '_epoch' + str(epoch + 1) + '_acc' + str(round(acc_validation * 100, 4)) + '.pt')

            if is_writer:
                writer.add_scalars('accuracy', {'training_acc': acc_training, 'val_acc': acc_validation, }, epoch + 1)

    logger.debug('Finished Training\n')
    logger.debug('MAX_ACCURACY : {:.2f}'.format(max_accuracy * 100))


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
    is_writer
    ):
    lossfunction = nn.CrossEntropyLoss()

    max_accuracy = 0.

    if low_ratio != 0:
        model_name = 'Student_LOW_{}x{}'.format(str(low_ratio), str(low_ratio)) + '_lr:' + str(init_lr) \
                    + '_decay:' + str(lr_decay) + '_T:' + str(temperature)
    else:
        print('are you serious ...?')

    if any(net.residuals):
        model_name = model_name + '_resAdapter' + str(net.residual_layer_str)

    writer = SummaryWriter('_'.join(('runs/' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'), model_name)))
    model_name = '/' + model_name

    teacher_net.eval()
    kdloss = AverageMeter()
    gtloss = AverageMeter()
    convloss = AverageMeter()
    # prev = []
    # bhlosses = []

    for epoch in range(epochs):
        loss= 0.

        kdloss.reset()
        gtloss.reset()
        convloss.reset()

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
                                     F.softmax(teacher.detach() / temperature, dim=1))
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

        writer.add_scalars('losses', {'KD_loss': kdloss.avg,
                                      'GT_loss': gtloss.avg,
                                      'CONV_loss': convloss.avg,
                                      }, epoch + 1)

        if (epoch + 1) % 10:
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

        else:
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

            if max_accuracy <= acc_validation:
                max_accuracy = acc_validation

                if save:
                    torch.save(net.state_dict(), result_path + model_name + '_epoch' + str(epoch + 1) + '_acc' + str(round(acc_validation* 100,4)) + '.pt')

            writer.add_scalars('accuracy', {'training_acc':acc_training, 'val_acc': acc_validation, }, epoch + 1)

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
    is_writer,
    c,
    s
    ):
    global glb_s_grad_at
    global glb_c_grad_at
    glb_s_grad_at = OrderedDict()
    glb_c_grad_at = OrderedDict()

    global glb_elem_grad_at
    glb_elem_grad_at = OrderedDict()

    ce_loss = nn.CrossEntropyLoss()
    # mse_loss = nn.MSELoss()
    mse_loss = nn.MSELoss(reduce=False)
    max_accuracy = 0.

    kdloss = AverageMeter()
    gtloss = AverageMeter()
    convloss = AverageMeter()

    if low_ratio != 0:
        model_name = 'Student_LOW_{}x{}_'.format(str(low_ratio), str(low_ratio)) + '_lr:' + str(lr_decay) \
                    + '_decay' + str(lr_decay) + '_T:' + str(temperature) + '_feat:' + gram_features
    else:
        print('are you serious ...?')

    if any(net.residuals):
        model_name = model_name + '_resAdapter' + str(net.residual_layer_str)

    writer = SummaryWriter('_'.join(('runs/' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'), model_name)))
    model_name = '/' + model_name

    teacher_net.eval()

    """
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
                if str(1) in gram_features:
                    loss.append(mse_loss(s_conv1, t_conv1) * style_weight)
                if str(2) in gram_features:
                    loss.append(mse_loss(s_conv2, t_conv2) * style_weight)
                if str(3) in gram_features:
                    loss.append(mse_loss(s_conv3, t_conv3) * style_weight)
                if str(4) in gram_features:
                    loss.append(mse_loss(s_conv4, t_conv4) * style_weight)
                if str(5) in gram_features:
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
    """

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
        gtloss.reset()
        kdloss.reset()
        convloss.reset()

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

            if str(1) in gram_features: t_conv1 = t_features['conv1'].detach()
            if str(2) in gram_features: t_conv2 = t_features['conv2'].detach()
            if str(3) in gram_features: t_conv3 = t_features['conv3'].detach()
            if str(4) in gram_features: t_conv4 = t_features['conv4'].detach()
            if str(5) in gram_features: t_conv5 = t_features['conv5'].detach()
            if str(7) in gram_features: t_fc7 = t_features['fc7'].detach()

            # # Calculate gradient && Backpropagate
            # optimizer.zero_grad()

            # Network output
            student, s_features = net(x_low)

            net.zero_grad()
            student.backward(gradient=one_hot_y, retain_graph=True)

            # Calculate gradient && Backpropagate
            optimizer.zero_grad()

            if str(1) in gram_features: s_conv1 = s_features['conv1']
            if str(2) in gram_features: s_conv2 = s_features['conv2']
            if str(3) in gram_features: s_conv3 = s_features['conv3']
            if str(4) in gram_features: s_conv4 = s_features['conv4']
            if str(5) in gram_features: s_conv5 = s_features['conv5']
            if str(7) in gram_features: s_fc7 = s_features['fc7']

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

                    # c_at = glb_c_grad_at[id(teacher_net.pool1)]
                    #s_at = calculate_s_at(t_conv1.detach(), s_conv1.detach(), glb_s_grad_at[id(teacher_net.pool1)])
                    # GRAM_loss += Feature_cs_at_loss(s_conv1, t_conv1, mse_loss, c_at, s_at, c, s)

                    at = glb_elem_grad_at[id(teacher_net.pool1)]
                    GRAM_loss += Feature_elem_at_loss(s_conv1, t_conv1, mse_loss, at)

                    # GRAM_loss += attention_gram_loss(s_conv1, t_conv1, s_at, mse_loss, s)
                if str(2) in gram_features:
                    # GRAM_loss += mse_loss(s_conv2, t_conv2)
                    # c_at = glb_c_grad_at[id(teacher_net.pool2)]
                    # s_at = calculate_s_at(t_conv2.detach(), s_conv2.detach(), glb_s_grad_at[id(teacher_net.pool2)])
                    # GRAM_loss += Feature_cs_at_loss(s_conv2, t_conv2, mse_loss, c_at, s_at, c, s)
                    # GRAM_loss += attention_gram_loss(s_conv2, t_conv2, s_at, mse_loss, s)
                    at = glb_elem_grad_at[id(teacher_net.pool2)]
                    GRAM_loss += Feature_elem_at_loss(s_conv2, t_conv2, mse_loss, at)
                if str(3) in gram_features:
                    # GRAM_loss += mse_loss(s_conv3, t_conv3)
                    # c_at = glb_c_grad_at[id(teacher_net.relu3)]
                    # s_at = calculate_s_at(t_conv3.detach(), s_conv3.detach(), glb_s_grad_at[id(teacher_net.relu3)])
                    # GRAM_loss += Feature_cs_at_loss(s_conv3, t_conv3, mse_loss, c_at, s_at, c, s)
                    # GRAM_loss += attention_gram_loss(s_conv3, t_conv3, s_at, mse_loss, s)
                    at = glb_elem_grad_at[id(teacher_net.relu3)]
                    GRAM_loss += Feature_elem_at_loss(s_conv3, t_conv3, mse_loss, at)
                if str(4) in gram_features:
                    # GRAM_loss += mse_loss(s_conv4, t_conv4)
                    # c_at = glb_c_grad_at[id(teacher_net.relu4)]
                    # s_at = calculate_s_at(t_conv4.detach(), s_conv4.detach(), glb_s_grad_at[id(teacher_net.relu4)])
                    # GRAM_loss += Feature_cs_at_loss(s_conv4, t_conv4, mse_loss, c_at, s_at, c, s)
                    # GRAM_loss += attention_gram_loss(s_conv4, t_conv4, s_at, mse_loss, s)
                    at = glb_elem_grad_at[id(teacher_net.relu4)]
                    GRAM_loss += Feature_elem_at_loss(s_conv4, t_conv4, mse_loss, at)
                if str(5) in gram_features:
                    # GRAM_loss += mse_loss(s_conv5, t_conv5)
                    # c_at = glb_c_grad_at[id(teacher_net.pool5)]
                    # s_at = calculate_s_at(t_conv5.detach(), s_conv5.detach(), glb_s_grad_at[id(teacher_net.pool5)])
                    # GRAM_loss += Feature_cs_at_loss(s_conv5, t_conv5, mse_loss, c_at, s_at, c, s)
                    # GRAM_loss += attention_gram_loss(s_conv5, t_conv5, s_at, mse_loss, s)
                    at = glb_elem_grad_at[id(teacher_net.pool5)]
                    GRAM_loss += Feature_elem_at_loss(s_conv5, t_conv5, mse_loss, at)
                if str(7) in gram_features:
                    GRAM_loss += mse_loss(s_fc7, t_fc7)

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

        writer.add_scalars('losses', {'KD_loss': kdloss.avg,
                                      'GT_loss': gtloss.avg,
                                      'CONV_loss': convloss.avg,
                                      }, epoch + 1)

        if (epoch + 1) % 10:
            logger.debug('[EPOCH{}][Training][KD loss : {:.3f}][GT_loss : {:.3f}][MSE_loss : {:.3f}]'
                         .format(epoch+1,kdloss.avg, gtloss.avg, convloss.avg))
        else:   # Test
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

            for x_low, y in eval_validationset_generator:
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
            logger.debug('    Training   set accuracy : {0:.2f}%, for {1:}/{2:}'
                  .format(acc_training*100, hit_training, num_training))
            logger.debug('    Validation set accuracy : {0:.2f}%, for {1:}/{2:}\n'
                  .format(acc_validation*100, hit_validation, num_validation))

            if max_accuracy <= acc_validation:
                max_accuracy = acc_validation

                if save:
                    torch.save(net.state_dict(), result_path + model_name + '_epoch' + str(epoch + 1) + '_acc' +
                               str(round(acc_validation * 100,4)) + '.pt')

            if acc_validation < 0.01:
                logger.error('This combination seems not working, stop training')
                exit(1)

            writer.add_scalars('accuracy', {'training_acc': acc_training, 'val_acc': acc_validation, }, epoch + 1)

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
    is_writer,
    description
    ):
    global glb_grad_at
    glb_grad_at = OrderedDict()

    teacher_net.conv5.register_backward_hook(save_grad_at)

    if low_ratio != 0:
        model_name = 'RACNN_{}x{}_'.format(str(low_ratio), str(low_ratio)) + '_lr:' + str(lr_decay) \
                     + '_decay' + str(lr_decay) + '_T:' + str(temperature) + '_feat:' + gram_features
    else:
        print('are you serious ...?')

    if any(net.residuals):
        model_name = model_name + '_resAdapter' + str(net.residual_layer_str)

    writer = SummaryWriter('_'.join(('runs/' + datetime.datetime.now().strftime('%Y-%m-%d'), model_name)))
    model_name = '/' + model_name

    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    max_accuracy = 0.

    gtloss = AverageMeter()
    srloss = AverageMeter()
    kdloss = AverageMeter()

    for epoch in range(epochs):
        loss= 0.

        gtloss.reset()
        srloss.reset()
        kdloss.reset()

        decay_lr(optimizer, epoch, init_lr, lr_decay)

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
            # sr_image, output = net(x_low)
            # output, s_features = output
            output, s_features, sr_image = net(x_low)

            """
            one_hot_y = torch.zeros(output.shape).float().cuda()
            for i in range(output.shape[0]):
                one_hot_y[i][y[i]] = 1.0

            teacher.backward(gradient = one_hot_y, retain_graph = True)


            # SR_loss = attendedFeature_loss(sr_image, x, attention_weight, mse_loss, at_ratio, glb_grad_at[id(net.srLayer)])
            SR_loss = attendedFeature_loss(s_features['conv5'], t_features['conv5'].detach(), 1, mse_loss, 2, glb_grad_at[id(teacher_net.conv5)])
            """

            SR_loss = 0

            if str(1) in gram_features:
                SR_loss += mse_loss(s_features['conv1'], t_features['conv1'].detach())
            if str(2) in gram_features:
                SR_loss += mse_loss(s_features['conv2'], t_features['conv2'].detach())
            if str(3) in gram_features:
                SR_loss += mse_loss(s_features['conv3'], t_features['conv3'].detach())
            if str(4) in gram_features:
                SR_loss += mse_loss(s_features['conv4'], t_features['conv4'].detach())
            if str(5) in gram_features:
                SR_loss += mse_loss(s_features['conv5'], t_features['conv5'].detach())
            SR_loss *= attention_weight 

            # SR_loss.backward(gradient=one_hot_y)
            GT_loss = ce_loss(output, y)
            KD_loss = nn.KLDivLoss()(F.log_softmax(output / temperature, dim=1),
                                     F.softmax(teacher.detach() / temperature, dim=1))    # teacher's hook is called in every loss.backward()
            KD_loss = torch.mul(KD_loss, temperature * temperature)

            loss = GT_loss + KD_loss + SR_loss
            optimizer.zero_grad()
            loss.backward()

            gtloss.update(GT_loss.item(), x_low.size(0))
            # srloss.update(SR_loss, x_low.size(0))
            srloss.update(SR_loss.item(), x_low.size(0))
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

        if (epoch + 1) % 10:
            logger.debug('[EPOCH{}][Training][GT_LOSS : {:.3f}][RECON_LOSS : {:.3f}][KD_LOSS : {:.3f}]'
                     .format(epoch+1, gtloss.avg, srloss.avg, kdloss.avg))
        else:
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
                # sr_image, output = net(x_low)
                # output, features = output
                output, features, sr_image = net(x_low)

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
                # sr_image, output = net(x_low)
                # output, features = output
                output, features, sr_image = net(x_low)

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

            if max_accuracy <= acc_validation:
                max_accuracy = acc_validation

                if save:
                    torch.save(net.state_dict(), result_path + model_name + '_epoch' + str(epoch + 1) + '_acc' + str(round(acc_validation* 100,4)) + '.pt')

            if acc_validation < 0.01 :
                logger.error('This combination seems not working, stop training')
                exit(1)

            writer.add_scalars('accuracy', {'training_acc': acc_training, 'val_acc': acc_validation, }, epoch + 1)
    logger.debug('Finished Training\n')
    logger.debug('MAX_ACCURACY : {:.2f}'.format(max_accuracy * 100))


def training_FSR(
    net,
    generator,
    discriminator,
    optimizer_G,
    optimizer_D,
    focal_loss_r,
    num_classes,
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
    is_writer
    ):
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss(reduce=False)
    pdist = torch.nn.modules.distance.PairwiseDistance()

    max_accuracy = 0.

    if low_ratio != 0:
        model_name = 'Student_LOW_{}x{}'.format(str(low_ratio), str(low_ratio)) + '_lr:' + str(init_lr) \
                    + '_decay:' + str(lr_decay) + '_r:' + str(focal_loss_r)
    else:
        print('are you serious ...?')

    if any(net.residuals):
        model_name = model_name + '_resAdapter' + str(net.residual_layer_str)

    writer = SummaryWriter('_'.join(('runs/' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'), model_name)))
    model_name = '/' + model_name

    new_fc8 = nn.Linear(4096, num_classes)
    new_fc8.cuda()
    new_fc8.load_state_dict(net.fc8.state_dict())
    new_fc8.train()

    net.eval()

    optimizer = optim.SGD(
        [{'params': new_fc8.parameters(), 'lr': init_lr * 10}],
        momentum=0.9,
        weight_decay=0.0005)

    gtloss = AverageMeter()
    genloss = AverageMeter()
    disloss = AverageMeter()

    for epoch in range(epochs):

        gtloss.reset()
        genloss.reset()
        disloss.reset()

        decay_lr_fc8(optimizer, epoch, init_lr, lr_decay)

        new_fc8.train()
        generator.train()
        discriminator.train()

        for x, x_low, y in training_generator:
            # To CUDA tensors
            x = x.cuda().float()
            x_low = x_low.cuda().float()
            y = y.cuda() - 1

            _, t_feature = net(x)
            high_embedding = t_feature['fc7'].detach()

            _, s_feature = net(x_low)
            low_embedding = s_feature['fc7'].detach()

            optimizer_G.zero_grad()
            optimizer_D.zero_grad()

            ### Train generative network(G)
            # freeze D
            for param in discriminator.parameters():
                param.requires_grad = False

            sr_embedding = generator(low_embedding)

            focal_loss = pdist(sr_embedding, high_embedding) ** focal_loss_r
            loss_G = -torch.mean(discriminator(sr_embedding)) + torch.mean(focal_loss)
            loss_G.backward()

            ### Train discriminative network(D)
            for param in discriminator.parameters():
                param.requires_grad = True

            sr_embedding = sr_embedding.detach()

            loss_D = torch.mean(discriminator(sr_embedding)) - torch.mean(discriminator(high_embedding))
            loss_D.backward()

            optimizer_G.step()
            optimizer_D.step()

            disloss.update(loss_D.item(), x_low.size(0))
            genloss.update(loss_G.item(), x_low.size(0))

            ### Train last fc layer to calculate class logit
            # optimizer.zero_grad()
            #
            # output = new_fc8(sr_embedding)
            # loss = ce_loss(output, y)
            #
            # loss.backward()
            # optimizer.step()
            # gtloss.update(loss.item(), x_low.size(0))

            if isNaN(loss_D.item()) or isNaN(loss_G.item()) is True:
                logger.error("This combination failed due to the NaN|inf loss value")
                exit(1)

            # print (
            #     '[Training][GT_loss : {:.3f}][GEN_loss : {:.3f}][DISC_LOSS : {:.3f}]').format(
            #     gtloss.avg, genloss.avg, disloss.avg)

        generator.eval()
        discriminator.eval()

        writer.add_scalars('losses', {'GEN_loss': genloss.avg,
                                      'GT_loss': gtloss.avg,
                                      'DISC_loss': disloss.avg,
                                      }, epoch + 1)

        if (epoch + 1) % 10:
            logger.debug('[EPOCH{}][Training][GT_loss : {:.3f}][GEN_loss : {:.3f}][DISC_LOSS : {:.3f}]'
                         .format(epoch+1, gtloss.avg, genloss.avg, disloss.avg))

        else:
            new_fc8.train()
            for i in range(10):
                for x, x_low, y in training_generator:
                    # To CUDA tensors
                    x_low = x_low.cuda().float()
                    y = y.cuda() - 1

                    _, s_feature = net(x_low)
                    low_embedding = s_feature['fc7'].detach()
                    sr_embedding = generator(low_embedding).detach()
                    output = new_fc8(sr_embedding)

                    optimizer.zero_grad()
                    loss = ce_loss(output, y)
                    loss.backward()
                    optimizer.step()

                    gtloss.update(loss.item(), x_low.size(0))
                    # print '[Training][GT_loss : {:.3f}]'.format(gtloss.avg)
            new_fc8.eval()

            # Test
            hit_training = 0
            hit_validation = 0
            for x_low, y in eval_trainset_generator:
                # To CUDA tensors
                x_low = torch.squeeze(x_low)
                x_low = x_low.cuda().float()
                y -= 1

                # Network output
                # output, _ = net(x_low)
                _, feature = net(x_low)
                output = new_fc8(generator(feature['fc7']))

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
            for x_low, y in eval_validationset_generator:
                # To CUDA tensors
                x_low = torch.squeeze(x_low)
                x_low = x_low.cuda().float()
                y -= 1

                # Network output
                # output, _ = net(x_low)
                _, feature = net(x_low)
                output = new_fc8(generator(feature['fc7']))

                if ten_crop is True:
                    prediction = torch.mean(output, dim=0)
                    prediction = prediction.cpu().detach().numpy()

                    if np.argmax(prediction) == y:
                        hit_validation += 1
                        if count_success > count_show :
                            continue
                        success.append(x_low[0])
                        count_success += 1
                    elif count_failure < count_show + 1:
                        failure.append(x_low[0])
                        count_failure += 1
                else:
                    _, prediction = torch.max(output, 1)
                    prediction = prediction.cpu().detach().numpy()
                    hit_validation += (prediction == y.numpy()).sum()

            if ten_crop is True:
                torch.stack(success, dim=0)
                torch.stack(failure, dim=0)
                success = vutils.make_grid(success, normalize=True, scale_each=True)
                failure = vutils.make_grid(failure, normalize=True, scale_each=True)

                writer.add_image('Success', success, epoch + 1)
                writer.add_image('Failure', failure, epoch + 1)
            # Trace
            acc_training = float(hit_training) / num_training
            acc_validation = float(hit_validation) / num_validation
            logger.debug('[EPOCH{}][Training][GT_loss : {:.3f}][GEN_loss : {:.3f}][DISC_LOSS : {:.3f}]'
                         .format(epoch+1, gtloss.avg, genloss.avg, disloss.avg))
            logger.debug('    Training   set accuracy : {0:.2f}%, for {1:}/{2:}'
                         .format(acc_training*100, hit_training, num_training))
            logger.debug('    Validation set accuracy : {0:.2f}%, for {1:}/{2:}\n'
                         .format(acc_validation*100, hit_validation, num_validation))

            if max_accuracy <= acc_validation:
                max_accuracy = acc_validation

                if save:
                    torch.save(net.state_dict(), result_path + model_name + '_epoch' + str(epoch + 1) + '_acc' + str(round(acc_validation* 100,4)) + '.pt')
                    #TODO: Save generator model

            writer.add_scalars('accuracy', {'training_acc':acc_training, 'val_acc': acc_validation, }, epoch + 1)

    logger.debug('Finished Training\n')
    logger.debug('MAX_ACCURACY : {:.2f}'.format(max_accuracy * 100))


def training_Disc(
    teacher_net,
    net,
    optimizer,
    discriminator,
    optimizer_D,
    w_clip,
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
    is_writer
    ):
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss(reduce=False)

    max_accuracy = 0.

    if low_ratio != 0:
        model_name = 'Student_LOW_{}x{}'.format(str(low_ratio), str(low_ratio)) + '_lr:' + str(init_lr) \
                    + '_decay:' + str(lr_decay)
    else:
        print('are you serious ...?')

    if any(net.residuals):
        model_name = model_name + '_resAdapter' + str(net.residual_layer_str)

    writer = SummaryWriter('_'.join(('runs/' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'), model_name)))
    model_name = '/' + model_name

    gtloss = AverageMeter()
    disloss = AverageMeter()
    genloss = AverageMeter()

    teacher_net.eval()

    # for x, x_low, y in training_generator:
    #     x = x.cuda().float()
    #     x_low = x_low.cuda().float()
    #     y = y.cuda() - 1
    #     _, t_feature = teacher_net(x)
    #     high_embedding = t_feature['fc7'].detach()
    #     _, s_feature = net(x_low)
    #     low_embedding = s_feature['fc7'].detach()
    #
    #     optimizer_D.zero_grad()
    #     ### Train discriminative network(D)
    #     loss_D = torch.mean(discriminator(low_embedding)) - torch.mean(discriminator(high_embedding))
    #     loss_D.backward()
    #
    #     optimizer.step()
    #     optimizer_D.step()
    #     print loss_D.item()
    #
    #     disloss.update(loss_D.item(), x_low.size(0))
    #     if loss_D.item() < -1:
    #         break

    for epoch in range(epochs):

        gtloss.reset()
        genloss.reset()
        disloss.reset()

        decay_lr(optimizer, epoch, init_lr, lr_decay)

        net.train()
        discriminator.train()

        for x, x_low, y in training_generator:
            # To CUDA tensors
            x = x.cuda().float()
            x_low = x_low.cuda().float()
            y = y.cuda() - 1

            _, t_feature = teacher_net(x)
            high_embedding = t_feature['fc7'].detach()

            output, s_feature = net(x_low)
            low_embedding = s_feature['fc7']

            # freeze D
            for param in discriminator.parameters():
                param.requires_grad = False

            optimizer.zero_grad()

            # feature_loss = torch.mean(torch.sum(mse_loss(low_embedding, high_embedding), dim=1))
            feature_loss = torch.mean(mse_loss(low_embedding, high_embedding))
            good_loss = torch.mean(discriminator(low_embedding))
            ce = ce_loss(output, y)

            loss_G = feature_loss + good_loss + ce
            loss_G.backward()

            genloss.update(loss_G.item(), x_low.size(0))
            gtloss.update(ce.item(), x_low.size(0))

            ### Train discriminative network(D)
            for param in discriminator.parameters():
                param.requires_grad = True

            for p in discriminator.parameters():
                p.data.clamp_(-w_clip, w_clip)

            optimizer_D.zero_grad()

            high_embedding = high_embedding.detach()
            low_embedding = low_embedding.detach()
            loss_D = torch.mean(discriminator(low_embedding)) - torch.mean(discriminator(high_embedding))
            loss_D.backward()

            optimizer.step()
            optimizer_D.step()

            disloss.update(loss_D.item(), x_low.size(0))

            if isNaN(loss_D.item()) or isNaN(loss_G.item()) is True:
                logger.error("This combination failed due to the NaN|inf loss value")
                exit(1)

            # print (
            #     '[Training][GT_loss : {:.3f}][GEN_loss : {:.3f}][DISC_LOSS : {:.3f}]').format(
            #     ce.item(), loss_G.item(), loss_D.item())

        discriminator.eval()
        net.eval()

        writer.add_scalars('losses', {'GEN_loss': genloss.avg,
                                      'GT_loss': gtloss.avg,
                                      'DISC_loss': disloss.avg,
                                      }, epoch + 1)

        if (epoch + 1) % 10:
            logger.debug('[EPOCH{}][Training][GT_loss : {:.3f}][DISC_LOSS : {:.3f}]'
                         .format(epoch+1, gtloss.avg, disloss.avg))

        else:
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
            for x_low, y in eval_validationset_generator:
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
                        if count_success > count_show :
                            continue
                        success.append(x_low[0])
                        count_success += 1
                    elif count_failure < count_show + 1:
                        failure.append(x_low[0])
                        count_failure += 1
                else:
                    _, prediction = torch.max(output, 1)
                    prediction = prediction.cpu().detach().numpy()
                    hit_validation += (prediction == y.numpy()).sum()

            if ten_crop is True:
                torch.stack(success, dim=0)
                torch.stack(failure, dim=0)
                success = vutils.make_grid(success, normalize=True, scale_each=True)
                failure = vutils.make_grid(failure, normalize=True, scale_each=True)

                writer.add_image('Success', success, epoch + 1)
                writer.add_image('Failure', failure, epoch + 1)
            # Trace
            acc_training = float(hit_training) / num_training
            acc_validation = float(hit_validation) / num_validation
            logger.debug('[EPOCH{}][Training][GT_loss : {:.3f}][GEN_loss : {:.3f}][DISC_LOSS : {:.3f}]'
                         .format(epoch+1, gtloss.avg, genloss.avg, disloss.avg))
            logger.debug('    Training   set accuracy : {0:.2f}%, for {1:}/{2:}'
                         .format(acc_training*100, hit_training, num_training))
            logger.debug('    Validation set accuracy : {0:.2f}%, for {1:}/{2:}\n'
                         .format(acc_validation*100, hit_validation, num_validation))
            if max_accuracy <= acc_validation:
                max_accuracy = acc_validation

                if save:
                    torch.save(net.state_dict(), result_path + model_name + '_epoch' + str(epoch + 1) + '_acc' + str(round(acc_validation* 100,4)) + '.pt')
            writer.add_scalars('accuracy', {'training_acc':acc_training, 'val_acc': acc_validation, }, epoch + 1)

    logger.debug('Finished Training\n')
    logger.debug('MAX_ACCURACY : {:.2f}'.format(max_accuracy * 100))

