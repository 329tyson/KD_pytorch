import torch
import torch.nn as nn
import numpy as np

import cv2

from torch.nn import functional as F

from collections import OrderedDict

global glb_spatial_grad # dictionary saving gradient of intermediate feature
global glb_feature
global glb_c_grad
global img_index

# TODO: How about batch_size > 1 ???

def save_grad_at(module, grad_in, grad_out):
    global glb_spatial_grad
    global img_index

    # grad_out[0].shape = (bn, c, h,w)
    # FIXME: absolute value? clamp(relu)?
    # grad_at = torch.abs(torch.sum(grad_out[0].detach(), dim=1))
    # grad_at = torch.sum(grad_out[0].detach(), dim=1)
    grad_at = torch.sum(torch.abs(grad_out[0].detach()), dim=1)
    # grad_at = torch.sum(torch.clamp(grad_out[0].detach(), min=0.0), dim=1)

    glb_spatial_grad[id(module)][img_index] = grad_at


def save_grad(module, grad_in, grad_out):
    global glb_c_grad
    global img_index

    grad_at = F.adaptive_avg_pool2d(grad_out[0].detach(), 1)
    glb_c_grad[id(module)][img_index] = grad_at


def save_feature(module, input, output):
    global glb_feature
    global img_index

    glb_feature[id(module)][img_index] = output.detach()


def compute_gradCAMs(feature, grad):
    bn, c, h, w = feature.shape

    # TODO: normalization is needed?
    # normalized_grad = grad / (torch.sqrt(torch.mean(torch.pow(grad, 2))) + 1e-5)
    # weight = F.adaptive_avg_pool2d(normalized_grad, 1)

    # weight = F.adaptive_avg_pool2d(grad, 1)
    weight = grad

    gradCAM = (feature * weight).sum(dim=1)
    gradCAM = torch.clamp(gradCAM, min=0.0)

    # gradCAM -= gradCAM.min()
    # gradCAM /= gradCAM.max()
    gradCAM = gradCAM.view(bn, -1)
    gradCAM /= torch.max(gradCAM, dim=1)[0].unsqueeze(1)
    gradCAM = gradCAM.view(bn, h, w)

    return gradCAM


def remove_hook(handlers):
    for handle in handlers:
        handle.remove()


def write_gradient(filename, at, image):
    h, w, _ = image.shape
    data = at.data.cpu().numpy()
    # print '[', filename, '] min, max : ', data.min(), data.max()
    print '[ {} ] {:.3f}, {:.3f}, {:.3f}'.format(filename.split('/')[-1], data.min(), data.mean(), data.max())
    data -= data.min()
    data /= data.max()
    data = cv2.resize(data, (w,h), interpolation=cv2.INTER_NEAREST)
    data = cv2.applyColorMap(np.uint8(data * 255.0), cv2.COLORMAP_JET)
    data = data.astype(np.float) + image.astype(np.float)
    data = data / data.max() * 255.0
    cv2.imwrite(filename, np.uint8(data))


def write_gradcam(filename, gcam, raw_image_path, image):
    h, w, _ = image.shape

    gcam = gcam.data.cpu().numpy()
    
    gcam = cv2.resize(gcam, (w, h))
    gcam = cv2.applyColorMap(np.uint8(gcam * 255.0), cv2.COLORMAP_JET)
    gcam = gcam.astype(np.float) + image.astype(np.float)
    gcam = gcam / gcam.max() * 255.0
    cv2.imwrite(filename, np.uint8(gcam))


def calculate_attention(
    teacher_net,
    student_net,
    training_generator,
    filepath,
    epoch):
    global glb_spatial_grad
    global img_index

    # teacher_layer = [teacher_net.pool1, teacher_net.pool2, teacher_net.relu3,
    #                  teacher_net.relu4, teacher_net.pool5]
    # student_layer = [student_net.pool1, student_net.pool2, student_net.relu3,
    #                  student_net.relu4, student_net.pool5]
    #
    # feature_name = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']

    teacher_layer = [teacher_net.conv5]
    student_layer = [student_net.conv5]

    feature_name = ['conv5']

    # initialize gradient dictionary
    glb_spatial_grad = OrderedDict()
    for i in range(len(teacher_layer)):
        glb_spatial_grad[id(teacher_layer[i])] = OrderedDict()
        glb_spatial_grad[id(student_layer[i])] = OrderedDict()

    img_index = 0

    # ce_loss = nn.CrossEntropyLoss()

    # net.train()
    teacher_net.eval()
    student_net.eval()

    handlers = []
    for i in range(len(teacher_layer)):
        handlers.append(teacher_layer[i].register_backward_hook(save_grad_at))
        handlers.append(student_layer[i].register_backward_hook(save_grad_at))

    for x, x_low, y, paths in training_generator:
        x = x.cuda().float()
        x_low = x_low.cuda().float()
        y = y.cuda() - 1

        bn, _, _, _ = x.shape

        one_hot_y = torch.zeros((bn, 200)).float().cuda()

        filenames = []
        raw_imgs = []
        low_raw_imgs = []

        for j in range(bn):
            one_hot_y[j][y[j]] = 1.0

            filename = paths[j].split("/")[-1].split(".")[0]
            filenames.append(filename)
            # species = paths[0].split("/")[-2]

            # recover raw image from tensor value
            raw_image = x[j].permute((1, 2, 0)).cpu().numpy()
            raw_image += np.array([123.68, 116.779, 103.939])
            raw_image[raw_image < 0] = 0
            raw_image[raw_image > 255.] = 255.
            raw_image = np.uint8(raw_image)
            raw_imgs.append(raw_image)

            low_raw_image = x_low[j].permute((1, 2, 0)).cpu().numpy()
            low_raw_image += np.array([123.68, 116.779, 103.939])
            low_raw_image[low_raw_image < 0] = 0
            low_raw_image[low_raw_image > 255.] = 255.
            low_raw_image = np.uint8(low_raw_image)
            low_raw_imgs.append(low_raw_image)

        teacher_net.zero_grad()

        t, t_features = teacher_net(x)

        t.backward(gradient=one_hot_y)

        student_net.zero_grad()

        s, s_features = student_net(x_low)

        s.backward(gradient=one_hot_y)

        for i in range(len(teacher_layer)):
            # file = filepath + filename + '_' + str(epoch) + 'th'

            teacher_at = glb_spatial_grad[id(teacher_layer[i])][img_index]
            student_at = glb_spatial_grad[id(student_layer[i])][img_index]
            residual_at = teacher_at - student_at

            t_activation = (torch.mean(t_features[feature_name[i]], dim=1)).detach()
            s_activation = (torch.mean(s_features[feature_name[i]], dim=1)).detach()
            res_activation = torch.clamp(t_activation - s_activation, min=0.0)
            weighted_activation = res_activation * t_activation

            t_mul = teacher_at * t_activation
            mul = teacher_at * res_activation

            t_act = torch.mean(torch.abs(t_features[feature_name[i]]), dim=1).detach()
            s_act = torch.mean(torch.abs(s_features[feature_name[i]]), dim=1).detach()
            # t_act = torch.sqrt(torch.mean(torch.abs(t_features[feature_name[i]]), dim=1)).detach()
            # s_act = torch.sqrt(torch.mean(torch.abs(s_features[feature_name[i]]), dim=1)).detach()
            # t_act = t_act / np.amax(t_act)
            # s_act = s_act / np.amax(s_act)
            res_act = torch.clamp(t_act - s_act, min=0.0)
            weighted_act = res_act * t_act

            # mul = F.adaptive_avg_pool2d(mul, 5)
            # mul = F.adaptive_avg_pool2d(mul, 6)
            print mul.shape

            for j in range(bn):
                file = filepath + filenames[j] + '_' + str(epoch) + 'th'

                write_gradient(file + "_grad_t_" + feature_name[i] + ".png", teacher_at[j], raw_imgs[j])
                write_gradient(file + "_grad_s_" + feature_name[i] + ".png", student_at[j], low_raw_imgs[j])
                write_gradient(file + "_grad_r_" + feature_name[i] + ".png", residual_at[j], raw_imgs[j])

                write_gradient(file + "_act_t_" + feature_name[i] + ".png", t_activation[j], raw_imgs[j])
                write_gradient(file + "_act_s_" + feature_name[i] + ".png", s_activation[j], low_raw_imgs[j])
                write_gradient(file + "_act_r_" + feature_name[i] + ".png", res_activation[j], low_raw_imgs[j])
                write_gradient(file + "_act_rXt_" + feature_name[i] + ".png", weighted_activation[j], low_raw_imgs[j]) # highlight teacher's high point

                write_gradient(file + "_act_t_actXtgrad_" + feature_name[i] + ".png", t_mul[j], raw_imgs[j])
                write_gradient(file + "_act_r_actXtgrad_" + feature_name[i] + ".png", mul[j], low_raw_imgs[j])

                write_gradient(file + "_abs_act_t_" + feature_name[i] + ".png", t_act[j], raw_imgs[j])
                write_gradient(file + "_abs_act_s_" + feature_name[i] + ".png", s_act[j], low_raw_imgs[j])
                write_gradient(file + "_abs_act_r_" + feature_name[i] + ".png", res_act[j], low_raw_imgs[j])
                write_gradient(file + "_abs_act_rXt_" + feature_name[i] + ".png", weighted_act[j], low_raw_imgs[j])

        # To save all gradient of training images, uncomment following lines
        img_index += 1
        if img_index > 1:
            break

    # net.eval()

    remove_hook(handlers)

    return


def calculate_gradCAM(
    net,
    training_generator,
    filepath,
    epoch ) :
    global glb_feature
    global glb_c_grad
    global img_index

    # layers = [net.pool1, net.pool2, net.relu3, net.relu4, net.pool5]
    layers = [net.conv1, net.conv2, net.conv3, net.conv4, net.conv5]
    feature_name = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']

    # glb_abs_grad = OrderedDict()
    glb_c_grad = OrderedDict()
    glb_feature = OrderedDict()
    for i in layers:
        glb_c_grad[id(i)] = OrderedDict()
        glb_feature[id(i)] = OrderedDict()

    img_index = 0

    ce_loss = nn.CrossEntropyLoss()

    # net.train()
    net.eval()

    handlers = []
    for i in layers:
        handlers.append(i.register_forward_hook(save_feature))
        handlers.append(i.register_backward_hook(save_grad))

    for x, _, y, path in training_generator:
        x = x.cuda().float()
        y = y.cuda() - 1

        bn, _, _, _ = x.shape

        one_hot_y = torch.zeros((bn, 200)).float().cuda()
        filename = []
        raw_imgs = []

        for i in range(bn):
            one_hot_y[i][y[i]] = 1.0
            filename.append(path[i].split("/")[-1].split(".")[0])

            raw_image = x[i].permute((1,2,0)).cpu().numpy()
            raw_image += np.array([123.68, 116.779, 103.939])
            raw_image[raw_image < 0] = 0
            raw_image[raw_image > 255.] = 255.
            raw_image = np.uint8(raw_image)
            raw_imgs.append(raw_image)

        net.zero_grad()
        t, t_features = net(x)

        t.backward(gradient=one_hot_y)

        for i in range(len(layers)):
            gcam = compute_gradCAMs(glb_feature[id(layers[i])][img_index], glb_c_grad[id(layers[i])][img_index])

            for j in range(bn):
                f = filepath + filename[j] + '_' + str(epoch) + 'th_' + feature_name[i] + ".png"
                write_gradcam(f, gcam[j], path[j], raw_imgs[j])

        # To save all grad_cam of training images, uncomment following lines
        img_index += 1
        if img_index > 4:
            break

    # net.eval()
    
    remove_hook(handlers)

    return
    

