import torch
import torch.nn as nn
import numpy as np

import cv2

from torch.nn import functional as F

from collections import OrderedDict

global glb_grad_teacher # dictionary saving gradient of intermediate feature
global glb_feature
global img_index

# TODO: How about batch_size > 1 ???

def save_grad_at(module, grad_in, grad_out):
    global glb_grad_teacher
    global img_index

    # Assume : batch size 1
    # grad_out[0].shape = (bn, c, h,w)
    # FIXME: absolute value? clamp(relu)?
    grad_at = torch.sum(grad_out[0].detach(), dim=1)
    # grad_at = torch.clamp(grad_at, min=0.0)

    glb_grad_teacher[id(module)][img_index] = grad_at[0].cpu().numpy()


def save_grad(module, grad_in, grad_out):
    global glb_gard_teacher
    global img_index

    glb_grad_teacher[id(module)][img_index] = grad_out[0].detach()


def save_feature(module, input, output):
    global glb_feature

    glb_feature[id(module)][img_index] = output.detach()


def compute_gradCAM(feature, grad):
    # TODO: normalization is needed?
    # normalized_grad = grad / (torch.sqrt(torch.mean(torch.pow(grad, 2))) + 1e-5)
    # weight = F.adaptive_avg_pool2d(normalized_grad, 1)

    weight = F.adaptive_avg_pool2d(grad, 1)

    gradCAM = (feature[0] * weight[0]).sum(dim=0)
    gradCAM = torch.clamp(gradCAM, min=0.0)

    gradCAM -= gradCAM.min()
    gradCAM /= gradCAM.max()

    return gradCAM.cpu().numpy()


def remove_hook(handlers):
    for handle in handlers:
        handle.remove()


def write_gradient(filename, data):
    data -= data.min()
    data /= data.max()
    data *= 255.0
    cv2.imwrite(filename, np.uint8(data))


def write_gradcam(filename, gcam, raw_image_path, image):
    h, w, _ = image.shape

    gcam = cv2.resize(gcam, (w, h))
    # gcam = cv2.resize(gcam, (227,227))
    gcam = cv2.applyColorMap(np.uint8(gcam * 255.0), cv2.COLORMAP_JET)
    gcam = gcam.astype(np.float) + image.astype(np.float)
    gcam = gcam / gcam.max() * 255.0
    cv2.imwrite(filename, np.uint8(gcam))

def calculate_grad(
    net,
    training_generator) :
    global glb_grad_teacher
    global img_index

    # initialize gradient dictionary
    glb_grad_teacher = OrderedDict()
    # FIXME: use for loop to cover all conv
    glb_grad_teacher[id(net.conv1)] = OrderedDict()
    glb_grad_teacher[id(net.conv2)] = OrderedDict()
    glb_grad_teacher[id(net.conv3)] = OrderedDict()
    glb_grad_teacher[id(net.conv4)] = OrderedDict()
    glb_grad_teacher[id(net.conv5)] = OrderedDict()

    img_index = 0

    ce_loss = nn.CrossEntropyLoss()

    # net.train()
    net.eval()

    handlers = []
    handlers.append( net.conv1.register_backward_hook(save_grad_at) )
    handlers.append( net.conv2.register_backward_hook(save_grad_at) )
    handlers.append( net.conv3.register_backward_hook(save_grad_at) )
    handlers.append( net.conv4.register_backward_hook(save_grad_at) )
    handlers.append( net.conv5.register_backward_hook(save_grad_at) )

    for x, _, y, paths in training_generator:
        x = x.cuda().float()
        y = y.cuda() - 1

        one_hot_y = torch.zeros((1, 200)).float().cuda()
        one_hot_y[0][y] = 1.0

        filename = paths[0].split("/")[-1].split(".")[0]
        species = paths[0].split("/")[-2]

        net.zero_grad()

        t, t_features = net(x)

        # GT_loss = ce_loss(t,y)
        # GT_loss.backward()

        t.backward(gradient=one_hot_y)

        write_gradient("img_results/grad/{}_conv1.png".format(filename), glb_grad_teacher[id(net.conv1)][img_index])
        write_gradient("img_results/grad/{}_conv2.png".format(filename), glb_grad_teacher[id(net.conv2)][img_index])
        write_gradient("img_results/grad/{}_conv3.png".format(filename), glb_grad_teacher[id(net.conv3)][img_index])
        write_gradient("img_results/grad/{}_conv4.png".format(filename), glb_grad_teacher[id(net.conv4)][img_index])
        write_gradient("img_results/grad/{}_conv5.png".format(filename), glb_grad_teacher[id(net.conv5)][img_index])

        # To save all gradient of training images, uncomment following lines
        img_index += 1
        if img_index > 10:
            break

    # net.eval()

    remove_hook(handlers)

    return


def calculate_gradCAM(
    net,
    training_generator) :
    global glb_feature
    global glb_grad_teacher
    global img_index

    glb_grad_teacher = OrderedDict()
    glb_grad_teacher[id(net.conv1)] = OrderedDict()
    glb_grad_teacher[id(net.conv2)] = OrderedDict()
    glb_grad_teacher[id(net.conv3)] = OrderedDict()
    glb_grad_teacher[id(net.conv4)] = OrderedDict()
    glb_grad_teacher[id(net.conv5)] = OrderedDict()

    glb_feature = OrderedDict()
    glb_feature[id(net.conv1)] = OrderedDict()
    glb_feature[id(net.conv2)] = OrderedDict()
    glb_feature[id(net.conv3)] = OrderedDict()
    glb_feature[id(net.conv4)] = OrderedDict()
    glb_feature[id(net.conv5)] = OrderedDict()

    img_index = 0

    ce_loss = nn.CrossEntropyLoss()

    # net.train()
    net.eval()

    handlers = []
    handlers.append( net.conv1.register_forward_hook(save_feature))
    handlers.append( net.conv2.register_forward_hook(save_feature))
    handlers.append( net.conv3.register_forward_hook(save_feature))
    handlers.append( net.conv4.register_forward_hook(save_feature))
    handlers.append( net.conv5.register_forward_hook(save_feature))

    handlers.append( net.conv1.register_backward_hook(save_grad))
    handlers.append( net.conv2.register_backward_hook(save_grad))
    handlers.append( net.conv3.register_backward_hook(save_grad))
    handlers.append( net.conv4.register_backward_hook(save_grad))
    handlers.append( net.conv5.register_backward_hook(save_grad))

    for x, _, y,path in training_generator:
        x = x.cuda().float()
        y = y.cuda() - 1

        one_hot_y = torch.zeros((1, 200)).float().cuda()
        one_hot_y[0][y] = 1.0

        filename = path[0].split("/")[-1].split(".")[0]

        # recover raw image from tensor value
        raw_image = x[0].permute((1,2,0)).cpu().numpy()
        raw_image += np.array([123.68, 116.779, 103.939])
        raw_image[raw_image < 0] = 0
        raw_image[raw_image > 255.] = 255.
        raw_image = np.uint8(raw_image)

        net.zero_grad()
        t, t_features = net(x)

        # GT_loss = ce_loss(t, y)
        # GT_loss.backward()

        t.backward(gradient=one_hot_y)

        conv1_gcam = compute_gradCAM(glb_feature[id(net.conv1)][img_index], glb_grad_teacher[id(net.conv1)][img_index])
        conv2_gcam = compute_gradCAM(glb_feature[id(net.conv2)][img_index], glb_grad_teacher[id(net.conv2)][img_index])
        conv3_gcam = compute_gradCAM(glb_feature[id(net.conv3)][img_index], glb_grad_teacher[id(net.conv3)][img_index])
        conv4_gcam = compute_gradCAM(glb_feature[id(net.conv4)][img_index], glb_grad_teacher[id(net.conv4)][img_index])
        conv5_gcam = compute_gradCAM(glb_feature[id(net.conv5)][img_index], glb_grad_teacher[id(net.conv5)][img_index])

        write_gradcam("img_results/gcam/{}_conv1.png".format(filename), conv1_gcam, path[0], raw_image)
        write_gradcam("img_results/gcam/{}_conv2.png".format(filename), conv2_gcam, path[0], raw_image)
        write_gradcam("img_results/gcam/{}_conv3.png".format(filename), conv3_gcam, path[0], raw_image)
        write_gradcam("img_results/gcam/{}_conv4.png".format(filename), conv4_gcam, path[0], raw_image)
        write_gradcam("img_results/gcam/{}_conv5.png".format(filename), conv5_gcam, path[0], raw_image)

        # To save all grad_cam of training images, uncomment following lines
        img_index += 1
        # if img_index > 10:
            # break

    # net.eval()

    remove_hook(handlers)

    return


