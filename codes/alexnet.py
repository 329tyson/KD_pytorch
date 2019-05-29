import torch
import torch.nn as nn
import numpy as np
import os
import time
import datetime
import logging

from gumbelmodule import GumbelSoftmax

class SRLayer(nn.Module):
    def __init__(self):
        super(SRLayer, self).__init__()
        # f1 = 9x9, 64
        # f2 = 5x5, 32
        # f3 = 5x5, 3
        # element-wise addition
        self.sconv1 = self.init_layer(nn.Conv2d(3, 64, kernel_size=9, padding=4))
        self.sconv2 = self.init_layer(nn.Conv2d(64, 32, kernel_size=5, padding=2))
        self.sconv3 = self.init_layer(nn.Conv2d(32, 3, kernel_size=5, padding=2))

        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

        print('\n==============================Network Structure=================================\n')
        print('[SR Layer]')
        print('sconv1', self.sconv1)
        print('relu1', self.relu1)
        print('sconv2', self.sconv2)
        print('relu2', self.relu2)
        print('sconv3', self.sconv3)
        print('\n')

    def forward(self, x):
        residual = x

        x = self.sconv1(x)
        x = self.relu1(x)

        x = self.sconv2(x)
        x = self.relu2(x)

        x = self.sconv3(x)
        x = torch.add(residual, x)

        return x

    def init_layer(self, layer):
        layer.weight.data.normal_(mean=0.0, std=0.001)
        layer.bias.data.fill_(0)
        return layer

class ResidualAdapter(nn.Module):
    def __init__(self, in_planes, inter_planes, out_planes, kernel_size=1, stride=1, padding=0, groups=1):
        super(ResidualAdapter, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=True, groups=groups)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, padding=1, bias=True, groups=groups)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x


class AttentionModule(nn.Module):
    def __init__(self, in_channels, kernel_size=7, stride=1, padding=3):
        super(AttentionModule, self).__init__()
        self.conv = self.init_layer(nn.Conv2d(2, 2, kernel_size=kernel_size, stride=stride, padding=padding))

    def init_layer(self, net):
        nn.init.xavier_uniform_(net.weight)
        nn.init.constant_(net.bias, 0.0)

        return net

    def forward(self, x):
        # first, channel pooling using Max & mean
        x = torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)
        x = self.conv(x)
        return x 


class conv1x1(nn.Module):
    def __init__(self, planes, inter_planes, out_planes, kernel_size=1, is_bn=0, stride=1, padding=0, groups=1):
        super(conv1x1, self).__init__()
        # self.conv = nn.Conv2d(planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)
        self.conv1 = nn.Conv2d(planes, inter_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=True, groups=groups)
        if kernel_size == 3:
            self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=kernel_size, padding=1, bias=True, groups=groups)
        else:
            self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=kernel_size, padding=0, bias=True, groups=groups)
        self.is_bn = is_bn
        self.relu = nn.ReLU(inplace=True)

        if is_bn:
            self.bn = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        if self.is_bn:
            x = self.bn(x)
        return x


class AlexNet(nn.Module):
    def __init__(self, keep_prob, num_classes, skip_layer, save_layer=None,
                 weights_path='DEFAULT', residual_layer=None, is_bn=0):
        super(AlexNet, self).__init__()
        # Parse input arguments into class variables
        self.NUM_CLASSES = num_classes
        self.KEEP_PROB = keep_prob
        self.SKIP_LAYER = skip_layer
        self.SAVE_LAYER = save_layer
        self.var_dict = {}

        # self.weights_dict = np.load(self.WEIGHTS_PATH, encoding='latin1').item()
        self.load = True
        self.residuals = [0, 0, 0, 0, 0]
        self.residual_layer_str = residual_layer

        self.gs = GumbelSoftmax()
        self.gs.cuda()
        self.temperature = 1

        # print(self.weights_dict.keys())

        self.create_network()
        if residual_layer:
            self.create_residual(residual_layer, is_bn)

    def forward(self, x):
        conv1 = self.conv1(x)
        if self.residuals[0]:
            at = self.at1(x)
            mask1 = self.gs(at, temp=self.temperature, force_hard=True)

            # res1 = self.res_adapter1(x[:, :, 4:-4, 4:-4])
            res1 = self.res_adapter1(x)
            res1 = res1 * mask1[:, 1, :, :].unsqueeze(1).expand(res1.shape)

            # res1 = self.res_adapter1(x[:, :, 5:-5, 5:-5]) # 1x1 filter case

            x = conv1 + res1
            # conv1 = x # uncomment when mse(t, s)
        else:
            x = conv1
        x = self.relu1(x)
        x = self.norm1(x)
        x = self.pool1(x)

        conv2 = self.conv2(x)
        if self.residuals[1]:
            at = self.at2(x)
            mask2 = self.gs(at, temp=self.temperature, force_hard=True)

            res2 = self.res_adapter2(x)

            res2 = res2 * mask2[:,1,:,:].unsqueeze(1).expand(res2.shape)

            x = conv2 + res2
            # conv2 = x
        else:
            x = conv2
        x = self.relu2(x)
        x = self.norm2(x)
        x = self.pool2(x)

        conv3 = self.conv3(x)
        if self.residuals[2]:
            at = self.at3(x)
            mask3 = self.gs(at, temp=self.temperature, force_hard=True)
            res3 = self.res_adapter3(x)

            res3 = res3 * mask3[:,1,:,:].unsqueeze(1).expand(res3.shape)
            x = conv3 + res3
            # conv3 = x
        else:
            x = conv3
        x = self.relu3(x)

        conv4 = self.conv4(x)
        if self.residuals[3]:
            at = self.at4(x)
            mask4 = self.gs(at, temp=self.temperature, force_hard=True)
            res4 = self.res_adapter4(x)

            res4 = res4 * mask4[:,1,:,:].unsqueeze(1).expand(res4.shape)
            x = conv4 + res4
            # conv4 = x
        else:
            x = conv4
        x = self.relu4(x)

        conv5 = self.conv5(x)
        if self.residuals[4]:
            at = self.at5(x)
            mask5 = self.gs(at, temp=self.temperature, force_hard=True)
            res5 = self.res_adapter5(x)

            res5 = res5 * mask5[:,1,:,:].unsqueeze(1).expand(res5.shape)
            x = conv5 + res5
            # conv5 = x
        else:
            x = conv5
        x = self.relu5(x)
        x = self.pool5(x)

        x = x.view(x.size(0), 256 * 6 * 6)

        x = self.fc6(x)
        x = self.relu6(x)
        x = self.dropout6(x)

        x = self.fc7(x)
        fc7 = x
        x = self.relu7(x)
        x = self.dropout7(x)

        x = self.fc8(x)

        feature = {}
        if self.SAVE_LAYER:
            if str(1) in self.SAVE_LAYER:
                feature['conv1'] = conv1
                if self.residuals[0]:
                    feature['res1'] = res1
                    feature['mask1'] = mask1
            if str(2) in self.SAVE_LAYER:
                feature['conv2'] = conv2
                if self.residuals[1]:
                    feature['res2'] = res2
                    feature['mask2'] = mask2
            if str(3) in self.SAVE_LAYER:
                feature['conv3'] = conv3
                if self.residuals[2]:
                    feature['res3'] = res3
                    feature['mask3'] = mask3
            if str(4) in self.SAVE_LAYER:
                feature['conv4'] = conv4
                if self.residuals[3]:
                    feature['res4'] = res4
                    feature['mask4'] = mask4
            if str(5) in self.SAVE_LAYER:
                feature['conv5'] = conv5
                if self.residuals[4]:
                    feature['res5'] = res5
                    feature['mask5'] = mask5
            if str(7) in self.SAVE_LAYER:
                feature['fc7'] = fc7

        return x, feature

    def create_network(self):
        self.conv1 = self.init_layer('conv1', nn.Conv2d(3, 96, kernel_size=11, stride=4))
        self.relu1 = nn.ReLU(inplace=True)
        # Tensorflow LRN
        self.norm1 = nn.LocalResponseNorm(size=4, alpha=2e-05, beta=0.75, k=1.0)
        # Caffe LRN
        # self.norm1 = nn.LocalResponseNorm(size=5, alpha=1e-04, beta=0.75, k=1.0)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = self.init_layer('conv2', nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2))
        self.relu2 = nn.ReLU(inplace=True)
        # Tensorflow LRN
        self.norm2 = nn.LocalResponseNorm(size=4, alpha=2e-05, beta=0.75, k=1.0)
        # Caffe LRN
        # self.norm2 = nn.LocalResponseNorm(size=5, alpha=1e-04, beta=0.75, k=1.0)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = self.init_layer('conv3', nn.Conv2d(256, 384, kernel_size=3, padding=1))
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = self.init_layer('conv4', nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2))
        self.relu4 = nn.ReLU(inplace=True)

        self.conv5 = self.init_layer('conv5',nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2))
        self.relu5 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.fc6 = self.init_layer('fc6', nn.Linear(256 * 6 * 6, 4096))
        self.relu6 = nn.ReLU(inplace=True)
        self.dropout6 = nn.Dropout(p=self.KEEP_PROB)

        self.fc7 = self.init_layer('fc7', nn.Linear(4096,4096))
        self.relu7 = nn.ReLU(inplace=True)
        self.dropout7 = nn.Dropout(p=self.KEEP_PROB)

        self.fc8 = self.init_layer('fc8', nn.Linear(4096, self.NUM_CLASSES))

        # for multi gpu training, uncomment below
        # self.relu1 = nn.DataParallel(self.relu1)
        # self.norm1 = nn.DataParallel(self.norm1)
        # self.pool1 = nn.DataParallel(self.pool1)

        # self.relu2 = nn.DataParallel(self.relu2)
        # self.norm2 = nn.DataParallel(self.norm2)
        # self.pool2 = nn.DataParallel(self.pool2)

        # self.relu3 = nn.DataParallel(self.relu3)
        # self.relu4 = nn.DataParallel(self.relu4)

        # self.relu5 = nn.DataParallel(self.relu5)
        # self.pool5 = nn.DataParallel(self.pool5)

        # self.relu6 = nn.DataParallel(self.relu6)
        # self.dropout6 = nn.DataParallel(self.dropout6)

        # self.relu7 = nn.DataParallel(self.relu7)
        # self.dropout7 = nn.DataParallel(self.dropout7)

        print('[AlexNet]')
        print('pool1', self.pool1)
        print('pool2', self.pool2)
        print('conv3', self.conv3)
        print('conv4', self.conv4)
        print('pool5', self.pool5)
        print('fc6', self.fc6)
        print('fc7', self.fc7)
        print('fc8', self.fc8)
        print('\n')

    def create_residual(self, residual_layer, is_bn):
        if str(1) in residual_layer:
            # self.res_adapter1 = conv1x1(3, 96, is_bn, stride=4)
            # self.res_adapter1 = conv1x1(3, 96, 96, 3, is_bn, stride=4)
            self.res_adapter1 = ResidualAdapter(3, 96, 96, 11, stride=4)
            self.residuals[0] = 1

            # self.at1 = self.init_layer('at1', nn.Conv2d(3, 2, kernel_size=11, stride=4))
            self.at1 = AttentionModule(3, kernel_size=11, stride=4, padding=0)

        if str(2) in residual_layer:
            # self.res_adapter2 = conv1x1(96, 256, 256, 3, is_bn, padding=1, groups=2)
            self.res_adapter2 = ResidualAdapter(96, 96, 256, 5, padding=2, groups=2)
            self.residuals[1] = 1

            # sel.fat2 = self.init_layer('at2', nn.Conv2d(96, 2, kernel_size=5, padding=2))
            self.at2 = AttentionModule(96)

        if str(3) in residual_layer:
            # self.res_adapter3 = conv1x1(256, 384, 384, 1, is_bn)
            self.res_adapter3 = ResidualAdapter(256, 256, 384, 3, padding=1)
            self.residuals[2] = 1

            # self.at3 = self.init_layer('at3', nn.Conv2d(256, 2, kernel_size=3, padding=1))
            self.at3 = AttentionModule(256, kernel_size=5, padding=2)

        if str(4) in residual_layer:
            # self.res_adapter4 = conv1x1(384, 384, 384, 1, is_bn, groups=2)
            self.res_adapter4 = ResidualAdapter(384, 384, 384, 3, padding=1, groups=2)
            self.residuals[3] = 1

            # self.at4 = self.init_layer('at4', nn.Conv2d(384, 2, kernel_size=3, padding=1))
            self.at4 = AttentionModule(384, kernel_size=5, padding=2)

        if str(5) in residual_layer:
            # self.res_adapter5 = conv1x1(384, 384, 256, 1, is_bn, groups=2)
            self.res_adapter5 = ResidualAdapter(384, 256, 256, 3, padding=1, groups=2)
            self.residuals[4] = 1

            self.at5 = AttentionModule(384, kernel_size=5, padding=2)

    def init_layer(self, name, net):
        nn.init.xavier_uniform_(net.weight)
        nn.init.constant_(net.bias, 0.0)

        # for multi gpu training, uncomment below
        # net = nn.DataParallel(net)

        return net

    def get_all_residual_adapter_params(self):
        b = []

        if self.residuals[0]:
            b.append(self.res_adapter1)
        if self.residuals[1]:
            b.append(self.res_adapter2)
        if self.residuals[2]:
            b.append(self.res_adapter3)
        if self.residuals[3]:
            b.append(self.res_adapter4)
        if self.residuals[4]:
            b.append(self.res_adapter5)

        for i in range(len(b)):
            for j in b[i].modules():
                for k in j.parameters():
                    if k.requires_grad:
                        yield k

    def get_all_at_layer_params(self):
        b = []

        if self.residuals[0]:
            b.append(self.at1)
        if self.residuals[1]:
            b.append(self.at2)
        if self.residuals[2]:
            b.append(self.at3)
        if self.residuals[3]:
            b.append(self.at4)
        if self.residuals[4]:
            b.append(self.at5)

        for i in range(len(b)):
            for j in b[i].modules():
                for k in j.parameters():
                    if k.requires_grad:
                        yield k


class RACNN(nn.Module):
    def __init__(self, keep_prob, num_classes, skip_layer,
                alex_weights_path=None, sr_weights_path=None, from_npy=False):
        super(RACNN, self).__init__()

        self.srLayer = SRLayer()
        self.classificationLayer = AlexNet(keep_prob, num_classes, skip_layer)

        # load 3 SRLayer
        if sr_weights_path:
            sr_weights = torch.load(sr_weights_path)
            self.srLayer.load_state_dict(sr_weights)
            print 'SR load successful'

        # load alexnet model
        if alex_weights_path:
            if from_npy:
                pretrained = np.load(alex_weights_path, encoding='latin1').item()
                converted = self.classificationLayer.state_dict()
                for lname, val in pretrained.items():
                    if 'conv' in lname:
                        converted[lname + ".weight"] = torch.from_numpy(val[0].transpose(3, 2, 0, 1))
                        converted[lname + ".bias"] = torch.from_numpy(val[1])
                    elif 'fc8' in lname:
                        continue
                    elif 'fc' in lname:
                        converted[lname + ".weight"] = torch.from_numpy(val[0].transpose(1, 0))
                        converted[lname + ".bias"] = torch.from_numpy(val[1])
                self.classificationLayer.load_state_dict(converted, strict=True)

            else:
                alex_weights = torch.load(alex_weights_path)
                self.classificationLayer.load_state_dict(alex_weights)

    def forward(self, x):
        sr_x = self.srLayer(x)
        output, features = self.classificationLayer(sr_x)

        return output, features, sr_x

    def get_all_params_except_last_fc(self):
        b = []

        # b.append(self.srLayer)
        b.append(self.classificationLayer.conv1)
        b.append(self.classificationLayer.conv2)
        b.append(self.classificationLayer.conv3)
        b.append(self.classificationLayer.conv4)
        b.append(self.classificationLayer.conv5)
        b.append(self.classificationLayer.fc6)
        b.append(self.classificationLayer.fc7)

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k
