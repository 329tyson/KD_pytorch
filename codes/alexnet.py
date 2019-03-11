import torch
import torch.nn as nn
import numpy as np
import os
import time
import datetime
import logging



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

class AlexNet(nn.Module):
    def __init__(self, keep_prob, num_classes, skip_layer,
                 weights_path='DEFAULT', res=None):
        super(AlexNet, self).__init__()
        # Parse input arguments into class variables
        self.NUM_CLASSES = num_classes
        self.KEEP_PROB = keep_prob
        self.SKIP_LAYER = skip_layer
        self.var_dict = {}

        # self.weights_dict = np.load(self.WEIGHTS_PATH, encoding='latin1').item()
        self.load=True

        # print(self.weights_dict.keys())

        self.create_network()


    def forward(self, x):
        x = self.conv1(x)
        conv1 = x
        x = self.relu1(x)
        x = self.norm1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        conv2 = x
        x = self.relu2(x)
        x = self.norm2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        conv3 = x
        x = self.relu3(x)

        x = self.conv4(x)
        conv4 = x
        x = self.relu4(x)

        x = self.conv5(x)
        conv5 = x
        x = self.relu5(x)
        x = self.pool5(x)

        x = x.view(x.size(0), 256 * 6 * 6)

        x = self.fc6(x)
        x = self.relu6(x)
        x = self.dropout6(x)

        x = self.fc7(x)
        x = self.relu7(x)
        x = self.dropout7(x)

        x = self.fc8(x)

        feature = {}
        feature['conv1'] = conv1
        feature['conv2'] = conv2
        feature['conv3'] = conv3
        feature['conv4'] = conv4
        feature['conv5'] = conv5

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

    def init_layer(self, name, net):
        nn.init.xavier_uniform_(net.weight)
        nn.init.constant_(net.bias, 0.0)

        # for multi gpu training, uncomment below
        # net = nn.DataParallel(net)

        return net


class RACNN(nn.Module):
    def __init__(self, keep_prob, num_classes, skip_layer,
                 alex_weights_path=None, alex_pretrained=False, from_npy=False,
                 sr_weights_path=None, sr_pretrained=False
                 ):
        super(RACNN, self).__init__()

        self.srLayer = SRLayer()
        self.classificationLayer = AlexNet(keep_prob, num_classes, skip_layer)

        if sr_pretrained:
            sr_weights = torch.load(sr_weights_path)
            self.srLayer.load_state_dict(sr_weights)

        if alex_pretrained:
            alex_weights = torch.load(alex_weights_path)
            self.classificationLayer.load_state_dict(alex_weights)

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

    def forward(self, x):
        sr_x = self.srLayer(x)
        output = self.classificationLayer(sr_x)

        return sr_x, output

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
