import torch
import torch.nn as nn
import numpy as np
import os
import time
import datetime
import logging



class AlexNet(nn.Module):
    def __init__(self, keep_prob, num_classes, skip_layer):
        super(AlexNet, self).__init__()
        # Parse input arguments into class variables
        self.NUM_CLASSES = num_classes
        self.KEEP_PROB   = keep_prob
        self.SKIP_LAYER  = skip_layer
        self.in_channels = [3, 96, 256, 384, 384]
        self.out_channels = [96, 256, 384, 384, 256]
        self.kernel_size= [11, 5, 3, 3 ,3]
        self.padding = [0, 2 ,1 ,1 ,1]
        self.stride = [4, 1, 1, 1, 1]
        self.groups = [1, 2, 1, 2, 2]

        self.create_network()


    def forward(self, x):
        x = self.conv1(x)
        conv1 = x
        x = self.relu1(x)
        x = self.norm1(x)
        x = self.pool1(x)
        # print 'conv shape : {}'.format(x.shape)

        x = self.conv2(x)
        conv2 = x
        x = self.relu2(x)
        x = self.norm2(x)
        x = self.pool2(x)
        # print 'conv shape : {}'.format(x.shape)

        conv3_input = x
        x = self.conv3(x)
        conv3 = x
        x = self.relu3(x)
        # print 'conv shape : {}'.format(x.shape)

        x = self.conv4(x)
        conv4 = x
        x = self.relu4(x)
        # print 'conv shape : {}'.format(x.shape)

        x = self.conv5(x)
        conv5 = x
        x = self.relu5(x)
        x = self.pool5(x)
        # print 'conv shape : {}'.format(x.shape)

        x = x.view(x.size(0), 256 * 6 * 6)

        x = self.fc6(x)
        x = self.relu6(x)
        x = self.dropout6(x)

        x = self.fc7(x)
        x = self.relu7(x)
        x = self.dropout7(x)

        x = self.fc8(x)

        # feature = {}
        # feature['conv1'] = conv1
        # feature['conv2'] = conv2
        # feature['conv3'] = conv3
        # feature['conv4'] = conv4
        # feature['conv5'] = conv5

        feature = []
        feature.append(conv1)
        feature.append(conv2)
        feature.append(conv3)
        feature.append(conv4)
        feature.append(conv5)

        return x, feature

    def create_network(self):
        self.conv1 = self.init_layer('conv1', nn.Conv2d(3, 96, kernel_size=11, stride=4))
        self.relu1 = nn.ReLU(inplace=True)
        self.norm1 = nn.LocalResponseNorm(size=4, alpha=2e-05, beta=0.75, k=1.0)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = self.init_layer('conv2', nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2))
        self.relu2 = nn.ReLU(inplace=True)
        self.norm2 = nn.LocalResponseNorm(size=4, alpha=2e-05, beta=0.75, k=1.0)
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

        return net

    def finetuning_params(self):
        ret = self.get_conv_list()

        for i in range(len(ret)):
            for j in ret[i].modules():
                jj=0
                for k in j.parameters():
                    jj +=1
                    if k.requires_grad:
                        yield k

    def get_conv_list(self):
        ret = []

        ret.append(self.conv1)
        ret.append(self.conv2)
        ret.append(self.conv3)
        ret.append(self.conv4)
        ret.append(self.conv5)

        return ret
