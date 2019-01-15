import torch
import torch.nn as nn
import numpy as np
import os
import time
import datetime
import logging

# log settings
logging.basicConfig(format='%(message)s',
                    filename=str(os.path.dirname(os.path.realpath(__file__)) + '/logs/'+ datetime.datetime.now().strftime('Day_%d,_%H:%M')) + '.log',
                    filemode='w',
                    level=logging.INFO)

# decorator for time logging
def timeit(method):
    def timed(*args, **kw):
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        # print('[%s] :: %s executed took %2.2f ms ' % (timestamp, method.__name__ +'(' + ')', (te-ts) * 1000))
        logging.info('[{}][{}] spent {} ms'.format(timestamp, "{:>17}".format(method.__name__ + '()'), "{0:.3f}".format((te-ts)*1000)))
        return result
    return timed

class AlexNet(nn.Module):
    @timeit
    def __init__(self, keep_prob, num_classes, skip_layer, train_mode,
                 weights_path='DEFAULT', res=None):
        super(AlexNet, self).__init__()
        # Parse input arguments into class variables
        self.NUM_CLASSES = num_classes
        self.KEEP_PROB = keep_prob
        self.SKIP_LAYER = skip_layer
        self.res = res
        self.var_dict = {}
        self.train_mode = train_mode

        if weights_path == 'DEFAULT':
            self.WEIGHTS_PATH = 'bvlc_alexnet.npy'
        else:
            self.WEIGHTS_PATH = weights_path

        self.weights_dict = np.load(self.WEIGHTS_PATH, encoding='latin1').item()
        self.load=True

        print(self.weights_dict.keys())

        self.create_network()


    @timeit
    def forward(self, x):
        # x = self.features(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.norm1(x)
        x = self.pool1(x)
        pool1 = x

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.norm2(x)
        x = self.pool2(x)
        pool2 = x

        x = self.conv3(x)
        x = self.relu3(x)
        conv3 = x

        x = self.conv4(x)
        x = self.relu4(x)
        conv4 = x

        x = self.conv5(x)
        x = self.relu5(x)
        x = self.pool5(x)
        pool5 = x

        x = x.view(x.size(0), 256 * 6 * 6)

        x = self.fc6(x)
        x = self.dropout6(x)
        fc6 = x

        x = self.fc7(x)
        x = self.dropout7(x)
        fc7 = x
        # x = self.classifier(x)

        x = self.fc8(x)

        # return pool1, pool2, conv3, conv4, pool5, fc6, fc7, x
        return x
        # return x,fc6, fc7

    @timeit
    def create_network(self):
        self.conv1 = self.init_layer('conv1', nn.Conv2d(3, 96, kernel_size=11, stride=4))
        self.relu1 = nn.ReLU(inplace=True)
        self.norm1 = nn.LocalResponseNorm(size=2, alpha=2e-05, beta=0.75, k=1.0)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = self.init_layer('conv2', nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2))
        self.relu2 = nn.ReLU(inplace=True)
        self.norm2 = nn.LocalResponseNorm(size=2, alpha=2e-05, beta=0.75, k=1.0)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = self.init_layer('conv3', nn.Conv2d(256, 384, kernel_size=3, padding=1))
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = self.init_layer('conv4', nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2))
        self.relu4 = nn.ReLU(inplace=True)

        self.conv5 = self.init_layer('conv5',nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2))
        self.relu5 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.fc6 = self.init_layer('fc6', nn.Linear(256 * 6 * 6, 4096))
        self.dropout6 = nn.Dropout(p=self.KEEP_PROB)

        self.fc7 = self.init_layer('fc7', nn.Linear(4096,4096))
        self.dropout7 = nn.Dropout(p=self.KEEP_PROB)

        self.fc8 = self.init_layer('fc8', nn.Linear(4096, self.NUM_CLASSES))

        # self.features = nn.Sequential(
            # self.conv1,
            # self.relu1,
            # self.norm1,
            # self.pool1,
            # self.conv2,
            # self.relu2,
            # self.norm2,
            # self.pool2,
            # self.conv3,
            # self.relu3,
            # self.conv4,
            # self.relu4,
            # self.conv5,
            # self.relu5,
            # self.pool5,
        # )
        # self.classifier = nn.Sequential(
            # self.fc6,
            # self.dropout6,
            # self.fc7,
            # self.dropout7
        # )
        print('\n==============================Network Structure=================================\n')
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
        params = list(net.parameters())
        # nn.init.constant_(net.weight, 0.0)
        nn.init.xavier_uniform_(net.weight)
        nn.init.constant_(net.bias, 0.0)

        # if name in self.SKIP_LAYER:
            # print('making {} layer from scratch'.format(name))
            # nn.init.xavier_uniform_(net.weight)
            # nn.init.constant_(net.bias, 0.0)
            # # nn.init.xavier_uniform(net.bias)
        # elif 'conv' in name:
            # params[0].data = torch.from_numpy(self.weights_dict[name][0].transpose(3,2,0,1))
            # params[1].data= torch.from_numpy(self.weights_dict[name][1])
            # # print('params[0] : {}. params[1] : {}'.format(params[0].data, params[1].data))
        # else:
            # params[0].data= torch.from_numpy(self.weights_dict[name][0].transpose(1,0))
            # params[1].data= torch.from_numpy(self.weights_dict[name][1])
            # # print('params[0] : {}. params[1] : {}'.format(params[0].data, params[1].data))
            # # print('weight[0] : {}. weight[1] : {}'.format(self.weights_dict[name][0], self.weights_dict[name][1]))

        # self.var_dict[name] = [params[0], params[1]]

        return net
