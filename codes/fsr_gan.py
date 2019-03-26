import torch.nn as nn

class FSR_Discriminator(nn.Module):

    """
    Architecture of feature discriminative network
    (kernel_size, stride, channel, output_size)
    conv1 : 5x5, 2, 8, 32x32x8
    conv2 : 5x5, 2, 16, 16x16x16
    conv3 : 3x3, 2, 32, 8x8x32
    conv4 : 3x3, 1, 64, 8x8x64
    linear
    """

    def __init__(self):
        super(FSR_Discriminator, self).__init__()

        self.conv1 = self.init_layer(nn.Conv2d(1, 8, kernel_size=5, stride=2, padding=2))
        self.conv2 = self.init_layer(nn.Conv2d(8, 16, kernel_size=5, stride=2, padding=2))
        self.conv3 = self.init_layer(nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1))
        self.conv4 = self.init_layer(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1))
        self.linear = self.init_layer(nn.Linear(64 * 8 * 8, 1))
        # self.conv1 = nn.Conv2d(1, 8, kernel_size=5, stride=2, padding=2)
        # self.conv2 = nn.Conv2d(8, 16, kernel_size=5, stride=2, padding=2)
        # self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        # self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # self.linear = nn.Linear(64 * 8 * 8, 1)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        # Assume : x.shape = [bn, 4096]
        x = x.view(x.size(0), 1, 64, 64)
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x

    def init_layer(self, layer):
        layer.weight.data.normal_(mean=0.0, std=0.001)
        layer.bias.data.fill_(0)
        return layer


class FSR_Generator(nn.Module):
    def __init__(self):
        super(FSR_Generator, self).__init__()

        self.conv1 = nn.Conv2d(1, 4, kernel_size=8, stride=1, padding=3)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)
        self.conv5 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.conv6 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)

        self.dropout = nn.Dropout(0.7, True)
        self.linear = nn.Linear(in_features=64*128, out_features=4096)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        # Assume: x.shape = [bn, 4096]
        x = x.view(x.size(0), 1, 64, 64)
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.conv5(x)
        x = self.leaky_relu(x)
        x = self.conv6(x)
        # x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.leaky_relu(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = self.leaky_relu(x)

        return x
