import torch.nn as nn

class VGG_gap(nn.Module):
    def __init__(self, vgg, num_classes):
        super(VGG_gap, self).__init__()
        self.features = vgg.features
        self.classifier = nn.Sequential(
            nn.Linear(1024, num_classes)
        )

        self.features = nn.Sequential(*list(self.features.children())[:-1] +
                                      [nn.Conv2d(512, 1024, 3, 1, 1)])

        self.gap = nn.AvgPool2d(14)

        self.classifier[-1].weight.data.normal_(0, 0.01)
        self.classifier[-1].bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        conv_feature = x
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x, conv_feature