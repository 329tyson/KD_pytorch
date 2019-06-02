from tensorboardX import SummaryWriter
import torchvision.utils as vutils

class Plotter(object):
    def __init__(self, path, test):
        if test is True:
            return
        self.writer = SummaryWriter(path)
        self.upto   = 3
        self.test   = test

    def plot(self, losses, epoch):
        if self.test is True:
            return
        scalars = {}
        for loss in losses:
            scalars[loss.getname()] = loss.avg
        self.writer.add_scalars('losses', scalars, epoch + 1)

    def plot_image(self, images, title, epoch):
        if self.test is True:
            return
        to_show = images[:self.upto]
        to_show = vutils.make_grid(to_show, normalize=True, scale_each=True)
        self.writer.add_image(title, to_show, epoch + 1)

    def plot_acc(self, accs, epoch):
        if self.test is True:
            return
        self.writer.add_scalars('accs', accs, epoch + 1)

