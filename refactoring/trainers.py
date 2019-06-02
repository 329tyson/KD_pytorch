import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from tqdm import tqdm
from logger import AverageMeter

class Trainer(object):
    def __init__(self, **kwargs):
        '''
            Must have following keyword arguments
            kwargs['network']
            kwargs['optimizer']
            kwargs['loaders'] - (train_loader, eval_train_loader, validation_loader)
            kwargs['lr']
            kwargs['logger']

            ALL PARAMS SHOULD BE ONE OBJECT(not container e.g list...)
        '''
        self.net        = kwargs['network']
        self.lr         = kwargs['lr']
        self.optimizer  = kwargs['optimizer']
        self.logger     = kwargs['logger']
        self.writer     = kwargs['writer']
        self.lossfunc   = kwargs['lossfunc']
        self.val_period = kwargs['val_period']
        self.val_train  = False
        self.losses     = []

        self.train_loader      = kwargs['train_loader']
        self.eval_train_loader = kwargs['eval_train_loader']
        self.validation_loader = kwargs['validation_loader']

    def write_epoch(self, epoch, train_acc, valid_acc):
        self.logger.iteration(losses = self.losses, EPOCH = str(epoch + 1), TRAIN_ACC= str(train_acc), VALID_ACC= str(valid_acc))

    def write_iter(self, epoch):
        self.logger.iteration(losses= self.losses, EPOCH = str(epoch + 1))

    def lr_decay(self, epochs, decay_period):
        token = False
        lr = self.lr * (0.1 ** (epochs // decay_period))
        for param_group in self.optimizer.param_groups[:-1]:
            if lr != param_group['lr']:
                token = True
                param_group['lr'] = lr
        self.optimizer.param_groups[-1]['lr'] = lr * 10
        if token is True :
            self.logger.message('Finetune lr : {} FC8 lr : {}'.format(lr, lr * 10))

    def prepX(self, data):
        return data.cuda().float()

    def prepY(self, data):
        data = data.cuda() - 1
        return data

    def cost(self, output, label):
        raise NotImplementedError

    def evaluate(self, output, label):
        pred = torch.mean(output, dim=0)
        pred = pred.cpu().detach().numpy()
        if np.argmax(pred) == label:
            return 1
        return 0

    def output(self, output):
        output, features = output
        return output

    def validate(self, epoch, dataloader):
        self.net.eval()
        hit = 0
        pbar = tqdm(
            enumerate(dataloader),
            desc='Validation [E {}]'.format(epoch + 1),
            bar_format='{desc:<5}[B {n_fmt}][R {rate_fmt}][{postfix[0]} {postfix[1][value]}{postfix[2]}] ',
            postfix=['HIT', dict(value=0), '/' + str(len(dataloader))])

        for i, (x, y)in pbar:
            x = torch.squeeze(x)
            x = self.prepX(x)
            y = self.prepY(y)

            output  = self.output(self.net(x))
            hit    += self.evaluate(output, y)
            pbar.postfix[1]['value'] = hit

        return float(hit) / len(dataloader) * 100

    def getPbar(self, dataloader, epoch):
        pbar = tqdm(
            enumerate(dataloader),
            desc='Training [E {}]'.format(epoch + 1),
            bar_format='{desc:<5}[B {n_fmt}][R {rate_fmt}][{postfix[0]} {postfix[1][value]}] ',
            postfix=['Loss', dict(value=0)],
            leave = False)
        return pbar

    def train(self, epochs, lr_decay):
        gtloss = AverageMeter('GTloss')
        self.losses.append(gtloss)

        for epoch in range(epochs):
            self.net.train()
            gtloss.reset()
            self.lr_decay(epoch, lr_decay)
            pbar = self.getPbar(self.train_loader, epoch)

            for i, (x, y) in pbar:
                x = self.prepX(x)
                y = self.prepY(y)
                # if i == 0 :
                    # self.writer.plot_image(images = x, title = 'HR', epoch = epoch)

                self.optimizer.zero_grad()
                output = self.output(self.net(x))

                loss = self.cost(output, y)
                pbar.postfix[1]['value'] = loss.item()
                gtloss.update(loss.item(), x.size(0))

                loss.backward()
                self.optimizer.step()

            self.write_iter(epoch)
            self.writer.plot(self.losses, epoch)
            if (epoch + 1) % self.val_period > 0:
                continue
            train_acc = self.validate(epoch, self.eval_train_loader) if self.val_train is True else 0.
            valid_acc = self.validate(epoch, self.validation_loader)

            self.writer.plot_acc({'train_acc' : train_acc, 'valid_acc' : valid_acc}, epoch)
            self.write_epoch(epoch, train_acc, valid_acc)
            print '\n'

class SingleResTrainer(Trainer):
    def __init__(self, **kwargs):
        super(SingleResTrainer, self).__init__(**kwargs)

    def cost(self, output, label):
        return self.lossfunc(output, label)

class KDTrainer(SingleResTrainer):
    def __init__(self, **kwargs):
        super(KDTrainer, self).__init__(**kwargs)
        self.teacher_net = kwargs['teacher']
        self.loss_for_kd = kwargs['kdlossfunc']
        self.temperature = kwargs['temperature']

    def KDcost(self, t_output, s_output):
        return self.loss_for_kd(F.log_softmax(s_output / self.temperature, dim=1), F.softmax(t_output / self.temperature, dim=1))

    def train(self, epochs, lr_decay):
        GTloss = AverageMeter('GTloss')
        KDloss = AverageMeter('KDloss')
        self.losses.append(GTloss)
        self.losses.append(KDloss)
        for epoch in range(epochs):
            self.net.train()
            self.lr_decay(epoch, lr_decay)
            GTloss.reset()
            KDloss.reset()
            pbar = self.getPbar(self.train_loader, epoch)

            for i, (x, x_low, y) in pbar:
                x     = self.prepX(x)
                x_low = self.prepX(x_low)
                y     = self.prepY(y)
                # if i == 0 :
                    # self.writer.plot_image(images = x, title = 'HR', epoch = epoch)
                    # self.writer.plot_image(images = x_low, title = 'LR', epoch = epoch)

                self.optimizer.zero_grad()
                t_output = self.output(self.teacher_net(x))
                s_output = self.output(self.net(x_low))

                kdloss = self.KDcost(t_output, s_output)
                kdloss = torch.mul(kdloss, self.temperature * self.temperature)
                gtloss = self.cost(s_output, y)

                KDloss.update(kdloss.item(), x.size(0))
                GTloss.update(gtloss.item(), x.size(0))
                loss = kdloss + gtloss
                pbar.postfix[1]['value'] = loss.item()

                loss.backward()
                self.optimizer.step()
            self.write_iter(epoch)
            self.writer.plot(self.losses, epoch)
            if (epoch + 1) % self.val_period > 0:
                continue
            train_acc = self.validate(epoch, self.eval_train_loader) if self.val_train is True else 0.
            valid_acc = self.validate(epoch, self.validation_loader)
            self.writer.plot_acc({'train_acc' : train_acc, 'valid_acc' : valid_acc}, epoch)
            self.write_epoch(epoch, train_acc, valid_acc)
            print '\n'

class FeatureMSETrainer(KDTrainer):
    def __init__(self, **kwargs):
        super(FeatureMSETrainer, self).__init__(**kwargs)
        self.regression_layers      = kwargs['regression_layers']
        self.loss_for_ft = kwargs['ftlossfunc']
        self.use_grad = False

    def FTcost(self, t_features, s_features):
        loss = 0.
        for i in range(len(t_features)):
            if str(i + 1) in self.regression_layers:
                loss += self.loss_for_ft(s_features[i], t_features[i])
        return loss


    def train(self, epochs, lr_decay):
        self.logger.message('Feature regression on {}'.format(self.regression_layers))
        GTloss = AverageMeter('GTloss')
        KDloss = AverageMeter('KDloss')
        FTloss = AverageMeter('FTloss')
        self.losses.append(GTloss)
        self.losses.append(KDloss)
        self.losses.append(FTloss)
        for epoch in range(epochs):
            self.net.train()
            self.lr_decay(epoch, lr_decay)
            GTloss.reset()
            KDloss.reset()
            FTloss.reset()
            pbar = self.getPbar(self.train_loader, epoch)

            for i, (x, x_low, y) in pbar:
                x     = self.prepX(x)
                x_low = self.prepX(x_low)
                y     = self.prepY(y)
                # if i == 0 :
                    # self.writer.plot_image(images = x, title = 'HR', epoch = epoch)
                    # self.writer.plot_image(images = x_low, title = 'LR', epoch = epoch)

                self.optimizer.zero_grad()
                t_output, t_features = self.teacher_net(x)
                s_output, s_features = self.net(x_low)

                if self.use_grad:
                    one_hot_y = torch.zeros(t_output.shape).float().cuda()
                    for i in range(t_output.shape[0]):
                        one_hot_y[i][y[i]] = 1.0
                    t_output.backward(gradient = one_hot_y, retain_graph = True)

                kdloss = self.KDcost(t_output, s_output)
                kdloss = torch.mul(kdloss, self.temperature * self.temperature)
                gtloss = self.cost(s_output, y)
                ftloss = self.FTcost(t_features, s_features)

                KDloss.update(kdloss.item(), x.size(0))
                GTloss.update(gtloss.item(), x.size(0))
                FTloss.update(ftloss.item(), x.size(0))
                loss = kdloss + gtloss + ftloss
                pbar.postfix[1]['value'] = loss.item()

                loss.backward()
                self.optimizer.step()
            self.write_iter(epoch)
            self.writer.plot(self.losses, epoch)
            if (epoch + 1) % self.val_period > 0:
                continue
            train_acc = self.validate(epoch, self.eval_train_loader) if self.val_train is True else 0.
            valid_acc = self.validate(epoch, self.validation_loader)
            self.writer.plot_acc({'train_acc' : train_acc, 'valid_acc' : valid_acc}, epoch)
            self.write_epoch(epoch, train_acc, valid_acc)
            print '\n'

class GradientMSETrainer(FeatureMSETrainer):
    def __init__(self, **kwargs):
        super(GradientMSETrainer, self).__init__(**kwargs)
        self.gradient_layers = kwargs['gradient_layers']
        self.gradient = {}
        self.use_grad = True

        self.register_hook()

    def save_grad(self, module, grad_in, grad_out):
        grad = grad_out[0].detach()
        grad = torch.abs(grad)
        self.gradient[id(module)] = grad

    def register_hook(self):
        convlist = self.teacher_net.get_conv_list()
        for i in range(len(convlist)):
            if str(i+1) in self.gradient_layers:
                convlist[i].register_backward_hook(self.save_grad)

    def spatial_weighted_loss(self, t_feature, s_feature, grad):
        spatial_grad = torch.mean(grad, dim = 1, keepdim =True)

        loss = torch.sub(t_feature, s_feature)
        loss = torch.mul(loss, loss)
        loss = torch.mul(loss, spatial_grad)
        loss = torch.mean(loss)
        return loss

    def channel_weighted_loss(self, t_feature, s_feature, grad):
        channelwise_grad = torch.mean(grad, dim = (2,3), keepdim=True)

        loss = torch.sub(t_feature, s_feature)
        loss = torch.mul(loss, loss)
        loss = torch.mul(loss, channelwise_grad)
        loss = torch.mean(loss)
        return loss

    def FTcost(self, t_features, s_features):
        loss = 0.
        type = True if 's' in self.gradient_layers else False
        convlist = self.teacher_net.get_conv_list()
        for i in range(len(t_features)):
            if str(i+1) in self.gradient_layers:
                if type is True:
                    loss += self.spatial_weighted_loss(
                        t_feature = t_features[i],
                        s_feature = s_features[i],
                        grad      = self.gradient[id(convlist[i])]
                    )
                else:
                    loss += self.channel_weighted_loss(
                        t_feature = t_features[i],
                        s_feature = s_features[i],
                        grad      = self.gradient[id(convlist[i])]
                    )
            elif str(i+1) in self.regression_layers:
                loss += super(GradientMSETrainer, self).FTcost(t_features, s_features)
        return loss
