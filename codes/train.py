import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
def decay_lr(optimizer, epoch, init_lr, decay_period):
    lr = init_lr * (0.1 ** (epoch // decay_period))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        # optimizer.param_groups[7]['lr'] = lr * 10

def training(
    net,
    optimizer,
    init_lr,
    lr_decay,
    epochs,
    ten_crop,
    training_generator,
    eval_trainset_generator,
    eval_validationset_generator,
    num_training,
    num_validation,
    low_ratio,
    result_path,
    logger):
    lossfunction = nn.CrossEntropyLoss()
    if low_ratio != 0:
        modelName = '/teacher_LOW_{}x{}_'.format(str(low_ratio), str(low_ratio))
    else:
        modelName = '/teacher_HIGH_'
    for epoch in range(epochs):
        loss= 0.
        decay_lr(optimizer, epoch, init_lr, lr_decay)
        net.train()
        for x, y in training_generator:
            # To CUDA tensors
            x = x.cuda().float()
            y = y.cuda() - 1

            # Calculate gradient && Backpropagate
            optimizer.zero_grad()

            # Network output
            output = net(x)

            loss = lossfunction(output, y)
            loss.backward()
            optimizer.step()
        net.eval()
        # Test only 10, 20, 30... epochs
        if (epoch + 1) % 10 > 0 :
            print('Epoch : {}, training loss : {}'.format(epoch + 1, loss))
            logger.debug('[EPOCH{}][Training] loss : {}'.format(epoch+1,loss))
            continue
        hit_training = 0
        hit_validation = 0
        for x, y in eval_trainset_generator:
            # To CUDA tensors
            x = torch.squeeze(x)
            x = x.cuda().float()
            y -= 1

            # Network output
            output= net(x)

            if ten_crop is True:
                prediction = torch.mean(output, dim=0)
                prediction = prediction.cpu().detach().numpy()

                if np.argmax(prediction) == y:
                    hit_training += 1
            else:
                _, prediction = torch.max(output, 1)
                prediction = prediction.cpu().detach().numpy()
                hit_training += (prediction == y.numpy()).sum()


        for x, y in eval_validationset_generator:
            # To CUDA tensors
            x = torch.squeeze(x)
            x = x.cuda().float()
            y -= 1

            # Network output
            output= net(x)

            if ten_crop is True:
                prediction = torch.mean(output, dim=0)
                prediction = prediction.cpu().detach().numpy()

                if np.argmax(prediction) == y:
                    hit_validation += 1
            else:
                _, prediction = torch.max(output, 1)
                prediction = prediction.cpu().detach().numpy()
                hit_validation += (prediction == y.numpy()).sum()

        acc_training = float(hit_training) / num_training
        acc_validation = float(hit_validation) / num_validation
        print('Epoch : {}, training loss : {}'.format(epoch + 1, loss))
        print('    Training   set accuracy : {0:.2f}%, for {1:}/{2:}'
              .format(acc_training*100, hit_training, num_training))
        print('    Validation set accuracy : {0:.2f}%, for {1:}/{2:}\n'
              .format(acc_validation*100, hit_validation, num_validation))
        logger.debug('Epoch : {}, training loss : {}'.format(epoch + 1, loss))
        logger.debug('    Training   set accuracy : {0:.2f}%, for {1:}/{2:}'
              .format(acc_training*100, hit_training, num_training))
        logger.debug('    Validation set accuracy : {0:.2f}%, for {1:}/{2:}\n'
              .format(acc_validation*100, hit_validation, num_validation))
        torch.save(net.state_dict(), result_path + modelName + str(epoch + 1) + '_epoch_acc_' + str(acc_validation*100) +'.pt')


def training_KD(
    teacher_net,
    net,
    optimizer,
    temperature,
    init_lr,
    lr_decay,
    epochs,
    ten_crop,
    training_generator,
    eval_trainset_generator,
    eval_validationset_generator,
    num_training,
    num_validation,
    low_ratio,
    result_path,
    logger):
    lossfunction = nn.CrossEntropyLoss()
    writer = SummaryWriter()
    if low_ratio != 0:
        modelName = '/Student_LOW_{}x{}_'.format(str(low_ratio), str(low_ratio))
    for epoch in range(epochs):
        loss= 0.
        decay_lr(optimizer, epoch, init_lr, lr_decay)
        net.train()
        for x, x_low, y in training_generator:
            # To CUDA tensors
            x = x.cuda().float()
            x_low = x_low.cuda().float()
            y = y.cuda() - 1

            teacher = teacher_net(x)

            # Calculate gradient && Backpropagate
            optimizer.zero_grad()

            # Network output
            _, student = net(x_low)
            KD_loss = nn.KLDivLoss()(F.log_softmax(student / temperature, dim=1),
                                  F.softmax(teacher / temperature, dim=1))

            KD_loss = torch.mul(KD_loss, temperature * temperature)

            GT_loss = lossfunction(student, y)

            loss = KD_loss + GT_loss

            loss.backward()
            optimizer.step()
        net.eval()

        if (epoch + 1) % 10 > 0 :
            # print('Epoch : {}, training loss : {}'.format(epoch + 1, loss))
            logger.debug('[EPOCH{}][Training] loss : {}'.format(epoch+1,loss))
            continue
        # Test
        hit_training = 0
        hit_validation = 0
        for x_low, y in eval_trainset_generator:
            # To CUDA tensors
            x_low = torch.squeeze(x_low)
            x_low = x_low.cuda().float()
            y -= 1

            # Network output
            _, output= net(x_low)

            if ten_crop is True:
                prediction = torch.mean(output, dim=0)
                prediction = prediction.cpu().detach().numpy()

                if np.argmax(prediction) == y:
                    hit_training += 1
            else:
                _, prediction = torch.max(output, 1)
                prediction = prediction.cpu().detach().numpy()
                hit_training += (prediction == y.numpy()).sum()


        count_success = 0
        count_failure = 0
        count_show = 3
        success = []
        failure = []
        sr_success = []
        sr_failure = []
        for  x_low, y in eval_validationset_generator:
            # To CUDA tensors
            x_low = torch.squeeze(x_low)
            x_low = x_low.cuda().float()
            y -= 1

            # Network output
            sr_x, output= net(x_low)

            if ten_crop is True:
                prediction = torch.mean(output, dim=0)
                prediction = prediction.cpu().detach().numpy()

                # sr_image = vutils.make_grid(sr_x[0], normalize=True, scale_each=True)
                # image = vutils.make_grid(x_low[0], normalize=True, scale_each=True)

                if np.argmax(prediction) == y:
                    hit_validation += 1
                    if count_success > count_show :
                        continue
                    success.append(x_low[0])
                    sr_success.append(sr_x[0])
                    count_success += 1
                elif count_failure < count_show:
                    count_failure += 1
                    failure.append(x_low[0])
                    sr_failure.append(sr_x[0])
            else:
                _, prediction = torch.max(output, 1)
                prediction = prediction.cpu().detach().numpy()
                hit_validation += (prediction == y.numpy()).sum()

        if ten_crop is True:
            torch.stack(success, dim=0)
            torch.stack(failure, dim=0)
            torch.stack(sr_success, dim=0)
            torch.stack(sr_failure, dim=0)
            success = vutils.make_grid(success, normalize=True, scale_each=True)
            failure = vutils.make_grid(failure, normalize=True, scale_each=True)
            sr_success = vutils.make_grid(sr_success, normalize=True, scale_each=True)
            sr_failure = vutils.make_grid(sr_failure, normalize=True, scale_each=True)

            writer.add_image('Success', success, epoch + 1)
            writer.add_image('SR_Success', sr_success, epoch + 1)
            writer.add_image('Failure', failure, epoch + 1)
            writer.add_image('SR_Failure', sr_failure, epoch + 1)
        # Trace
        acc_training = float(hit_training) / num_training
        acc_validation = float(hit_validation) / num_validation
        logger.debug('Epoch : {}, training loss : {}'.format(epoch + 1, loss))
        logger.debug('    Training   set accuracy : {0:.2f}%, for {1:}/{2:}'
              .format(acc_training*100, hit_training, num_training))
        logger.debug('    Validation set accuracy : {0:.2f}%, for {1:}/{2:}\n'
              .format(acc_validation*100, hit_validation, num_validation))
        torch.save(net.state_dict(), result_path + modelName + str(epoch + 1) + '_epoch_acc_' + str(acc_validation* 100) + '.pt')
    # print('Finished Training')
