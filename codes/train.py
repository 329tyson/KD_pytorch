import torch
import alexnet
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torchvision import transforms
from torch.utils import data

import time
import datetime
import logging
import os

# from dataloader import CUBDataset
from car_dataloader import CARDataset

# Training Params
params = {'batch_size': 111,
          'shuffle': True,
          'num_workers': 6,
          'drop_last' : True}

eval_params = {'batch_size': 1,
          'shuffle': True,
          'num_workers': 6,
          'drop_last' : True}

weights_path = './bvlc_alexnet.npy'
init_lr = 0.001
decay_period = 20

logging.info('=======================Training High Resolution images alone============================')
logging.info('[{}][train.py][          lr] : {}'.format(datetime.datetime.now().strftime('%dDT%HH%MM'),str(init_lr)))
logging.info('[{}][train.py][decay_period] : {}'.format(datetime.datetime.now().strftime('%dDT%HH%MM'),str(decay_period)))
logging.info('[{}][train.py][  batch_size] : {}'.format(datetime.datetime.now().strftime('%dDT%HH%MM'),str(params['batch_size'])))

writer = SummaryWriter(log_dir = './runs/train.py_lr:{}_batch_size:{}_decay:{}@{}'.format(init_lr, params['batch_size'], decay_period,datetime.datetime.now().strftime('D%dT%HH%MM')))
# writer = SummaryWriter()

# for CUB dataset
# net = alexnet.AlexNet(0.5, 200, ['fc8'], True)

# for stanford CAR dataset
net = alexnet.AlexNet(0.5, 196, ['fc8'], True)

# Small testset & test.csv
# training_set = CUBDataset('../TestImagelabels.csv','../TestImages/')

# Generate training dataset
# training_set = CUBDataset('../labels/label_train_cub200_2011.csv', '../CUB_200_2011/images/', True)
# training_generator = data.DataLoader(training_set, **params)

# # Generate datasets for Test
# eval_trainset = CUBDataset('../labels/label_train_cub200_2011.csv', '../CUB_200_2011/images/', False)
# eval_trainset_generator = data.DataLoader(eval_trainset, **eval_params)
# eval_validationset = CUBDataset('../labels/label_val_cub200_2011.csv', '../CUB_200_2011/images/', False)
# eval_validationset_generator = data.DataLoader(eval_validationset, **eval_params)

training_set = CARDataset('../stanford/cars_annos.mat', '../stanford/', True, 'Train')
training_generator = data.DataLoader(training_set, **params)

# Generate datasets for Test
eval_trainset = CARDataset('../stanford/cars_annos.mat', '../stanford/', False, 'Train')
eval_trainset_generator = data.DataLoader(eval_trainset, **eval_params)
eval_validationset = CARDataset('../stanford/cars_annos.mat', '../stanford/', False, 'Eval')
eval_validationset_generator = data.DataLoader(eval_validationset, **eval_params)
# Fetch lengths
num_training = len(training_set)
num_eval_trainset = len(eval_trainset)
num_eval_validationset = len(eval_validationset)

print('Training images : ', num_training)
print('Validation images : ', num_eval_validationset)
# loading pretrained weights from bvlc_alexnet.npy
pretrained= np.load('bvlc_alexnet.npy', encoding='latin1').item()
converted = net.state_dict()
for lname, val in pretrained.items():
    if 'conv' in lname:
        converted[lname+".weight"] = torch.from_numpy(val[0].transpose(3,2,0,1))
        converted[lname+".bias"] = torch.from_numpy(val[1])
    elif 'fc8' in lname:
        continue
    elif 'fc' in lname:
        converted[lname+".weight"] = torch.from_numpy(val[0].transpose(1,0))
        converted[lname+".bias"] = torch.from_numpy(val[1])

net.load_state_dict(converted, strict = True)
net.cuda()
lossfunction = nn.CrossEntropyLoss()

def decay_lr(optimizer, epoch):
    lr = init_lr * (0.1 ** (epoch // decay_period))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.param_groups[7]['lr'] = lr * 10

optimizer= optim.SGD(
    [{'params':net.conv1.parameters()},
     {'params':net.conv2.parameters()},
     {'params':net.conv3.parameters()},
     {'params':net.conv4.parameters()},
     {'params':net.conv5.parameters()},
     {'params':net.fc6.parameters()},
     {'params':net.fc7.parameters()},
     {'params':net.fc8.parameters(), 'lr':0.01}],
     lr=init_lr,
     momentum = 0.9,
     weight_decay = 0.0005)

for epoch in range(100):
    loss= 0.
    decay_lr(optimizer, epoch)
    net.train()
    for x, _, y in training_generator:
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
        writer.add_scalar('loss', loss, epoch)
        print('Epoch : {}, training loss : {}'.format(epoch + 1, loss))
        # torch.save(net.state_dict(), './stanford/teachernet_' + str(epoch) + '_epoch.pt')
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
        logging.info('[{}][EPOCH{}][Training] loss : {}'.format(timestamp, epoch+1,loss))
        continue
    # trace accuracy conditionally
    # if (epoch + 1) < 35 :
        # writer.add_scalar('loss', loss, epoch)
        # print('Epoch : {}, training loss : {}'.format(epoch + 1, loss))
        # torch.save(net.state_dict(), './models/teachernet_' + str(epoch) + '_epoch.pt')
        # timestamp = datetime.datetime.now().strftime('%Y%m%d %H:%M')
        # logging.info('[{}][{}] spent {} ms'.format(timestamp, "{:>17}".format(method.__name__ + '()'), "{0:.3f}".format((te-ts)*1000)))
        # continue
    hit_training = 0
    hit_validation = 0
    for x, _, y in eval_trainset_generator:
        # To CUDA tensors
        x = torch.squeeze(x)
        x = x.cuda().float()
        y -= 1

        # Network output
        output= net(x)
        prediction = torch.mean(output, dim=0)
        prediction = prediction.cpu().detach().numpy()

        if np.argmax(prediction) == y:
            hit_training += 1

        # Count prediction hit on training set
        # prediction = torch.max(output, 1)[1]
        # hit_training += np.sum(prediction.cpu().numpy() ==  y.numpy())

    for x, _, y in eval_validationset_generator:
        # To CUDA tensors
        x = torch.squeeze(x)
        x = x.cuda().float()
        y -= 1

        # Network output
        output= net(x)
        prediction = torch.mean(output, dim=0)
        prediction = prediction.cpu().detach().numpy()

        if np.argmax(prediction) == y:
            hit_validation += 1

        # Count prediction hit on training set
        # prediction = torch.max(output, 1)[1]
        # hit_validation += np.sum(prediction.cpu().numpy() ==  y.numpy())

    # Trace
    acc_training = float(hit_training) / num_eval_trainset
    acc_validation = float(hit_validation) / num_eval_validationset
    print('Epoch : {}, training loss : {}'.format(epoch + 1, loss))
    print('    Training   set accuracy : {0:.2f}%, for {1:}/{2:}'
          .format(acc_training*100, hit_training, num_eval_trainset))
    print('    Validation set accuracy : {0:.2f}%, for {1:}/{2:}\n'
          .format(acc_validation*100, hit_validation, num_eval_validationset))
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    logging.info('[{}][EPOCH{}][Evaluate] loss : {} Training :{} Validation : {}'
                 .format(timestamp,
                         epoch + 1,
                         loss,
                         acc_training,
                         acc_validation
                         )
                 )
    # Log Tensorboard
    writer.add_scalar('loss', loss, epoch)
    writer.add_scalars('Accuracies',
                       {'Training accuracy': acc_training,
                        'Validation accuracy': acc_validation}, epoch)
    torch.save(net.state_dict(), './stanford/teachernet_' + str(epoch) + '_epoch_acc_' + str(acc_validation*100) +'.pt')

print('Finished Training')
writer.close()
