import torch
import alexnet
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from dataloader import CUBDataset
from torchvision import transforms
from torch.utils import data

# Training Params
params = {'batch_size': 128,
          'shuffle': True,
          'num_workers': 6,
          'drop_last' : True}

writer = SummaryWriter()

weights_path = './bvlc_alexnet.npy'
softmax = nn.Softmax(dim = 1)

net = alexnet.AlexNet(0.5, 200, ['fc8'], True)

# training_set = CUBDataset('../TestImagelabels.csv','../TestImages/')
training_set = CUBDataset('../labels/label_train_cub200_2011.csv', '../CUB_200_2011/images/')
training_generator = data.DataLoader(training_set, **params)

validation_set = CUBDataset('../labels/label_val_cub200_2011.csv', '../CUB_200_2011/images/')
validation_generator = data.DataLoader(validation_set, **params)

num_training = len(training_set)
num_validation = len(validation_set)

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

optimizer= optim.SGD(
    [{'params':net.conv1.parameters()},
     {'params':net.conv2.parameters()},
     {'params':net.conv3.parameters()},
     {'params':net.conv4.parameters()},
     {'params':net.conv5.parameters()},
     {'params':net.fc6.parameters()},
     {'params':net.fc7.parameters()},
     {'params':net.fc8.parameters(), 'lr':0.01}],
     lr=0.001,
     momentum = 0.9,
     weight_decay = 0.0005)
for epoch in range(100):
    loss= 0.
    for x, y in training_generator:
        x = x.cuda().float()
        y = y.cuda() - 1

        # Calculate gradient && Backpropagate
        optimizer.zero_grad()

        # Network output
        # print(x[0])
        output = net(x)
        # conv1, conv2, conv3, conv4, conv5, fc6, fc7, output = net(x)
        # output, fc6, fc7 = net(x)

        # print('conv1 : ', conv1[0])
        # print('conv2 : ', conv2[0])
        # print('conv3 : ', conv3[0])
        # print('conv4 : ', conv4[0])
        # print('conv5 : ', conv5[0])
        # print('fc6   : ',fc6[0])
        # print('fc7   : ',fc7[0])
        # print('output: ',output[0])
        # print(torch.max(output,1)[1][0])
        # print(y[0])
        # import ipdb; ipdb.set_trace()


        loss = lossfunction(output, y)
        loss.backward()
        optimizer.step()

    # Test
    hit_training = 0
    hit_validation = 0
    for x, y in training_generator:
        x = x.cuda().float()
        y -= 1

        # Network output
        output= net(x)

        prediction = torch.max(output, 1)[1]
        hit_training = np.sum(prediction.cpu().numpy() ==  y.numpy())

        # print('\n===Training Samples\n')
        # print(output[0])
        # print(prediction[0])
        # print(y[0])

    for x, y in validation_generator:
        x = x.cuda().float()
        y -= 1

        # Network output
        output= net(x)

        prediction = torch.max(output, 1)[1]
        hit_validation = np.sum(prediction.cpu().numpy() ==  y.numpy())

        # print('\n===Validation Samples\n')
        # print(output[0])
        # print(prediction[0])

    # Trace
    print('Epoch : {}, training loss : {}'.format(epoch + 1, loss))
    print('    Training   set accuracy : {0:.2f}%, for {1:}/{2:}'
          .format((hit_training/num_training)*100, hit_training, num_training))
    print('    Validation set accuracy : {0:.2f}%, for {1:}/{2:}\n'
          .format((hit_validation/num_validation)*100, hit_validation, num_validation))

    # Log Tensorboard
    # writer.add_scalar('GroundTruth loss', loss)

print('Finished Training')
writer.close()
