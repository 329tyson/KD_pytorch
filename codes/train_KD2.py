import torch
import alexnet
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tensorboardX import SummaryWriter
from dataloader import CUBDataset
from torchvision import transforms
from torch.utils import data

soft = nn.Softmax(dim=1)
temperature = 3.0
low_res_ratio = 4

# Training Params
params = {'batch_size': 128,
          'shuffle': True,
          'num_workers': 6,
          'drop_last' : True}

writer = SummaryWriter()

weights_path = './bvlc_alexnet.npy'
init_lr = 0.001

net = alexnet.AlexNet(0.5, 200, ['fc8'], True)
teacher_net = alexnet.AlexNet(0.5, 200, ['fc8'], True)

# Small testset & test.csv
# training_set = CUBDataset('../TestImagelabels.csv','../TestImages/')

# Generate training dataset
training_set = CUBDataset('../labels/label_train_cub200_2011.csv', '../CUB_200_2011/images/')
training_generator = data.DataLoader(training_set, **params)

# Generate validation dataset
validation_set = CUBDataset('../labels/label_val_cub200_2011.csv', '../CUB_200_2011/images/')
validation_generator = data.DataLoader(validation_set, **params)

# Fetch lengths
num_training = len(training_set)
num_validation = len(validation_set)

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

# TODO: fc8's weight should be assigned
teacher_weight = torch.load('teachernet_43_epoch.pt')
teacher_net.load_state_dict(teacher_weight, strict=True)
teacher_net.cuda()
teacher_net.eval()

lossfunction = nn.CrossEntropyLoss()

def decay_lr(optimizer, epoch):
    lr = init_lr * (0.1 ** (epoch //25))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

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
    for x, x_low, y in training_generator:
        # To CUDA tensors
        x = x.cuda().float()
        x_low = x_low.cuda().float()
        y = y.cuda() - 1

        teacher = teacher_net(x)

        # Calculate gradient && Backpropagate
        optimizer.zero_grad()

        # Network output
        student = net(x_low)

        # KD_loss = lossfunction(torch.div(student, temperature), torch.div(teacher, temperature))
        KD_loss = nn.KLDivLoss()(F.log_softmax(student / temperature, dim=1),
                                 F.softmax(teacher / temperature, dim=1))

        KD_loss = torch.mul(KD_loss, temperature * temperature)

        GT_loss = lossfunction(student, y)

        # TODO: alpha? balance parameter?
        loss = KD_loss + GT_loss

        loss.backward()
        optimizer.step()

    net.eval()

    # Test
    hit_training = 0
    hit_validation = 0
    for _, x_low, y in training_generator:
        # To CUDA tensors
        x_low = torch.squeeze(x_low)
        x_low = x_low.cuda().float()
        y -= 1

        # Network output
        output = net(x_low)

        prediction = torch.mean(output, dim=0)
        prediction = prediction.cpu().detach().numpy()

        if np.argmax(prediction) == y:
            hit_training += 1

    for _, x_low, y in validation_generator:
        # To CUDA tensors
        x_low = torch.squeeze(x_low)
        x_low = x_low.cuda().float()
        y -= 1

        # Network output
        output = net(x_low)

        prediction = torch.mean(output, dim=0)
        prediction = prediction.cpu().detach().numpy()

        if np.argmax(prediction) == y:
            hit_training += 1

    # Trace
    acc_training = float(hit_training) / num_training
    acc_validation = float(hit_validation) / num_validation
    print('Epoch : {}, training loss : {}'.format(epoch + 1, loss))
    print('    Training   set accuracy : {0:.2f}%, for {1:}/{2:}'
          .format(acc_training*100, hit_training, num_training))
    print('    Validation set accuracy : {0:.2f}%, for {1:}/{2:}\n'
          .format(acc_validation*100, hit_validation, num_validation))

    # Log Tensorboard
    writer.add_scalar('GroundTruth loss', loss, epoch)
    writer.add_scalars('Accuracies',
                       {'Training accuracy': acc_training,
                        'Validation accuracy': acc_validation}, epoch)
    # writer.add_scalar('Validation Accuracy', acc_validation, epoch)

print('Finished Training')
writer.close()
