import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
def decay_lr(optimizer, epoch, init_lr, decay_period):
    lr = init_lr * (0.1 ** (epoch // decay_period))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        optimizer.param_groups[7]['lr'] = lr * 10

def training(
    net,
    optimizer,
    init_lr,
    lr_decay,
    epochs,
    training_generator,
    eval_trainset_generator,
    eval_validationset_generator,
    num_training,
    num_validation):
    lossfunction = nn.CrossEntropyLoss()
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
            # writer.add_scalar('loss', loss, epoch)
            print('Epoch : {}, training loss : {}'.format(epoch + 1, loss))
            # torch.save(net.state_dict(), './stanford/teachernet_' + str(epoch) + '_epoch.pt')
            # timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
            # logging.info('[{}][EPOCH{}][Training] loss : {}'.format(timestamp, epoch+1,loss))
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
            prediction = torch.mean(output, dim=0)
            prediction = prediction.cpu().detach().numpy()

            if np.argmax(prediction) == y:
                hit_training += 1


        for x, y in eval_validationset_generator:
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
        acc_training = float(hit_training) / num_training
        acc_validation = float(hit_validation) / num_validation
        print('Epoch : {}, training loss : {}'.format(epoch + 1, loss))
        print('    Training   set accuracy : {0:.2f}%, for {1:}/{2:}'
              .format(acc_training*100, hit_training, num_training))
        print('    Validation set accuracy : {0:.2f}%, for {1:}/{2:}\n'
              .format(acc_validation*100, hit_validation, num_validation))
    torch.save(net.state_dict(), './stanford/teachernet_' + str(epoch) + '_epoch_acc_' + str(acc_validation*100) +'.pt')


def training_KD(
    teacher_net,
    net,
    optimizer,
    temperature,
    init_lr,
    lr_decay,
    epochs,
    training_generator,
    eval_trainset_generator,
    eval_validationset_generator,
    num_training,
    num_validation):
    lossfunction = nn.CrossEntropyLoss()
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
            student = net(x_low)
            KD_loss = nn.KLDivLoss()(F.log_softmax(student / temperature, dim=1),
                                  F.softmax(teacher / temperature, dim=1))

            KD_loss = torch.mul(KD_loss, temperature * temperature)

            GT_loss = lossfunction(student, y)

            loss = KD_loss + GT_loss

            loss.backward()
            optimizer.step()
        net.eval()

        if (epoch + 1) % 10 > 0 :
            print('Epoch : {}, training loss : {}'.format(epoch + 1, loss))
            # torch.save(net.state_dict(), './KDmodels/studentnet_' + str(epoch) + '_epoch.pt.pt')
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
            output= net(x_low)
            prediction = torch.mean(output, dim=0)
            prediction = prediction.cpu().detach().numpy()

            if np.argmax(prediction) == y:
                hit_training += 1


        for  x_low, y in eval_validationset_generator:
            # To CUDA tensors
            x_low = torch.squeeze(x_low)
            x_low = x_low.cuda().float()
            y -= 1

            # Network output
            output= net(x_low)
            prediction = torch.mean(output, dim=0)
            prediction = prediction.cpu().detach().numpy()

            if np.argmax(prediction) == y:
                hit_validation += 1


        # Trace
        acc_training = float(hit_training) / num_training
        acc_validation = float(hit_validation) / num_validation
        print('Epoch : {}, training loss : {}'.format(epoch + 1, loss))
        print('    Training   set accuracy : {0:.2f}%, for {1:}/{2:}'
              .format(acc_training*100, hit_training, num_training))
        print('    Validation set accuracy : {0:.2f}%, for {1:}/{2:}\n'
              .format(acc_validation*100, hit_validation, num_validation))

    torch.save(net.state_dict(), './stanford_KD/studentnet_' + str(epoch) + '_epoch_acc_' + str(acc_validation* 100) + '.pt')
    print('Finished Training')
