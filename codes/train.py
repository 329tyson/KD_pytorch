import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def decay_lr(optimizer, epoch, init_lr, decay_period):
    lr = init_lr * (0.1 ** (epoch // decay_period))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        optimizer.param_groups[7]['lr'] = lr * 10


def calculate_Gram_loss(s_feature, t_feature, norm_type, patch_num, style_weight, mse_loss):
    bn, c, h, w = t_feature.shape

    gram_losses = []

    for i in range(patch_num):
        x_ = int(round((w * i / float(patch_num))))
        x_w = int(round((w * (i + 1) / float(patch_num))))

        for j in range(patch_num):
            y_ = int(round((h * j / float(patch_num))))
            y_h = int(round((h * (j + 1) / float(patch_num))))

            t_vec = t_feature[:, :, y_: y_h, x_:x_w]
            s_vec = s_feature[:, :, y_: y_h, x_:x_w]

            t_vec = t_vec.contiguous().view(bn, c, -1)
            s_vec = s_vec.contiguous().view(bn, c, -1)

            if norm_type == 1:
                t_vec = t_vec.div((x_w - x_) * (y_h - y_))
                s_vec = s_vec.div((x_w - x_) * (y_h - y_))

            if norm_type == 2:
                t_vec = F.normalize(t_vec, p=2, dim=2)
                s_vec = F.normalize(s_vec, p=2, dim=2)

            t_Gram = torch.bmm(t_vec, t_vec.permute((0, 2, 1)))
            s_Gram = torch.bmm(s_vec, s_vec.permute((0, 2, 1)))

            if norm_type == 3 or norm_type == 4:
                t_Gram = t_Gram.div((x_w - x_) * (y_h - y_))
                s_Gram = s_Gram.div((x_w - x_) * (y_h - y_))

            gram_losses.append(mse_loss(s_Gram, t_Gram))

    if norm_type == 4:
        loss = style_weight * torch.mean(torch.stack(gram_losses)) / (c ** 2)

    else:
        loss = style_weight * torch.mean(torch.stack(gram_losses))

    return loss


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
    logger,
    save):
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
            logger.info('[EPOCH{}][Training] loss : {}'.format(epoch+1,loss))
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
        logger.info('Epoch : {}, training loss : {}'.format(epoch + 1, loss))
        logger.info('    Training   set accuracy : {0:.2f}%, for {1:}/{2:}'
              .format(acc_training*100, hit_training, num_training))
        logger.info('    Validation set accuracy : {0:.2f}%, for {1:}/{2:}\n'
              .format(acc_validation*100, hit_validation, num_validation))
        if save:
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
    logger,
    save):
    lossfunction = nn.CrossEntropyLoss()
    if low_ratio != 0:
        modelName = '/Student_LOW_{}x{}_'.format(str(low_ratio), str(low_ratio))
    else:
        print('are you serious ...?')

    teacher_net.eval()

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
            logger.info('[EPOCH{}][Training] loss : {}'.format(epoch+1,loss))
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

            if ten_crop is True:
                prediction = torch.mean(output, dim=0)
                prediction = prediction.cpu().detach().numpy()

                if np.argmax(prediction) == y:
                    hit_training += 1
            else:
                _, prediction = torch.max(output, 1)
                prediction = prediction.cpu().detach().numpy()
                hit_training += (prediction == y.numpy()).sum()


        for  x_low, y in eval_validationset_generator:
            # To CUDA tensors
            x_low = torch.squeeze(x_low)
            x_low = x_low.cuda().float()
            y -= 1

            # Network output
            output = net(x_low)

            if ten_crop is True:
                prediction = torch.mean(output, dim=0)
                prediction = prediction.cpu().detach().numpy()

                if np.argmax(prediction) == y:
                    hit_validation += 1
            else:
                _, prediction = torch.max(output, 1)
                prediction = prediction.cpu().detach().numpy()
                hit_validation += (prediction == y.numpy()).sum()


        # Trace
        acc_training = float(hit_training) / num_training
        acc_validation = float(hit_validation) / num_validation
        print('Epoch : {}, training loss : {}'.format(epoch + 1, loss))
        print('    Training   set accuracy : {0:.2f}%, for {1:}/{2:}'
              .format(acc_training*100, hit_training, num_training))
        print('    Validation set accuracy : {0:.2f}%, for {1:}/{2:}\n'
              .format(acc_validation*100, hit_validation, num_validation))
        logger.info('Epoch : {}, training loss : {}'.format(epoch + 1, loss))
        logger.info('    Training   set accuracy : {0:.2f}%, for {1:}/{2:}'
              .format(acc_training*100, hit_training, num_training))
        logger.info('    Validation set accuracy : {0:.2f}%, for {1:}/{2:}\n'
              .format(acc_validation*100, hit_validation, num_validation))
        if save:
            torch.save(net.state_dict(), result_path + modelName + str(epoch + 1) + '_epoch_acc_' + str(acc_validation* 100) + '.pt')
    print('Finished Training')


def training_Gram_KD(
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
    logger,
    style_weight,
    norm_type,
    patch_num,
    gram_features,
    hint,
    save
    ):

    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    if low_ratio != 0:
        modelName = '/Student_LOW_{}x{}_'.format(str(low_ratio), str(low_ratio))
    else:
        print('are you serious ...?')

    teacher_net.eval()

    if hint:
        print('1st stage Training using Gram loss')
        for epoch in range(40):
            loss = 0.
            decay_lr(optimizer, epoch, init_lr, lr_decay)
            net.train()

            for x, x_low, y in training_generator:
                x = x.cuda().float()
                x_low = x_low.cuda().float()
                y = y.cuda() - 1

                _, t_conv1, t_conv2, t_conv3, t_conv4, t_conv5 = teacher_net(x)

                t_conv1 = t_conv1.detach()
                t_conv2 = t_conv2.detach()
                t_conv3 = t_conv3.detach()
                t_conv4 = t_conv4.detach()
                t_conv5 = t_conv5.detach()

                optimizer.zero_grad()

                _, s_conv1, s_conv2, s_conv3, s_conv4, s_conv5 = net(x_low)

                loss = []

                if 1 in gram_features:
                    loss.append(calculate_Gram_loss(s_conv1, t_conv1, norm_type, patch_num, style_weight, mse_loss))
                if 2 in gram_features:
                    loss.append(calculate_Gram_loss(s_conv2, t_conv2, norm_type, patch_num, style_weight, mse_loss))
                if 3 in gram_features:
                    loss.append(calculate_Gram_loss(s_conv3, t_conv3, norm_type, patch_num, style_weight, mse_loss))
                if 4 in gram_features:
                    loss.append(calculate_Gram_loss(s_conv4, t_conv4, norm_type, patch_num, style_weight, mse_loss))
                if 5 in gram_features:
                    loss.append(calculate_Gram_loss(s_conv5, t_conv5, norm_type, patch_num, style_weight, mse_loss))

                # print loss

                loss = torch.mean(torch.stack(loss))

                loss.backward()
                optimizer.step()
            print('In 1st stage, epoch : {}, total loss : {}'.format(
                    epoch, loss.data.cpu()))

    for epoch in range(epochs):
        loss= 0.
        decay_lr(optimizer, epoch, init_lr, lr_decay)
        net.train()
        for x, x_low, y in training_generator:
            # To CUDA tensors
            x = x.cuda().float()
            x_low = x_low.cuda().float()
            y = y.cuda() - 1

            teacher, t_conv1, t_conv2, t_conv3, t_conv4, t_conv5 = teacher_net(x)

            t_conv1 = t_conv1.detach()
            t_conv2 = t_conv2.detach()
            t_conv3 = t_conv3.detach()
            t_conv4 = t_conv4.detach()
            t_conv5 = t_conv5.detach()

            # Calculate gradient && Backpropagate
            optimizer.zero_grad()

            # Network output
            student, s_conv1, s_conv2, s_conv3, s_conv4, s_conv5 = net(x_low)

            KD_loss = nn.KLDivLoss()(F.log_softmax(student / temperature, dim=1),
                                  F.softmax(teacher / temperature, dim=1))

            KD_loss = torch.mul(KD_loss, temperature * temperature)

            GT_loss = ce_loss(student, y)

            GRAM_loss = .0

            if hint == False:
                if 1 in gram_features:
                    GRAM_loss += calculate_Gram_loss(s_conv1, t_conv1, norm_type, patch_num, style_weight, mse_loss)
                if 2 in gram_features:
                    GRAM_loss += calculate_Gram_loss(s_conv2, t_conv2, norm_type, patch_num, style_weight, mse_loss)
                if 3 in gram_features:
                    GRAM_loss += calculate_Gram_loss(s_conv3, t_conv3, norm_type, patch_num, style_weight, mse_loss)
                if 4 in gram_features:
                    GRAM_loss += calculate_Gram_loss(s_conv4, t_conv4, norm_type, patch_num, style_weight, mse_loss)
                if 5 in gram_features:
                    GRAM_loss += calculate_Gram_loss(s_conv5, t_conv5, norm_type, patch_num, style_weight, mse_loss)

                GRAM_loss /= len(gram_features)

            loss = KD_loss + GT_loss + GRAM_loss

            
            # if hint == False:
            #     print KD_loss.data.cpu(), GT_loss.data.cpu(), GRAM_loss.data.cpu()
            # else:
            #     print KD_loss.data.cpu(), GT_loss.data.cpu()
                
            loss.backward()
            optimizer.step()
        net.eval()

        if (epoch + 1) % 10 > 0 :
            print('Epoch : {}, training loss : {}'.format(epoch + 1, loss))
            logger.info('[EPOCH{}][Training] loss : {}'.format(epoch+1,loss))
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
            output, _, _, _, _, _ = net(x_low)

            if ten_crop is True:
                prediction = torch.mean(output, dim=0)
                prediction = prediction.cpu().detach().numpy()

                if np.argmax(prediction) == y:
                    hit_training += 1
            else:
                _, prediction = torch.max(output, 1)
                prediction = prediction.cpu().detach().numpy()
                hit_training += (prediction == y.numpy()).sum()


        for  x_low, y in eval_validationset_generator:
            # To CUDA tensors
            x_low = torch.squeeze(x_low)
            x_low = x_low.cuda().float()
            y -= 1

            # Network output
            output, _, _, _, _, _  = net(x_low)

            if ten_crop is True:
                prediction = torch.mean(output, dim=0)
                prediction = prediction.cpu().detach().numpy()

                if np.argmax(prediction) == y:
                    hit_validation += 1
            else:
                _, prediction = torch.max(output, 1)
                prediction = prediction.cpu().detach().numpy()
                hit_validation += (prediction == y.numpy()).sum()


        # Trace
        acc_training = float(hit_training) / num_training
        acc_validation = float(hit_validation) / num_validation
        print('Epoch : {}, training loss : {}'.format(epoch + 1, loss))
        print('    Training   set accuracy : {0:.2f}%, for {1:}/{2:}'
              .format(acc_training*100, hit_training, num_training))
        print('    Validation set accuracy : {0:.2f}%, for {1:}/{2:}\n'
              .format(acc_validation*100, hit_validation, num_validation))
        logger.info('Epoch : {}, training loss : {}'.format(epoch + 1, loss))
        logger.info('    Training   set accuracy : {0:.2f}%, for {1:}/{2:}'
              .format(acc_training*100, hit_training, num_training))
        logger.info('    Validation set accuracy : {0:.2f}%, for {1:}/{2:}\n'
              .format(acc_validation*100, hit_validation, num_validation))

        if save:
            torch.save(net.state_dict(), result_path + modelName + str(epoch + 1) + '_epoch_acc_' + str(acc_validation* 100) + '.pt')
