import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def decay_lr(optimizer, epoch, init_lr, decay_period):
    lr = init_lr * (0.1 ** (epoch // decay_period))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.param_groups[7]['lr'] = lr * 10

def decay_lr_vgg(optimizer, epoch, init_lr, decay_period):
    lr = init_lr * (0.1 ** (epoch // decay_period))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def calculate_Gram_loss(s_feature, t_feature, norm_type, patch_num, style_weight, mse_loss, ratio):
    bn, c, h, w = t_feature.shape

    gram_losses = []

    # ratio = 4
    reduced_c = int(c / ratio)

    for i in range(patch_num):
        x_ = int(round((w * i / float(patch_num))))
        x_w = int(round((w * (i + 1) / float(patch_num))))

        for j in range(patch_num):
            y_ = int(round((h * j / float(patch_num))))
            y_h = int(round((h * (j + 1) / float(patch_num))))

            spatial_size = (y_h - y_) * (x_w - x_)

            t_vec = t_feature[:, :, y_: y_h, x_:x_w]    # shape: [bn, c, patch_h, patch_w]
            s_vec = s_feature[:, :, y_: y_h, x_:x_w]

            t_vec = t_vec.contiguous().view(bn, c, -1)  # shape: [bn, c, patch_h*w]
            s_vec = s_vec.contiguous().view(bn, c, -1)

            t_at = torch.sum(t_vec, 2)  # shape: [bn, c], channel attention map

            _, index = torch.sort(t_at, dim=1, descending=True) # shape: [bn, c]
            index, _ = torch.sort(index[:, :reduced_c], dim=1)    # shape: [bn, reduced_c]

            # indices = index.view(-1,1).repeat(1, spatial_size).view(t_vec.shape)
            # t_vec = torch.gather(t_vec, 1, indices)
            # s_vec = torch.gather(s_vec, 1, indices)
            # t_vec = t_vec[:,:c/ratio,:]
            # s_vec = s_vec[:,:c/ratio,:]

            t_vec = torch.cat([torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(t_vec, index)])
            s_vec = torch.cat([torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(s_vec, index)])

            if norm_type == 1:
                t_vec = t_vec.div(spatial_size)
                s_vec = s_vec.div(spatial_size)

            if norm_type == 2:
                t_vec = F.normalize(t_vec, p=2, dim=2)
                s_vec = F.normalize(s_vec, p=2, dim=2)

            t_Gram = torch.bmm(t_vec, t_vec.permute((0, 2, 1)))
            s_Gram = torch.bmm(s_vec, s_vec.permute((0, 2, 1)))

            if norm_type == 3 or norm_type == 4:
                t_Gram = t_Gram.div(spatial_size)
                s_Gram = s_Gram.div(spatial_size)

            gram_losses.append(mse_loss(s_Gram, t_Gram))

    if norm_type == 4:
        loss = style_weight * torch.mean(torch.stack(gram_losses)) / (reduced_c ** 2)

    else:
        loss = style_weight * torch.mean(torch.stack(gram_losses))

    return loss


def calculate_attendedGram_loss(s_feature, t_feature, norm_type, style_weight, mse_loss, ratio):
    bn, c, h, w = t_feature.shape
    """
    calculate l2 loss b.t.w teacher and student attended gram
    :param s_feature, t_feature: shape=[bn, c, h, w]
    :param at: shape = [bn, h*w]
    :param norm_type (3: mat normlaized by h*w, 4: normalized by c*h*w)
    """

    spatial_size = h * w

    # ratio = 10

    reduced_size = int(spatial_size / ratio)

    t_feature = t_feature.view(bn,c,-1)
    s_feature = s_feature.view(bn,c,-1)

    at = torch.sum(t_feature, dim=1)

    _, index= torch.sort(at, dim=1, descending=True) # [batch, h*w]
    index, _ = torch.sort(index[:, :reduced_size], dim=1)   # [batch, reduced_size]
    # index = index.detach()

    # indices = index.view(-1, 1).repeat(1, c).view(bn, -1, c).permute((0, 2, 1))
    # indices = indices.detach()
    # t_feature = torch.gather(t_feature, 2, indices)
    # s_feature = torch.gather(s_feature, 2, indices)
    # t_feature = t_feature[:,:,:(h*w)/ratio]
    # s_feature = s_feature[:,:,:(h*w)/ratio]

    t_feature = torch.cat([torch.index_select(a, 1, i).unsqueeze(0) for a, i in zip(t_feature, index)])
    s_feature = torch.cat([torch.index_select(a, 1, i).unsqueeze(0) for a, i in zip(s_feature, index)])

    t_Gram = torch.bmm(t_feature, t_feature.permute((0, 2, 1)))
    s_Gram = torch.bmm(s_feature, s_feature.permute((0, 2, 1)))

    loss = mse_loss(s_Gram, t_Gram)

    if norm_type == 3:
        loss = loss / (reduced_size ** 2)
    elif norm_type == 4:
        loss = loss / ((c*reduced_size) ** 2)

    loss *= style_weight

    return loss


def CAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps
    bz, nc, h, w = feature_conv.shape

    cam = weight_softmax[class_idx].unsqueeze(2).permute(0, 2, 1) * feature_conv.view(bz, nc, -1).permute(0, 2, 1)
    cam = cam.permute(0,2,1).view(bz, nc, h, w)
    cam = torch.sum(cam, dim=1)
    cam = cam.unsqueeze(1)

    return cam


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
    vgg_gap,
    save
    ):
    lossfunction = nn.CrossEntropyLoss()
    if low_ratio != 0:
        modelName = '/teacher_LOW_{}x{}_'.format(str(low_ratio), str(low_ratio))
    else:
        modelName = '/teacher_HIGH_'
    print 'modelName = ', result_path + modelName
    for epoch in range(epochs):
        loss= 0.
        if not vgg_gap:
            decay_lr(optimizer, epoch, init_lr, lr_decay)
        else:
            decay_lr_vgg(optimizer, epoch, init_lr, lr_decay)
        net.train()
        for x, y in training_generator:
            # To CUDA tensors
            x = x.cuda().float()
            y = y.cuda() - 1

            # Calculate gradient && Backpropagate
            optimizer.zero_grad()

            # Network output
            output, _ = net(x)

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
            output, _ = net(x)

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
            output, _ = net(x)

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
            # torch.save(net.state_dict(), result_path + modelName + str(epoch + 1) + '_epoch_acc_' + str(acc_validation*100) +'.pt')
            torch.save(net.state_dict(),
                       result_path + modelName + str(epoch + 1) + '_epoch_acc_' + str(acc_validation * 100) + '.pt')


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
    vgg_gap,
    save
    ):
    lossfunction = nn.CrossEntropyLoss()
    if low_ratio != 0:
        modelName = '/Student_LOW_{}x{}_'.format(str(low_ratio), str(low_ratio))
    else:
        print('are you serious ...?')

    teacher_net.eval()

    """
    # get the softmax(?) weight
    params = list(net.parameters())
    weight_softmax = params[-2].data.detach()
    # shape : [200, 1024]
    """

    temperature2 = 5

    for epoch in range(epochs):
        loss= 0.
        if not vgg_gap:
            decay_lr(optimizer, epoch, init_lr, lr_decay)
        else:
            decay_lr_vgg(optimizer, epoch, init_lr, lr_decay)
        net.train()
        for x, x_low, y in training_generator:
            # To CUDA tensors
            x = x.cuda().float()
            x_low = x_low.cuda().float()
            y = y.cuda() - 1

            # teacher = teacher_net(x)
            teacher, t_feature = teacher_net(x)
            t_feature = t_feature.detach()

            # Calculate gradient && Backpropagate
            optimizer.zero_grad()

            # Network output
            student, s_feature = net(x_low)

            # Calculate Region KD between CAM region of teacher & student
            """
            bn, c, h, w = t_feature.shape

            t_CAMs = CAM(t_feature, weight_softmax, y).view(bn, -1, h, w)
            s_CAMs = CAM(s_feature, weight_softmax, y).view(bn, -1, h, w)
            t_CAMs = F.upsample(t_CAMs, size=(25,25), mode='bilinear').view(bn, -1)
            s_CAMs = F.upsample(s_CAMs, size=(25,25), mode='bilinear').view(bn, -1)

            Region_KD_loss = nn.KLDivLoss()(F.log_softmax(s_CAMs / temperature2, dim=1),
                                            F.softmax(t_CAMs / temperature2, dim=1))
            # TODO: is this correct?
            Region_KD_loss = torch.mul(Region_KD_loss, temperature2 * temperature2)
            """

            KD_loss = nn.KLDivLoss()(F.log_softmax(student / temperature, dim=1),
                                     F.softmax(teacher / temperature, dim=1))

            KD_loss = torch.mul(KD_loss, temperature * temperature)

            GT_loss = lossfunction(student, y)

            loss = KD_loss + GT_loss
            # loss = KD_loss + GT_loss + Region_KD_loss

            # print KD_loss, GT_loss, Region_KD_loss

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
            output, _ = net(x_low)

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
            output, _ = net(x_low)

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
    at_enabled,
    at_ratio,
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

                _, t_features = teacher_net(x)

                t_conv1 = t_features['conv1'].detach()
                t_conv2 = t_features['conv2'].detach()
                t_conv3 = t_features['conv3'].detach()
                t_conv4 = t_features['conv4'].detach()
                t_conv5 = t_features['conv5'].detach()

                optimizer.zero_grad()

                _, s_features = net(x_low)

                s_conv1 = s_features['conv1']
                s_conv2 = s_features['conv2']
                s_conv3 = s_features['conv3']
                s_conv4 = s_features['conv4']
                s_conv5 = s_features['conv5']

                loss = []

                if 1 in gram_features:
                    if at_enabled:
                        loss.append(calculate_attendedGram_loss(s_conv1, t_conv1, norm_type, style_weight, mse_loss, at_ratio))
                    else:
                        loss.append(calculate_Gram_loss(s_conv1, t_conv1, norm_type, patch_num, style_weight, mse_loss, at_ratio))
                if 2 in gram_features:
                    if at_enabled:
                        loss.append(calculate_attendedGram_loss(s_conv2, t_conv2, norm_type, style_weight, mse_loss, at_ratio))
                    else:
                        loss.append(calculate_Gram_loss(s_conv2, t_conv2, norm_type, patch_num, style_weight, mse_loss, at_ratio))
                if 3 in gram_features:
                    if at_enabled:
                        loss.append(calculate_attendedGram_loss(s_conv3, t_conv3, norm_type, style_weight, mse_loss, at_ratio))
                    else:
                        loss.append(calculate_Gram_loss(s_conv3, t_conv3, norm_type, patch_num, style_weight, mse_loss, at_ratio))
                if 4 in gram_features:
                    if at_enabled:
                        loss.append(calculate_attendedGram_loss(s_conv4, t_conv4, norm_type, style_weight, mse_loss, at_ratio))
                    else:
                        loss.append(calculate_Gram_loss(s_conv4, t_conv4, norm_type, patch_num, style_weight, mse_loss, at_ratio))
                if 5 in gram_features:
                    if at_enabled:
                        loss.append(calculate_attendedGram_loss(s_conv5, t_conv5, norm_type, style_weight, mse_loss, at_ratio))
                    else:
                        loss.append(calculate_Gram_loss(s_conv5, t_conv5, norm_type, patch_num, style_weight, mse_loss, at_ratio))

                # print loss

                loss = torch.mean(torch.stack(loss))

                # print loss.data.cpu()

                if loss == float('inf') or loss != loss:
                    print('Loss is infinity, stop!')
                    return

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

            teacher, t_features = teacher_net(x)

            t_conv1 = t_features['conv1'].detach()
            t_conv2 = t_features['conv2'].detach()
            t_conv3 = t_features['conv3'].detach()
            t_conv4 = t_features['conv4'].detach()
            t_conv5 = t_features['conv5'].detach()

            # Calculate gradient && Backpropagate
            optimizer.zero_grad()

            # Network output
            student, s_features = net(x_low)

            s_conv1 = s_features['conv1']
            s_conv2 = s_features['conv2']
            s_conv3 = s_features['conv3']
            s_conv4 = s_features['conv4']
            s_conv5 = s_features['conv5']

            KD_loss = nn.KLDivLoss()(F.log_softmax(student / temperature, dim=1),
                                  F.softmax(teacher / temperature, dim=1))

            KD_loss = torch.mul(KD_loss, temperature * temperature)

            GT_loss = ce_loss(student, y)

            GRAM_loss = .0

            if hint == False:
                if 1 in gram_features:
                    if at_enabled:
                        GRAM_loss += calculate_attendedGram_loss(s_conv1, t_conv1, norm_type,
                                                                 style_weight, mse_loss, at_ratio)
                    else:
                        GRAM_loss += calculate_Gram_loss(s_conv1, t_conv1, norm_type, patch_num,
                                                         style_weight, mse_loss, at_ratio)
                if 2 in gram_features:
                    if at_enabled:
                        GRAM_loss += calculate_attendedGram_loss(s_conv2, t_conv2, norm_type,
                                                                 style_weight, mse_loss, at_ratio)
                    else:
                        GRAM_loss += calculate_Gram_loss(s_conv2, t_conv2, norm_type, patch_num,
                                                         style_weight, mse_loss, at_ratio)
                if 3 in gram_features:
                    if at_enabled:
                        GRAM_loss += calculate_attendedGram_loss(s_conv3, t_conv3, norm_type,
                                                                 style_weight, mse_loss, at_ratio)
                    else:
                        GRAM_loss += calculate_Gram_loss(s_conv3, t_conv3, norm_type, patch_num,
                                                         style_weight, mse_loss, at_ratio)
                if 4 in gram_features:
                    if at_enabled:
                        GRAM_loss += calculate_attendedGram_loss(s_conv4, t_conv4, norm_type,
                                                                 style_weight, mse_loss, at_ratio)
                    else:
                        GRAM_loss += calculate_Gram_loss(s_conv4, t_conv4, norm_type, patch_num,
                                                         style_weight, mse_loss, at_ratio)
                if 5 in gram_features:
                    if at_enabled:
                        GRAM_loss += calculate_attendedGram_loss(s_conv5, t_conv5, norm_type,
                                                                 style_weight, mse_loss, at_ratio)
                    else:
                        GRAM_loss += calculate_Gram_loss(s_conv5, t_conv5, norm_type, patch_num,
                                                         style_weight, mse_loss, at_ratio)

                GRAM_loss /= len(gram_features)

            loss = KD_loss + GT_loss + GRAM_loss

            if loss == float('inf') or loss != loss:
                print('Loss is infinity, stop!')
                return

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
            output, _ = net(x_low)

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
            output, _ = net(x_low)

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
