import torch
import alexnet
from tqdm import tqdm
from preprocess import *
from save_gradient import calculate_gradCAM

glb_grad_at = {}


net = alexnet.AlexNet(0.5, 200, ['fc8'])
load_weight(net, './models/cub_teacher_68.pt')

train_loader, eval_train_loader, eval_validation_loader, num_training, num_validation = generate_dataset(
    'cub',
    1,
    '/home/tyson/cv/KD_pytorch/labels/label_train_cub200_2011.csv',
    '/home/tyson/cv/KD_pytorch/labels/label_val_cub200_2011.csv',
    '/home/tyson/cv/KD_pytorch/CUB_200_2011/images/',
    25,
    True,
    True,
    True)
net.cuda()
net.eval()

calculate_gradCAM(net, train_loader)
# for x, x_low, y in train_gen:
    # x = x.cuda().float()
    # y = y.cuda() -1

    # output, features = net(x)
    # one_hot_y = torch.zeros(output.shape).float().cuda()

    # for i in range(output.shape[0]):
        # one_hot_y[i][y[i]] = 1.0

    # output.backward(gradient = one_hot_y, retain_graph = True)
    # gCam = compute_gradCAM(output['conv5'], glb_grad_at[id(net.conv5)])
    # print glb_grad_at[id(net.conv5)]
