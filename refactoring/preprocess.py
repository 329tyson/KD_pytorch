import numpy as np
import torch
from dataloader import Dataset
from torch.utils import data
def load_weight(net, pretrained_path, fit=True, shared = False, ratios = []):
    if pretrained_path == 'NONE':
        print('\tloading weight from bvlc_alexnet.npy')
        pretrained= np.load('./models/bvlc_alexnet.npy', encoding='latin1').item()
        converted = net.state_dict()
        for lname, val in pretrained.items():
            if 'conv' in lname:
                # converted[lname+".module.weight"] = torch.from_numpy(val[0].transpose(3,2,0,1))
                # converted[lname+".module.bias"] = torch.from_numpy(val[1])
                converted[lname+".weight"] = torch.from_numpy(val[0].transpose(3,2,0,1))
                converted[lname+".bias"] = torch.from_numpy(val[1])
            elif 'fc8' in lname:
                continue
            elif 'fc' in lname:
                converted[lname+".weight"] = torch.from_numpy(val[0].transpose(1,0))
                converted[lname+".bias"] = torch.from_numpy(val[1])
                # converted[lname+".module.weight"] = torch.from_numpy(val[0].transpose(1,0))
                # converted[lname+".module.bias"] = torch.from_numpy(val[1])
        net.load_state_dict(converted, strict = fit)
    else:
        # Load weights on shraed alexnet
        weight = torch.load(pretrained_path)
        current = net.state_dict()
        hr_str = 'convs_hr'
        lr_str = 'convs_lr'
        shared_str = 'convs_shared'
        ratios = [float(1 - ratio) for ratio in ratios]
        group_convolution_layers = ['2','4','5']
        if shared is True:
            for k, v in weight.items():
                if any(gcl in k for gcl in group_convolution_layers):
                    hr = '.'.join([hr_str, str(int(k.split('.')[0][4]) - 1)])
                    lr = '.'.join([lr_str, str(int(k.split('.')[0][4]) - 1)])
                    shared = '.'.join([shared_str, str(int(k.split('.')[0][4]) - 1)])

                    indexes = [len(v) * 0.25, len(v) * 0.5, len(v) * 0.75, len(v)]
                    indexes = [int(index) for index in indexes]
                    blocks = []
                    blocks.append(v[:indexes[0]])
                    blocks.append(v[indexes[0]:indexes[1]])
                    blocks.append(v[indexes[1]:indexes[2]])
                    blocks.append(v[indexes[2]:])

                    if 'weight' in k:
                        current[hr+'.weight'] = torch.cat((blocks[0], blocks[3]), 0)
                        current[lr+'.weight'] = torch.cat((blocks[0], blocks[3]), 0)
                        current[shared+'.weight'] = torch.cat((blocks[1], blocks[2]), 0)
                    elif 'bias' in k:
                        current[hr+'.bias'] = torch.cat((blocks[0], blocks[3]), 0)
                        current[lr+'.bias'] = torch.cat((blocks[0], blocks[3]), 0)
                        current[shared+'.bias'] = torch.cat((blocks[1], blocks[2]), 0)
                elif 'conv' in k :
                    hr = '.'.join([hr_str, str(int(k.split('.')[0][4]) - 1)])
                    lr = '.'.join([lr_str, str(int(k.split('.')[0][4]) - 1)])
                    shared = '.'.join([shared_str, str(int(k.split('.')[0][4]) - 1)])

                    indep = v[:int(len(v) * ratios[int(k.split('.')[0][4]) - 1])]
                    depen = v[int(len(v) * ratios[int(k.split('.')[0][4]) - 1]):]
                    # print 'indep.shape : {} depen.shape : {}'.format(indep.shape, depen.shape)
                    # print 'indep.type :  {} depen.type  : {}'.format(type(indep), type(depen))
                    # print 'weight : {} weight.shape : {}'.format(weight[k], weight[k].shape)
                    # print 'indep + depen : {}, sum shape : {}'.format(torch.cat((indep, depen), 0), torch.cat((indep, depen), 0).shape)
                    # import ipdb; ipdb.set_trace()

                    if 'weight' in k:
                        current[hr + '.weight'] = indep.clone()
                        current[lr + '.weight'] = indep.clone()
                        current[shared + '.weight'] = depen.clone()
                    elif 'bias' in k:
                        current[hr + '.bias'] = indep.clone()
                        current[lr + '.bias'] = indep.clone()
                        current[shared + '.bias'] = depen.clone()
                elif 'fc' in k:
                    lr = k.split('.')[0] + '_lr'
                    hr = k.split('.')[0] + '_hr'

                    if 'weight' in k:
                        current[hr + '.weight'] = v.clone()
                        current[lr + '.weight'] = v.clone()
                    elif 'bias' in k:
                        current[hr + '.bias'] = v.clone()
                        current[lr + '.bias'] = v.clone()
                else:
                    print 'Error loading weights !'
                    exit(0)
            net.load_state_dict(current, strict = True)
        else:
            net.load_state_dict(weight, strict = True)
    net.cuda()

def generate_dataset(
    dataset,
    batch_size,
    annotation_train,
    annotation_val,
    image_path,
    low_ratio,
    is_KD = False,
    is_fitnet = False,
    ):
    # Training Params
    params = {'batch_size': batch_size,
              'shuffle': True,
              'num_workers': 6,
              'drop_last' : True}
    # PARAMS FOR TESTING
    # params = {'batch_size': 1,
              # 'shuffle': False,
              # 'num_workers': 1,
              # 'drop_last' : True}

    eval_params = {'batch_size': 1,
               'shuffle': True,
               'num_workers': 6,
               'drop_last' : True}

    is_KD = is_KD | is_fitnet
    if dataset.lower() == 'cub':
        #generate CUB datasets
        print('\t generating CUB dataset')
        training_set       = Dataset(dataset, annotation_train, image_path, low_ratio, 'Train', KD_flag = is_KD)
        eval_trainset      = Dataset(dataset, annotation_train, image_path, low_ratio, 'Validation')
        eval_validationset = Dataset(dataset, annotation_val, image_path, low_ratio, 'Validation')


        training_generator           = data.DataLoader(training_set, **params)
        eval_trainset_generator      = data.DataLoader(eval_trainset, **eval_params)
        eval_validationset_generator = data.DataLoader(eval_validationset, **eval_params)

    elif dataset.lower() == 'stanford':
        #generate stanford car datasets
        print('\t generating STANFORD dataset')
        training_set = Dataset(dataset, annotation_train, image_path, low_ratio, 'Train', False, is_KD)
        eval_trainset = Dataset(dataset, annotation_train, image_path, low_ratio, 'Validation', ten_crop)
        eval_validationset = Dataset(dataset, annotation_val, image_path, low_ratio, 'Validation', ten_crop)

        num_training = len(training_set)
        num_validation = len(eval_validationset)

        training_generator = data.DataLoader(training_set, **params)
        eval_trainset_generator = data.DataLoader(eval_trainset, **eval_params)
        eval_validationset_generator = data.DataLoader(eval_validationset, **eval_params)

    else:
        raise ValueError

    return [training_generator, eval_trainset_generator, eval_validationset_generator]

