import numpy as np
import torch
from dataloader import Dataset
from torch.utils import data
def load_weight(net, pretrained_path, fit=True):
    if pretrained_path == 'NONE':
        print('\tloading weight from bvlc_alexnet.npy')
        pretrained= np.load('bvlc_alexnet.npy', encoding='latin1').item()
        converted = net.state_dict()
        for lname, val in pretrained.items():
            if 'conv' in lname:
                # converted[lname+".modu.weight"] = torch.from_numpy(val[0].transpose(3,2,0,1))
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
        weight = torch.load(pretrained_path)

        # preserve initial residual adapter parameters
        ori_params = net.state_dict().copy()
        for i in ori_params:
            if 'res' in i:
                weight[i] = ori_params[i]
            if 'at' in i:
                weight[i] = ori_params[i]

        # delete unnecessary adapters parameters
        for i in weight:
            if 'adapters' in i:
                del weight[i]

        # teacher model doesn't have residual adpater
        if not any(net.residuals):
            for i in weight:
                if 'res' in i:
                    del weight[i]

        net.load_state_dict(weight, strict = fit)

def generate_dataset(
    dataset,
    batch_size,
    annotation_train,
    annotation_val,
    image_path,
    low_ratio,
    ten_crop,
    verbose,
    is_KD = False):
    # Training Params
    params = {'batch_size': batch_size,
              'shuffle': True,
              'num_workers': 6,
              'drop_last' : True}

    eval_params = {'batch_size': 1,
               'shuffle': True,
               'num_workers': 6,
               'drop_last' : True}
    if ten_crop is False:
        eval_params['batch_size'] = batch_size

    if dataset.lower() == 'cub':
        #generate CUB datasets
        print('\t generating CUB dataset')
        training_set = Dataset(dataset, annotation_train, image_path, low_ratio, 'Train', False,  is_KD)
        eval_trainset = Dataset(dataset, annotation_train, image_path, low_ratio, 'Train', ten_crop)
        eval_validationset = Dataset(dataset, annotation_val, image_path, low_ratio, 'Validation', ten_crop)

        num_training = len(training_set)
        num_validation = len(eval_validationset)

        training_generator = data.DataLoader(training_set, **params)
        eval_trainset_generator = data.DataLoader(eval_trainset, **eval_params)
        eval_validationset_generator = data.DataLoader(eval_validationset, **eval_params)

    elif dataset.lower() == 'stanford':
        #generate stanford car datasets
        print('\t generating STANFORD dataset')
        training_set = Dataset(dataset, annotation_train, image_path, low_ratio, 'Train', False, is_KD)
        eval_trainset = Dataset(dataset, annotation_train, image_path, low_ratio, 'Train', ten_crop)
        eval_validationset = Dataset(dataset, annotation_val, image_path, low_ratio, 'Validation', ten_crop)

        num_training = len(training_set)
        num_validation = len(eval_validationset)

        training_generator = data.DataLoader(training_set, **params)
        eval_trainset_generator = data.DataLoader(eval_trainset, **eval_params)
        eval_validationset_generator = data.DataLoader(eval_validationset, **eval_params)

    else:
        raise ValueError

    return training_generator, eval_trainset_generator, eval_validationset_generator, num_training, num_validation

