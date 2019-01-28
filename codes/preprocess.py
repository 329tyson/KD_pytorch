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
                converted[lname+".weight"] = torch.from_numpy(val[0].transpose(3,2,0,1))
                converted[lname+".bias"] = torch.from_numpy(val[1])
            elif 'fc8' in lname:
                continue
            elif 'fc' in lname:
                converted[lname+".weight"] = torch.from_numpy(val[0].transpose(1,0))
                converted[lname+".bias"] = torch.from_numpy(val[1])
        net.load_state_dict(converted, strict = fit)
        net.cuda()
    else:
        weight = torch.load(pretrained_path)
        net.load_state_dict(weight, strict = fit)
        net.cuda()


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

