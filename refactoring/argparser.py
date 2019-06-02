import argparse
import os


def parse():
    test_annot = '../TestImagelabels.csv'
    test_data  = '../TestImages/'
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--root',
        '--r',
        default=os.getcwd(),
        type=str,
        help='Root directory path of data, set to PWD as default')

    parser.add_argument(
        '--data',
        '--d',
        default='../CUB_200_2011/images/',
        type=str,
        help='Directory to data')

    parser.add_argument(
        '--annotation_val',
        default='../labels/label_val_cub200_2011.csv',
        type=str,
        help='Annotation file path')

    parser.add_argument(
        '--annotation_train',
        default='../labels/label_train_cub200_2011.csv',
        type=str,
        help='Annotation file path')

    parser.add_argument(
        '--dataset',
        default='cub',
        type=str,
        help='dataset to use')

    parser.add_argument(
        '--classes',
        default=200,
        type=int,
        help=
        'Number of classes'
    )

    parser.add_argument(
        '--lr',
        default=0.001,
        type=float,
        help=
        'Initial learning rate (divided by 10 while training by lr scheduler)')

    parser.add_argument(
        '--lr_decay',
        default=20,
        type=int,
        help='Learning rate decaying period'
    )

    parser.add_argument(
        '--batch', default=128, type=int, help='Batch Size')

    parser.add_argument(
        '--epochs',
        default=1000,
        type=int,
        help='Number of total epochs to run')

    parser.add_argument(
        '--pretrain_path', default='NONE', type=str, help='Pretrained model (.pth)')

    parser.add_argument(
        '--low_ratio',
        default=0,
        choices=[50, 25],
        type=int,
        help='low resolution ratio, default to 0'
    )

    parser.add_argument(
        '--kd',
        action='store_true',
        help='if true, KD training is executed, and it requires pretrained model to be used as teachernet'
    )

    parser.add_argument(
        '--kd_temperature',
        default=3,
        type=int,
        help='distillation temperature, need to be set if you train KD model')

    parser.add_argument(
        '--log_dir',
        default='./logs',
        help='Log directory, set to default as ./logs'
    )

    parser.add_argument(
        '--gpu',
        default=0,
        type=int,
        help='Use specified gpu')

    parser.add_argument(
        '--save',
        action='store_true'
    )
    parser.add_argument(
        '--message',
        type=str,
        default = 'no settings specified',
        help = 'short description for this experiment'
    )

    parser.add_argument(
        '--test',
        action='store_true'
    )

    parser.add_argument(
        '--val_period',
        type=int,
        default = 10
    )

    parser.add_argument(
        '--rg_layers',
        type=str,
        default = ''
    )

    parser.add_argument(
        '--grad_layers',
        type=str,
        default = ''
    )

    parser.set_defaults(kd=False)
    parser.set_defaults(save=False)
    parser.set_defaults(test=False)

    args = parser.parse_args()
    args.annotation_train = os.path.join(args.root, args.annotation_train) if args.test is False else os.path.join(args.root, test_annot)
    args.annotation_val   = os.path.join(args.root, args.annotation_val)   if args.test is False else os.path.join(args.root, test_annot)
    args.data             = os.path.join(args.root, args.data)             if args.test is False else os.path.join(args.root, test_data)
    args.log_dir          = os.path.join(args.root, args.log_dir)

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

    if args.pretrain_path != 'NONE':
        args.pretrain_path = os.path.join(args.root, args.pretrain_path)

    args.classes = 200 if args.dataset.lower() == 'cub' else 196

    return args
