import argparse
import os


def parse():
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
        default='data',
        type=str,
        help='Directory to data')
    parser.add_argument(
        '--annotation_val',
        default='annotations',
        type=str,
        help='Annotation file path')
    parser.add_argument(
        '--annotation_train',
        default='annotations',
        type=str,
        help='Annotation file path')
    parser.add_argument(
        '--result',
        default='results',
        type=str,
        help='Result directory path')
    parser.add_argument(
        '--dataset',
        default='None',
        type=str,
        help='dataset to use')
    parser.add_argument(
        '--classes',
        default=0,
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
        '--batch', default=111, type=int, help='Batch Size')
    parser.add_argument(
        '--epochs',
        default=100,
        type=int,
        help='Number of total epochs to run')
    parser.add_argument(
        '--resume',
        default=False,
        type=bool,
        help='Boolean variable for resume training, default to False')
    parser.add_argument(
        '--pretrain_path', default='NONE', type=str, help='Pretrained model (.pth)')
    parser.add_argument(
        '--checkpoint',
        default=10,
        type=int,
        help='Trained model is saved at every this epochs.')
    parser.add_argument(
        '--low_ratio',
        default=0,
        choices=[50, 25],
        type=int,
        help='low resolution ratio, default to 0'
    )
    parser.add_argument(
        '--verbose',
        '--v',
        action='store_true',
    )
    parser.add_argument(
        '--kd_enabled',
        action='store_true',
        help='if true, KD training is executed, and it requires pretrained model to be used as teachernet'
    )
    parser.add_argument(
        '--kd_temperature',
        default=3,
        type=int,
        help='distillation temperature, need to be set if you train KD model')
    parser.add_argument(
        '--ten_batch_eval',
        action='store_false',
        help='if true, ten batch mean evaluation is performed'
    )
    parser.add_argument(
        '--log_dir',
        default='./logs',
        help='Log directory, set to default as ./logs'
    )
    parser.add_argument(
        '--gpu',
        default=0,
        type=int,
        help='Use specified gpu, not allowing multi-gpu')
    parser.add_argument(
        '--style_weight',
        default=1.0,
        type=float
    )
    parser.add_argument(
        '--gram_enabled',
        action='store_true',
        help='if true, 1st is trained using Gram loss and KD in 2nd stage'
    )
    parser.add_argument(
        '--norm_type',
        default=0,
        type = int,
        help='Normalization type of feature map for gram matrix (0:no, 1:v/[h*w], 2:l2 vec, 3:mat/[h*w], 4:mat/[c*h*w])'
    )
    parser.add_argument(
        '--patch_num',
        default=1,
        type=int
    )
    # parser.add_argument('--gram_features', nargs='+', type=int)
    parser.add_argument(
        '--gram_features',
        default = None,
        help = 'convnets to be used in mse loss'
    )
    parser.add_argument(
        '--hint',
        action='store_true'
    )
    parser.add_argument(
        '--save',
        action='store_true'
    )
    parser.add_argument(
        '--vgg_gap',
        action='store_true'
    )
    parser.add_argument(
        '--at_ratio',
        type=float,
        default=1
    )
    parser.add_argument(
        '--at_enabled',
        action='store_true'
    )
    parser.add_argument(
        '--message',
        type=str,
        default = 'no settings specified',
        help = 'short description for this experiment'
    )
    parser.set_defaults(bn=False)
    parser.set_defaults(ten_batch_eval=True)
    parser.set_defaults(kd_enabled=False)
    parser.set_defaults(gram_enabled=False)
    parser.set_defaults(hint=False)
    parser.set_defaults(verbose=False)
    parser.set_defaults(save=False)
    parser.set_defaults(vgg_gap=False)
    parser.set_defaults(at_enabled=False)

    args = parser.parse_args()

    return args
