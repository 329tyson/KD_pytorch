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
        '--noise',
        action='store_true',
    )
    parser.set_defaults(noise=False)
    parser.set_defaults(ten_batch_eval=True)
    parser.set_defaults(kd_enabled=False)
    parser.set_defaults(verbose=False)


    args = parser.parse_args()

    return args
