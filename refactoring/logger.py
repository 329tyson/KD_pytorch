import time
import datetime
import os
from colorlog import ColoredFormatter   # pip install colorlog
import tqdm
import logging


class AverageMeter(object):
    def __init__(self, name):
        self.reset()
        self.name = name

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def getname(self):
        return self.name

class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.DEBUG):
        logging.Handler.__init__(self)

    def emit(self, record):
        msg = self.format(record)
        tqdm.tqdm.write(msg)
        self.flush()

class Logger(object):
    def __init__(self, name, level = logging.DEBUG, is_test = False, stdout = True):
        # super(Logger, self).__init__()
        self.name= name
        self.level = level
        self.is_test= is_test
        self.stdout = stdout
        self.logger = self.create_logger()

    def create_logger(self):
        filename =  self.name + '_' + datetime.datetime.now().strftime('D%d,%H:%M')

        formatter = ColoredFormatter(
            "%(log_color)s[%(asctime)s] %(message_log_color)s%(message)s",
            datefmt=None, reset=True, style='%',
            log_colors={
                'DEBUG':    'cyan',
                'INFO':     'white,bold',
                'INFOV':    'cyan,bold',
                'WARNING':  'yellow',
                'ERROR':    'red,bold',
                'CRITICAL': 'red,bg_white',
            },
            secondary_log_colors={
                'message': {
                    'DEBUG': 'white',
                    'ERROR': 'red',
                    'CRITICAL': 'red'
                }
            })
        tqdm_handler = TqdmLoggingHandler()
        tqdm_handler.setFormatter(formatter)
        tqdm_handler.setLevel(self.level)

        log = logging.getLogger(__name__)
        log.setLevel(self.level)
        log.handlers = []       # No duplicated handlers
        log.propagate = False   # workaround for duplicated logs in ipython

        if self.stdout is True:
            log.addHandler(tqdm_handler)

        if self.is_test is False:
            fh = logging.FileHandler(filename + '.log')
            log.addHandler(fh)
            Cmd = 'tail -f ' + filename + '.log'
            title = filename + '.log'
            title = title.split('/')[-1]
            os.system('tmux split-window -h "{}"'.format(Cmd))
            os.system('tmux select-pane -T "{}"'.format(title))

        return log

    def message(self, str):
        self.logger.debug(str)

    def iteration(self, losses, **kwargs):
        s = ''
        s += '[EPOCH' + ' : ' + str(kwargs['EPOCH']) + ']'
        kwargs.pop('EPOCH', None)
        for k,v in kwargs.items():
            s += '[' + k + ' : ' + str(v) + ']'

        for elem in losses:
            s += '[' + elem.getname() + ' : ' + '{:.5f}'.format(elem.avg) + ']'
        self.logger.debug(s)

    def write_args(self, args):
        for arg in vars(args):
            self.message('\t{} - {}'.format(str(arg), str(getattr(args, arg))))

def display_function_stack(function):
    def wrapper(*args, **kwargs):
        logger = kwargs['logger']
        logger.message('[EXECUTION STACK- {}]'.format(function.__name__))
        # logger.message('=========================================================================================')
        # for arg in args:
            # logger.message('\t{}'.format(arg))
        # logger.message('=========================================================================================')
        for k,v in kwargs.items():
            logger.message('\t{} - {}'.format(k,v))
        logger.message('=========================================================================================')
        return function(*args, **kwargs)
    return wrapper

