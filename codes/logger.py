import time
import datetime
import os


def getlogger(loggername):
    filename =  loggername + '_' + datetime.datetime.now().strftime('D%d,%H:%M')
    from colorlog import ColoredFormatter   # pip install colorlog
    import logging

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
                'DEBUG': 'white,bold',
                'ERROR': 'red',
                'CRITICAL': 'red'
            }
        })
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    ch.setLevel(logging.DEBUG)

    fh = logging.FileHandler(filename + '.log')

    log = logging.getLogger(__name__)
    log.setLevel(logging.DEBUG)
    log.handlers = []       # No duplicated handlers
    log.propagate = False   # workaround for duplicated logs in ipython
    log.addHandler(ch)
    log.addHandler(fh)

    return log
