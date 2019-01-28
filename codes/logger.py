import time
import datetime
import logging
import os


def getlogger(loggername):
    filename =  loggername + '_' + datetime.datetime.now().strftime('D%d,%H:%M')
    logging.basicConfig(
        format='%(message)s',
        filename= filename+ '.log',
        filemode='w',
        level=logging.INFO
    )
    return logging.getLogger(filename)

