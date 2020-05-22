# Author: Ioannis Matzakos | Date: 22/07/2019

import logging
import sys


# Configure the logging system
def setup_logger(name):
    # setup logging format
    formatter = logging.Formatter(fmt='%(asctime)s | %(levelname)s | %(module)s : %(message)s')
    # display log in console
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    # save log in a file
    filehandler = logging.FileHandler(r'Logs/predict-covid19-in-greece-log.txt')
    filehandler.setFormatter(formatter)
    # setup logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.addHandler(filehandler)
    return logger
