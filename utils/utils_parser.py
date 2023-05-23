import argparse
import logging
import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from datasets.data_utils import init_config_data
from models.model_utils import init_model_config
from utils.utils_gpu import GPUManager
from utils.utils_log import init_config_log


class DefaultArgumentParser(object):
    def __init__(self):
        parser = argparse.ArgumentParser()

        # experiment general settings
        parser.add_argument('--root', default=os.path.join(os.path.dirname(__file__), '..'), type=str)
        parser.add_argument('--timestamp', default=datetime.now().strftime('%Y-%m-%d %H-%M-%S'), type=str)
        parser.add_argument('--gpu_id', default='-1', type=str, help='Which gpu to use; if -1, select automatically')
        parser.add_argument('--seed', default=None, type=int)

        # experiment specific settings
        parser.add_argument('--exp_name', default='None', type=str)
        parser.add_argument('--log', default=False, type=bool, help='Whether to pollute other files exclude log dir; '
                                                                    'If False, we do not care about exp_name')
        parser.add_argument('--data', default='simulation_v4', type=str)

        self.parser = parser

    def get_parser(self):
        return self.parser


def report_args(obj):
    logger = logging.getLogger()
    logger.info('------------ Options ------------')
    for key, val in vars(obj).items():
        logger.info('--{:24} {}'.format(key, val))
    logger.info('-------------- End --------------')


def init_config_seed(opt):
    if opt.seed is not None:
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.cuda.manual_seed_all(opt.seed)
        np.random.seed(opt.seed)
        random.seed(opt.seed)


def init_config_device(opt):
    if torch.cuda.is_available():
        if opt.gpu_id == '-1':
            gm = GPUManager()
            gpu_id = gm.auto_choice()
        else:
            gpu_id = opt.gpu_id
        opt.device = torch.device(f'cuda:{gpu_id}')
    else:
        opt.logger.info('Using CPU')
        opt.device = torch.device('cpu')


def init_config_pandas():
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)


def init_config(opt, model_config=False):
    init_config_log(opt)
    init_config_seed(opt)
    init_config_device(opt)
    init_config_data(opt)
    init_config_pandas()

    if model_config:
        init_model_config(opt)

    report_args(opt)
