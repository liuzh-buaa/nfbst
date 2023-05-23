import datetime
import logging
import os.path
import sys


class Log(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.logger = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.logger.write(message)

    def flush(self):
        pass

    def close(self):
        self.logger.close()


def get_local_time():
    cur = datetime.datetime.now()
    cur = cur.strftime("%Y-%m-%d_%H-%M-%S")
    return cur


def get_logger(log_dir, log_filename=None, name=None):
    if log_filename is None:
        log_filename = '{}.log'.format(get_local_time())
    _log_filepath = os.path.join(log_dir, log_filename)

    # 日志级别，logger 和 handler以最高级别为准，不同handler之间可以不一样，不相互影响
    _logger = logging.getLogger(name)
    _logger.setLevel(logging.DEBUG)
    _formatter = logging.Formatter(fmt="[%(asctime)s|%(filename)s|%(levelname)s] %(message)s",
                                   datefmt="%Y-%m-%d %H:%M:%S")

    # StreamHandler
    _sHandler = logging.StreamHandler()
    _sHandler.setLevel(logging.INFO)
    _sHandler.setFormatter(_formatter)

    # FileHandler
    _fHandler = logging.FileHandler(_log_filepath, mode='w')
    _fHandler.setLevel(logging.DEBUG)
    _fHandler.setFormatter(_formatter)

    _logger.addHandler(_sHandler)
    _logger.addHandler(_fHandler)

    return _logger


def init_config_log(opt):
    opt.data_root = f'{opt.root}/data/{opt.data}'
    opt.log_dir = f'{opt.data_root}/log/{opt.exp_name}/{opt.timestamp}'

    opt.data_dir = f'{opt.data_root}/data'
    opt.results_dir = f'{opt.data_root}/results'

    for directory in [opt.log_dir, opt.data_dir, opt.results_dir]:
        if not os.path.isdir(directory):
            os.makedirs(directory)

    log_filename = None
    if opt.exp_name in ['train_model']:
        log_filename = f'{opt.model_type}.log'
    elif opt.exp_name in ['get_statistic_distri', 'ensemble_statistic_distri']:
        log_filename = f'{opt.model_name}_{opt.interpret_method}_{opt.y_index}.log'
    elif opt.exp_name in ['get_bayes_factors', 'analyse_bayes_factors', 'ensemble_bayes_factors', 'local_2_global']:
        log_filename = f'{opt.model_name}_{opt.interpret_method}_{opt.algorithm}_{opt.y_index}.log'

    opt.logger = get_logger(opt.log_dir, log_filename)


if __name__ == '__main__':
    logger = get_logger('./', 'test_log.txt')

    var = 1
    logger.debug(f'var={var}')
    logger.info(f'var={var}')
    logger.warning(f'var={var}')
