import logging
import os

import torch.cuda


def check_gpus():
    """
        GPU available check
        https://pytorch-cn.readthedocs.io/zh/latest/package_references/torch-cuda/
    """
    logger = logging.getLogger()
    if not torch.cuda.is_available():
        logger.warning('This script could only be used to manage NVIDIA GPUs,but no GPU found in your device')
        return False
    elif 'NVIDIA System Management' not in os.popen('nvidia-smi -h').read():
        logger.warning('nvidia-smi tool not found.')
        return False
    else:
        return True


def parse(line, qargs):
    """
        Parsing a line of csv format text returned by nvidia-smi
        解析一行nvidia-smi返回的csv格式文本

        line:
            a line of text
        qargs:
            query arguments
        return:
            a dict of gpu infos
    """
    numeric_args = ['memory.free', 'memory.total']  # 可计数的参数

    def power_manage_enable(_):
        """
            显卡是否滋瓷power management（笔记本可能不滋瓷）
        """
        return 'Not Support' not in _

    def to_numeric(_):
        """
            带单位字符串去掉单位
        """
        return float(_.upper().strip().replace('MIB', '').replace('W', ''))

    def process(k, v):
        if k in numeric_args:
            if power_manage_enable(v):
                return int(to_numeric(v))
            else:
                return 1
        else:
            return v.strip()

    return {k: process(k, v) for k, v in zip(qargs, line.strip().split(','))}


def query_gpu(qargs=None):
    """
        Query GPUs infos

        qargs:
            query arguments
        return:
            a list of dict
    """
    if qargs is None:
        qargs = []
    qargs = ['index', 'gpu_name', 'memory.free', 'memory.total', 'power.draw', 'power.limit'] + qargs
    cmd = 'nvidia-smi --query-gpu={} --format=csv,noheader'.format(','.join(qargs))
    results = os.popen(cmd).readlines()
    return [parse(line, qargs) for line in results]


def _sort_by_memory(gpus):
    return sorted(gpus, key=lambda d: d['memory.free'], reverse=True)


def _sort_by_memory_rate(gpus):
    return sorted(gpus, key=lambda d: float(d['memory.free']) / d['memory.total'], reverse=True)


def _sort_by_power(gpus):
    logger = logging.getLogger()

    def by_power(d):
        """ helper function fo sorting gpus by power """
        power_infos = (d['power.draw'], d['power.limit'])
        if any(v == 1 for v in power_infos):
            logger.warning('Power management unable for GPU {}'.format(d['index']))
            return 1
        return float(d['power.draw']) / d['power.limit']

    return sorted(gpus, key=by_power)


def _sort_by_custom(gpus, key, reverse=False, qargs=None):
    if qargs is None:
        qargs = []
    if isinstance(key, str) and (key in qargs):
        return sorted(gpus, key=lambda d: d[key], reverse=reverse)
    if isinstance(key, type(lambda a: a)):
        return sorted(gpus, key=key, reverse=reverse)
    raise ValueError(
        "The argument 'key' must be a function or a key in query args,please read the documentation of nvidia-smi")


class GPUManager:
    """
        A manager which can list all available GPU devices
        and sort them and choose the most free one.
        Unspecified ones pref.
        GPU设备管理器,考虑列举出所有可用GPU设备,并加以排序,自动选出
        最空闲的设备。在一个GPUManager对象内会记录每个GPU是否已被指定,
        优先选择未指定的GPU。
    """

    def __init__(self, qargs=None):
        assert check_gpus(), "GPU available check failed."
        if qargs is None:
            qargs = []
        self.qargs = qargs
        self.gpus = query_gpu(qargs)
        for gpu in self.gpus:
            gpu['specified'] = False
        self.gpu_num = len(self.gpus)

    def auto_choice(self, mode=0):
        """
            Auto choice the freest GPU device,not specified ones
            自动选择最空闲GPU,返回索引

            mode:
                0: (default) sorted by free memory size
            return:
                a TF device object
        """
        for old_infos, new_infos in zip(self.gpus, query_gpu(self.qargs)):
            old_infos.update(new_infos)
        unspecified_gpus = [gpu for gpu in self.gpus if not gpu['specified']] or self.gpus

        logger = logging.getLogger()
        if mode == 0:
            logger.info('Choosing the GPU device has largest free memory...')
            chosen_gpu = _sort_by_memory(unspecified_gpus)[0]
        elif mode == 1:
            logger.info('Choosing the GPU device has highest free memory rate...')
            chosen_gpu = _sort_by_memory_rate(unspecified_gpus)[0]
        elif mode == 2:
            logger.info('Choosing the GPU device by power...')
            chosen_gpu = _sort_by_power(unspecified_gpus)[0]
        else:
            logger.info('Default: Choosing the GPU device has largest free memory rate...')
            chosen_gpu = _sort_by_memory_rate(unspecified_gpus)[0]

        chosen_gpu['specified'] = True
        index = chosen_gpu['index']
        logger.info('Using GPU {}'.format(index))
        logger.info('Memory Usage: {}/{}'.format(chosen_gpu['memory.total'] - chosen_gpu['memory.free'],
                                                 chosen_gpu['memory.total']))
        return int(index)
