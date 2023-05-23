"""
    【Distribution of Testing Statistics Generation Process】
    
    python get_statistic_distri.py --log True --data XXX --model_type gaussian --model_name gaussian_1/gaussian_2/gaussian_3 --interpret_method gradient/DeepLIFT/LRP/LIME --y_index 0
    python get_statistic_distri.py --log True --data XXX --model_type nn --sample_T 3 --model_name nn_1 --interpret_method gradient/DeepLIFT/LRP/LIME --y_index 0

    【Distribution of GradientXInput Generation Process】(Optional)
    The results are similar to LRP.
    [Kindermans, Investigating the influence of noise and distractors on the interpretation of neural networks.] shows that
    the LRP rules for ReLU networks are equivalent within a scaling factor to gradient × input in some conditions.

    python get_statistic_distri.py --log True --data XXX --model_type gaussian --model_name gaussian_1/gaussian_2/gaussian_3 --interpret_method gradientXinput --y_index 0
    python get_statistic_distri.py --log True --data XXX --model_type nn --sample_T 3 --model_name nn_1 --interpret_method gradientXinput --y_index 0
"""
import shutil
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets.regdata import build_reg_loader
from models.model_utils import load_model, test_model, test_model_acc
from utils.utils_file import generate_statistic_sample_filename, generate_statistic_distri_filename
from utils.utils_parser import DefaultArgumentParser, init_config


def get_statistic(_model, _dataloader, _opt):
    if _opt.interpret_method == 'LIME':
        statistic = [_model.get_interpret(inputs.to(_opt.device), _opt, n_samples=_opt.lime_samples) for inputs, _ in dataloader]
    else:
        statistic = [_model.get_interpret(inputs.to(_opt.device), _opt) for inputs, _ in dataloader]
    statistic = torch.cat(statistic, dim=0)
    return statistic


if __name__ == '__main__':
    start_time = time.time()
    parser = DefaultArgumentParser().get_parser()

    # model settings
    parser.add_argument('--model_type', default='gaussian', type=str, help='Variation inference family')
    parser.add_argument('--model_name', default='gaussian_1', type=str, help='choose which model to get distri')

    parser.add_argument('--activation', default='relu', type=str, help='activation function')
    parser.add_argument('--rep', default=40, type=int, help='repetition times when predicting')
    parser.add_argument('--sigma_pi', default=0.1, type=float, help='Gaussian prior')
    parser.add_argument('--sigma_start', default=0.1, type=float, help='Initial std of posterior')

    # train settings
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')

    # gradient
    parser.add_argument('--interpret_method', default='gradient', type=str, help='testing statistic',
                        choices=['gradient', 'Saliency', 'DeepLIFT', 'LRP', 'LIME', 'gradientXinput'])
    parser.add_argument('--sample_T', default=100, type=int, help='the number of samples to build distri')
    parser.add_argument('--y_index', default=0, type=int, help='gradient to which output (for multi-outputs)')
    parser.add_argument('--indices', default=None, type=int, help='dataset indices')
    parser.add_argument('--lime_samples', default=100, type=int, help='lime samples')

    opt = parser.parse_args()
    opt.exp_name = 'get_statistic_distri'
    if opt.interpret_method == 'LIME':
        opt.batch_size = 1  # /home/liuzh/miniconda3/envs/bayes/lib/python3.8/site-packages/captum/attr/_core/lime.py:1101: UserWarning: You are providing multiple inputs for Lime / Kernel SHAP attributions. This trains a separate interpretable model for each example, which can be time consuming. It is recommended to compute attributions for one example at a time.
    init_config(opt, model_config=True)

    # Note: get the whole dataset, and its order must be fixed (shuffle=False)
    dataset, dataloader = build_reg_loader(opt, shuffle=False, train=False, indices=opt.indices)
    model = load_model(opt.model_name, opt, resume=True, last=True)

    if opt.n_targets > 1:
        criterion = nn.CrossEntropyLoss()
        loss = test_model(model, dataloader, criterion, opt)
        correct, total, acc = test_model_acc(model, dataloader, opt)
        opt.logger.info(f'Test loss={loss}, acc={acc:.4f}({correct}/{total})')
        opt.n_samples = len(dataset)
    else:
        criterion = nn.MSELoss()
        loss = test_model(model, dataloader, criterion, opt)
        opt.logger.info(f'Test loss={loss}')

    data = []
    if opt.interpret_method == 'gradientXinput':
        entire_dataloader = DataLoader(dataset, batch_size=len(dataset))
        entire_data = next(iter(entire_dataloader))[0].numpy()
        opt.interpret_method = 'gradient'
        grad_np_data = np.load(generate_statistic_sample_filename(opt, 'total', last=True))
        opt.interpret_method = 'gradientXinput'
        for i in range(opt.sample_T):
            opt.interpret_method = 'gradient'
            grad = grad_np_data[i]
            opt.interpret_method = 'gradientXinput'
            gradXinput = grad * entire_data
            np.save(generate_statistic_sample_filename(opt, i, last=False), gradXinput)
            data.append(gradXinput)
            if (i + 1) % 100 == 0:
                opt.logger.info(f'Finish getting sample{i} {opt.interpret_method} of {opt.model_name}: {gradXinput.shape}')
        np_data = np.stack(data, axis=0)
    else:
        for i in range(opt.sample_T):
            grad = get_statistic(model, dataloader, opt)
            np.save(generate_statistic_sample_filename(opt, i, last=False), grad.cpu().numpy())
            data.append(grad)
            if (i + 1) % 100 == 0:
                opt.logger.info(f'Finish getting sample{i} {opt.interpret_method} of {opt.model_name}: {grad.shape}')

        data = torch.stack(data, dim=0)
        np_data = data.cpu().numpy()  # (sample_T, n_data, **n_features)
    np.save(generate_statistic_sample_filename(opt, 'total'), np_data)

    for i in range(opt.n_samples):
        grad = np_data[:, i, :]  # (sample_T, **n_features)
        np.save(generate_statistic_distri_filename(opt, i, last=False), grad)
        if (i + 1) % 100 == 0:
            opt.logger.info(
                f'Finish getting data point{i} {opt.interpret_method} distri of {opt.model_name}: {grad.shape}')

    if opt.log:
        opt.logger.info(f'Copying total {opt.interpret_method} samples of {opt.model_name}...')
        shutil.copyfile(generate_statistic_sample_filename(opt, 'total', last=False),
                        generate_statistic_sample_filename(opt, 'total', last=True))

        opt.logger.info(f'Copying data {opt.interpret_method} distri of {opt.model_name}...')
        for k in range(opt.n_samples):
            shutil.copyfile(generate_statistic_distri_filename(opt, k, last=False),
                            generate_statistic_distri_filename(opt, k, last=True))

    end_time = time.time()
    elapse_time = end_time - start_time
    opt.logger.info(f'All end  in {elapse_time // 60:.0f}m {elapse_time % 60:.0f}s.')
