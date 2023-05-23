"""
    【Grid Search】(Optional)
    Before training, one can run grid search for the best hyper-parameters.
    The hyper-parameters you would like to adjust are needed to set in the code.
    Then, save the best ones in `models/model_config.py` file.

    `python grid_search.py --log True --data XXX --model_type gaussian`(default)
    `python grid_search.py --log True --data XXX --model_type nn`
"""
import itertools
import time

import pandas as pd
import torch
from torch import nn, optim
from torch.optim import lr_scheduler

from datasets.regdata import build_reg_loader
from models.model_utils import load_model, train_model, test_model, test_model_acc
from utils.utils_file import generate_model_filename
from utils.utils_parser import DefaultArgumentParser, init_config, report_args

if __name__ == '__main__':
    start_time = time.time()
    parser = DefaultArgumentParser().get_parser()

    # model settings
    parser.add_argument('--model_nums', default=3, type=int, help='Number of bnn to ensemble')
    parser.add_argument('--model_type', default='gaussian', type=str, help='Variation inference family')
    parser.add_argument('--models_struct', default=None, type=list,
                        help='Hidden layers for each model. If None, it is set in `init_config`')

    parser.add_argument('--activation', default='relu', type=str, help='activation function')
    parser.add_argument('--rep', default=40, type=int, help='repetition times when predicting')
    parser.add_argument('--sigma_pi', default=0.1, type=float, help='Gaussian prior')
    parser.add_argument('--sigma_start', default=0.1, type=float, help='Initial std of posterior')

    # train settings
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--epochs', default=20000, type=int, help='max train epochs')
    parser.add_argument('--monitor', default='loss', type=str, choices=['loss', 'val_loss'],
                        help='judge overfit by monitor when training')
    parser.add_argument('--patience', default=40, type=int, help='loss not decrease after patience epochs;'
                                                                 'If -1, not EarlyStopping')

    parser.add_argument('--lr', default=0.01, type=float, help='learning rate of optimizer')
    parser.add_argument('--lr_steps', default=100, type=int, help='decay lr every lr_steps epochs, e.g 50-gaussian')
    parser.add_argument('--lr_gamma', default=0.5, type=float, help='decay lr by a factor of lr_gamma')
    parser.add_argument('--beta', default=0, type=float, help='elbo: beta * kl, actually y~N(yhat, beta^2)')

    opt = parser.parse_args()
    opt.exp_name = 'grid_search'
    init_config(opt)

    dataset, dataloader = build_reg_loader(opt, shuffle=True)

    model_names = [f'{opt.model_type}_{i}' for i in range(1, opt.model_nums + 1)]

    neurons, depths = [20], [3]
    grid_models_struct = [[neuron for _ in range(depth)] for neuron, depth in itertools.product(neurons, depths)]
    grid_lr = [0.001, 0.01, 0.05, 0.1]
    grid_lr_steps = [30, 100, 10000]
    grid_lr_gamma = [0.1, 0.5]
    if opt.model_type == 'gaussian':
        grid_beta = [0, 0.01, 0.05, 0.1]
    else:
        grid_beta = [0, 0.01, 0.005, 0.001]
    if opt.model_type == 'gaussian':
        grid_sigma = [0.01, 0.05, 0.1, 0.5, 1]  # if nn, ignore this
    else:
        grid_sigma = [0]

    print(f'grid_models_struct: {grid_models_struct}')
    print(f'grid_lr: {grid_lr}')
    print(f'grid_lr_steps: {grid_lr_steps}')
    print(f'grid_lr_gamma: {grid_lr_gamma}')
    print(f'grid_beta: {grid_beta}')
    print(f'grid_sigma: {grid_sigma}')

    for models_struct in grid_models_struct:
        for lr in grid_lr:
            for lr_steps in grid_lr_steps:
                for lr_gamma in grid_lr_gamma:
                    for sigma in grid_sigma:
                        for beta in grid_beta:
                            opt.models_struct = models_struct
                            opt.lr = lr
                            opt.lr_steps = lr_steps
                            opt.lr_gamma = lr_gamma
                            opt.beta = beta
                            opt.sigma_pi = sigma
                            opt.sigma_start = sigma
                            print('---------------------- Grid Search Start ----------------------')
                            print(f'models_struct: {models_struct}, lr: {lr}, lr_steps: {lr_steps}, '
                                  f'lr_gamma: {lr_gamma}, beta: {beta}, sigma: {sigma}')

                            histories, losses = [], []
                            for model_name in model_names:
                                model = load_model(model_name, opt)

                                if opt.data in ['mnist']:
                                    criterion = nn.CrossEntropyLoss()
                                else:
                                    criterion = nn.MSELoss()

                                if opt.model_type == 'nn':
                                    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=beta)
                                    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_steps,
                                                                           gamma=opt.lr_gamma)
                                    histories.append(
                                        train_model(opt, model, dataloader, criterion, optimizer, exp_lr_scheduler,
                                                    beta=None, train_log=False,
                                                    save_file=generate_model_filename(opt, model_name, False)))
                                elif opt.model_type == 'gaussian':
                                    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
                                    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_steps,
                                                                           gamma=opt.lr_gamma)
                                    histories.append(
                                        train_model(opt, model, dataloader, criterion, optimizer, exp_lr_scheduler,
                                                    beta=2.0 * opt.beta ** 2, train_log=False,
                                                    save_file=generate_model_filename(opt, model_name, False)))
                                else:
                                    raise NotImplementedError(f'No such train model type of {opt.model_type}')

                                model = load_model(model_name, opt, resume=True, last=False)
                                losses.append(test_model(model, dataloader, criterion, opt))
                                if opt.data in ['mnist']:
                                    testset, testloader = build_reg_loader(opt, shuffle=False, train=False)
                                    correct, total, acc = test_model_acc(model, testloader, opt)
                                    print(f'Test acc={acc:.4f}({correct}/{total})')

                            print('Analysing training results...')
                            df = pd.DataFrame(
                                index=model_names, columns=['loss', 'test_loss'], data=[
                                    [histories[i]['loss'][-1], losses[i]] for i in range(opt.model_nums)
                                ])
                            print(df)

                            print('---------------------- Grid Search End ----------------------')

    end_time = time.time()
    elapse_time = end_time - start_time
    print(f'All end  in {elapse_time // 60:.0f}m {elapse_time % 60:.0f}s.')
