"""
    【Train Model】
    在对应数据集上训练对应的模型，训练过程以及对应的loss曲线保存在log文件夹内，同时将训练好的模型保存到results/model文件夹。

    `python train_model.py --log True --data XXX --model_type gaussian`(default)
    `python train_model.py --log True --data XXX --model_type nn`
"""
import shutil
import time

import numpy as np
import pandas as pd
from torch import nn, optim
from torch.optim import lr_scheduler

from datasets.regdata import build_reg_loader
from models.model_utils import train_model, test_model, load_model, test_model_acc
from utils.utils_file import generate_model_filename, generate_history_filename, generate_history_figname
from utils.utils_parser import DefaultArgumentParser, init_config
from utils.utils_plot import visualize_loss

if __name__ == '__main__':
    start_time = time.time()
    parser = DefaultArgumentParser().get_parser()

    # model settings
    parser.add_argument('--model_nums', default=10, type=int, help='Number of bnn to train, then select the best')
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
    opt.exp_name = 'train_model'
    init_config(opt, model_config=True)

    dataset, dataloader = build_reg_loader(opt, shuffle=True, train=True)
    testset, testloader = build_reg_loader(opt, shuffle=False, train=False)

    model_names = [f'{opt.model_type}_{i}' for i in range(1, opt.model_nums + 1)]

    histories, losses, acces = [], [], []
    for model_name in model_names:
        opt.logger.info(f'==> Training {model_name}...')
        model = load_model(model_name, opt)

        if opt.n_targets > 1:
            criterion = nn.CrossEntropyLoss()
            correct, total, acc = test_model_acc(model, testloader, opt)
            loss = test_model(model, testloader, criterion, opt)
            opt.logger.info(f'Test acc={acc:.4f}({correct}/{total}), loss={loss:.4f}')
        else:
            criterion = nn.MSELoss()
            loss = test_model(model, testloader, criterion, opt)
            opt.logger.info(f'Test loss={loss:.4f}')

        if opt.model_type == 'nn':
            optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.beta)
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_steps, gamma=opt.lr_gamma)
            histories.append(train_model(opt, model, dataloader, criterion, optimizer, exp_lr_scheduler,
                                         beta=None, save_file=generate_model_filename(opt, model_name, False)))
        elif opt.model_type == 'gaussian':
            optimizer = optim.Adam(model.parameters(), lr=opt.lr)
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_steps, gamma=opt.lr_gamma)
            histories.append(train_model(opt, model, dataloader, criterion, optimizer, exp_lr_scheduler,
                                         beta=2.0 * opt.beta ** 2,
                                         save_file=generate_model_filename(opt, model_name, False)))
        else:
            raise NotImplementedError(f'No such train model type of {opt.model_type}')

        model = load_model(model_name, opt, resume=True, last=False)
        losses.append(test_model(model, testloader, criterion, opt))
        opt.logger.info(f'Test loss: {losses[-1]}')

        if opt.n_targets > 1:
            correct, total, acc = test_model_acc(model, testloader, opt)
            acces.append(acc)
            opt.logger.info(f'Test acc={acc:.4f}({correct}/{total})')

        np.save(generate_history_filename(opt, model_name), histories[-1])
        visualize_loss(histories[-1], model_name, generate_history_figname(opt, model_name))

    opt.logger.info('Analysing training results...')
    df = pd.DataFrame(
        index=model_names, columns=['loss', 'test_loss'], data=[
            [histories[i]['loss'][-1], losses[i]] for i in range(opt.model_nums)
        ])
    opt.logger.info(df)

    if opt.log:
        opt.logger.info('Copying models from `timestamp` to `models`...')
        if opt.n_targets > 1:
            opt.logger.info('Sort all models based on test accuracy')
            sorted_ids = sorted(range(opt.model_nums), key=lambda k: acces[k], reverse=True)
            for seq_id, sorted_id in enumerate(sorted_ids):
                opt.logger.info(f'Copying {model_names[sorted_id]}({acces[sorted_id]}) to {model_names[seq_id]}...')
                shutil.copyfile(generate_model_filename(opt, model_names[sorted_id], False),
                                generate_model_filename(opt, model_names[seq_id], True))
        else:
            opt.logger.info('Sort all models based on test loss')
            sorted_ids = sorted(range(opt.model_nums), key=lambda k: losses[k])  # test_loss lower, its order ahead
            for seq_id, sorted_id in enumerate(sorted_ids):
                opt.logger.info(f'Copying {model_names[sorted_id]}({losses[sorted_id]}) to {model_names[seq_id]}...')
                shutil.copyfile(generate_model_filename(opt, model_names[sorted_id], False),
                                generate_model_filename(opt, model_names[seq_id], True))

    end_time = time.time()
    elapse_time = end_time - start_time
    opt.logger.info(f'All end  in {elapse_time // 60:.0f}m {elapse_time % 60:.0f}s.')
