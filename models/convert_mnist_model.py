from collections import OrderedDict

import torch

if __name__ == '__main__':
    for i in range(1, 11):
        params = torch.load(f'../data/mnist/results/models/nn_{i}')
        new_params = OrderedDict()
        new_params['conv.0.weight'] = params['model.0.weight']
        new_params['conv.0.bias'] = params['model.0.bias']
        new_params['conv.3.weight'] = params['model.3.weight']
        new_params['conv.3.bias'] = params['model.3.bias']
        new_params['fc.0.weight'] = params['model.7.weight']
        new_params['fc.0.bias'] = params['model.7.bias']
        new_params['fc.2.weight'] = params['model.9.weight']
        new_params['fc.2.bias'] = params['model.9.bias']
        torch.save(new_params, f'../data/mnist/results/models/nn_{i}')

    for i in range(1, 11):
        params = torch.load(f'../data/mnist/results/models/gaussian_{i}')
        new_params = OrderedDict()
        new_params['conv.0.mu_weight'] = params['model.0.mu_weight']
        new_params['conv.0.log_sigma_weight'] = params['model.0.log_sigma_weight']
        new_params['conv.0.mu_bias'] = params['model.0.mu_bias']
        new_params['conv.0.log_sigma_bias'] = params['model.0.log_sigma_bias']
        new_params['conv.0.eps_weight'] = params['model.0.eps_weight']
        new_params['conv.0.eps_bias'] = params['model.0.eps_bias']
        new_params['conv.3.mu_weight'] = params['model.3.mu_weight']
        new_params['conv.3.log_sigma_weight'] = params['model.3.log_sigma_weight']
        new_params['conv.3.mu_bias'] = params['model.3.mu_bias']
        new_params['conv.3.log_sigma_bias'] = params['model.3.log_sigma_bias']
        new_params['conv.3.eps_weight'] = params['model.3.eps_weight']
        new_params['conv.3.eps_bias'] = params['model.3.eps_bias']
        new_params['fc.0.mu_weight'] = params['model.7.mu_weight']
        new_params['fc.0.log_sigma_weight'] = params['model.7.log_sigma_weight']
        new_params['fc.0.mu_bias'] = params['model.7.mu_bias']
        new_params['fc.0.log_sigma_bias'] = params['model.7.log_sigma_bias']
        new_params['fc.0.eps_weight'] = params['model.7.eps_weight']
        new_params['fc.0.eps_bias'] = params['model.7.eps_bias']
        new_params['fc.2.mu_weight'] = params['model.9.mu_weight']
        new_params['fc.2.log_sigma_weight'] = params['model.9.log_sigma_weight']
        new_params['fc.2.mu_bias'] = params['model.9.mu_bias']
        new_params['fc.2.log_sigma_bias'] = params['model.9.log_sigma_bias']
        new_params['fc.2.eps_weight'] = params['model.9.eps_weight']
        new_params['fc.2.eps_bias'] = params['model.9.eps_bias']
        torch.save(new_params, f'../data/mnist/results/models/gaussian_{i}')

    print('ok ')
