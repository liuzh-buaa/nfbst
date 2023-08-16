import numpy as np
import torch
from captum.attr import Saliency

from datasets.regdata import build_reg_dataset
from models.model_utils import load_model, init_model_config
from utils.utils_file import generate_bayes_factors_filename
from utils.utils_parser import DefaultArgumentParser, init_config

if __name__ == '__main__':
    parser = DefaultArgumentParser().get_parser()

    opt = parser.parse_args()
    opt.model_type = 'nn'
    opt.model_name = 'nn_1'
    opt.interpret_method = 'gradient'
    opt.data = 'mnist'
    # opt.data = 'simulation_v4'
    opt.activation = 'relu'
    init_config(opt, model_config=True)

    dataset = build_reg_dataset(opt, train=False)

    model = load_model(opt.model_name, opt, resume=True, last=True)

    # check the approach of calculating gradient
    for j in range(10):
        opt.y_index = j
        for i in range(10):
            data = torch.unsqueeze(dataset[i][0].to(opt.device), dim=0)
            attributions1 = model.get_interpret(data, opt)
            model.convert_to_standard_model()
            saliency = Saliency(model)
            attributions2 = saliency.attribute(data, target=j)
            model.recover_from_standard_model()
            print('ok')

    # check the approach of calculating bayes factors
    for j in range(10):
        opt.y_index = j
        bayes_factors = np.load(generate_bayes_factors_filename(opt, last=True))
        for i in range(10):
            data = torch.unsqueeze(dataset[i][0].to(opt.device), dim=0)
            attributions1 = bayes_factors[i]
            model.convert_to_standard_model()
            saliency = Saliency(model)
            attributions2 = saliency.attribute(data, target=j)
            model.recover_from_standard_model()
            print('ok')
