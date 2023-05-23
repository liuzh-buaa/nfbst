import time

import torch

from models.model_config import BestConfig
from utils.utils_file import generate_model_filename


def train_model_epoch(model, dataloader, criterion, optimizer, beta, opt):
    total_loss = 0.0
    total_n = 0

    model.train()
    for inputs, targets in dataloader:
        inputs = inputs.to(opt.device)
        targets = targets.to(opt.device)

        # zero the parameter gradients
        optimizer.zero_grad()

        outputs, kl = model(inputs)
        mse = criterion(outputs, targets)
        if opt.model_type == 'gaussian':
            loss = mse + beta * kl / inputs.size(0)
        else:
            loss = mse

        loss.backward()

        total_loss += loss.item() * inputs.size(0)
        total_n += inputs.size(0)

        optimizer.step()

        if opt.model_type in ['agl_net']:
            model.proximal(beta, opt.lr)

    return total_loss / total_n


def judge_overfit(metrics, patience):
    if patience == -1 or len(metrics) <= patience:
        return False

    overfit = True
    for i in range(1, patience + 1):
        if metrics[-i] < metrics[-patience - 1]:
            overfit = False
            break
    return overfit


def train_model(opt, model, dataloader, criterion, optimizer, scheduler=None, beta=None, valloader=None,
                save_file=None, train_log=True):
    since = time.time()

    if valloader is None:
        history = {'loss': []}
    else:
        history = {'loss': [], 'val_loss': []}

    best_loss = float('inf')
    for epoch in range(1, opt.epochs + 1):
        if train_log and (epoch == 1 or epoch % 100 == 0):
            opt.logger.info(f'Epoch {epoch}/{opt.epochs}')

        train_loss = train_model_epoch(model, dataloader, criterion, optimizer, beta, opt)
        history['loss'].append(train_loss)

        if scheduler is not None:
            scheduler.step()

        if valloader is not None:
            val_loss = test_model(model, valloader, criterion, opt)
            history['val_loss'].append(val_loss)
            if train_log and (epoch == 1 or epoch % 100 == 0):
                opt.logger.info(f'Epoch {epoch}: loss - {train_loss:.4f}, val_loss - {val_loss:.4f}')
        else:
            if train_log and (epoch == 1 or epoch % 100 == 0):
                opt.logger.info(f'Epoch {epoch}: loss - {train_loss:.4f}')

        # EarlyStopping
        if save_file is not None:
            if opt.monitor == 'loss':
                loss = history['loss'][-1]
                if judge_overfit(history['loss'], opt.patience):
                    opt.logger.info(f'Epoch {epoch}: loss - {train_loss:.4f}')
                    break
                if loss < best_loss:
                    best_loss = loss
                    torch.save(model.state_dict(), save_file)
            elif opt.monitor == 'val_loss':
                val_loss = history['val_loss'][-1]
                if judge_overfit(history['val_loss'], opt.patience):
                    opt.logger.info(f'Epoch {epoch}: loss - {train_loss:.4f}, val_loss - {val_loss:.4f}')
                    break
                if val_loss < best_loss:
                    best_loss = val_loss
                    torch.save(model.state_dict(), save_file)
            else:
                raise NotImplementedError('Unknown monitor when training model.')

    end = time.time()
    elapse = end - since

    if train_log:
        opt.logger.info(f'Training complete in {elapse // 60:.0f}m {elapse % 60:.0f}s')
    return history


def test_model(model, dataloader, criterion, opt):
    total_loss = 0.0
    total_n = 0
    model.eval()
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(opt.device)
            targets = targets.to(opt.device)
            predicts = model.predict(inputs, rep=opt.rep)
            loss = criterion(predicts, targets)
            total_n += inputs.size(0)
            total_loss += loss.item() * inputs.size(0)

    return total_loss / total_n


def test_model_acc(model, dataloader, opt):
    total, correct = 0, 0
    model.eval()
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(opt.device)
            targets = targets.to(opt.device)
            predicts = model.predict(inputs, rep=opt.rep)
            _, predicts = torch.max(predicts, 1)
            total += targets.size(0)
            correct += (predicts == targets).sum()

    return correct, total, correct / total


def load_model(model_name, opt, resume=False, last=True):
    if opt.model_type == 'gaussian':
        if opt.data == 'mnist':
            from models.gaussian.mnist_bnn import MnistBNN
            model = MnistBNN(opt.sigma_pi, opt.sigma_start).to(opt.device)
        elif opt.data == 'cifar10':
            from models.gaussian.vgg_bnn import vgg16bnn, vgg16bnn_bn
            # model = vgg16bnn(opt.sigma_pi, opt.sigma_start).to(opt.device)
            model = vgg16bnn_bn(opt.sigma_pi, opt.sigma_start).to(opt.device)
        else:
            from models.gaussian.bnn import BNN
            model = BNN(opt.sigma_pi, opt.sigma_start, hidden=opt.models_struct, in_features=opt.n_features,
                        out_features=opt.n_targets, activation=opt.activation).to(opt.device)
    elif opt.model_type in ['nn', 'aglnet']:
        if opt.data == 'mnist':
            from models.nn.mnist_cnn import MnistCNN
            model = MnistCNN().to(opt.device)
        elif opt.data == 'cifar10':
            from models.nn.vgg import vgg16, vgg16_bn
            # model = vgg16().to(opt.device)
            model = vgg16_bn().to(opt.device)
        else:
            from models.nn.nn import NN
            model = NN(hidden=opt.models_struct, in_features=opt.n_features,
                       out_features=opt.n_targets, activation=opt.activation).to(opt.device)
    else:
        raise NotImplementedError(f'No such a bnn mode of {opt.model_type}.')

    if resume:
        opt.logger.info(
            f'Loading model {model_name} from {generate_model_filename(opt, model_name, last)}: {opt.models_struct}')
        model.load_state_dict(torch.load(generate_model_filename(opt, model_name, last), map_location=opt.device))
    else:
        opt.logger.info(f'Building model {model_name}: {opt.models_struct}')

    return model


def init_model_config(opt):
    config = BestConfig().get(opt.data, opt.model_type)
    if config is not None:
        for k, v in config.items():
            setattr(opt, k, v)
    else:
        raise NotImplementedError(f'No pretrained model config for {opt.model_type}.')
