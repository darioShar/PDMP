import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.optim as optim
import torch.nn as nn

import matplotlib.pyplot as plt

import DDLM.models.Model as Model
from DDLM.manage.Manager import Manager
import DDLM.compute.Diffusion as Diffusion
import DDLM.levy_utils.Data as Data

import DDLM.models.unet as unet
import DDLM.evaluate.Eval as Eval

import torchvision
from transformers import get_scheduler

from DDLM.datasets import get_dataset, is_image_dataset

''''''''''' PREPARE FROM PARAMETER DICT '''''''''''

# for the moment, only unconditional models
def _unet_model(p):
    image_size = p['data']['image_size']
    # the usual channel multiplier. Can choose otherwise in config files.
    '''if image_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif image_size == 64:
        channel_mult = (1, 2, 3, 4)
    elif image_size == 32:
        channel_mult = (1, 2, 2, 2)
    else:
        raise ValueError(f"unsupported image size: {image_size}")
    '''

    learn_gamma = p['model']['compute_gamma']
    channels = p['data']['channels']
    model = unet.UNetModel(
            in_channels=channels,
            model_channels=p['model']['model_channels'],
            out_channels= (channels if not learn_gamma else 2*channels),
            num_res_blocks=p['model']['num_res_blocks'],
            attention_resolutions=p['model']['attn_resolutions'],# tuple([2, 4]), # adds attention at image_size / 2 and /4
            dropout= p['model']['dropout'],
            channel_mult= p['model']['channel_mult'], # divides image_size by two at each new item, except first one. [i] * model_channels
            dims = 2,
            num_classes= None,#(NUM_CLASSES if class_cond else None),
            use_checkpoint=False,
            num_heads=p['model']['num_heads'],
            num_heads_upsample=-1, # same as num_heads
            use_scale_shift_norm=True,
        )
    return model

def init_model_by_parameter(p):
    # model
    if not is_image_dataset(p['data']['dataset']):
        model = Model.LevyDiffusionModel(p)
        model = model.to(p['device'])
    else:
        model = _unet_model(p)
        model = model.to(p['device'])
    return model


def init_data_by_parameter(p):
    # get the dataset
    dataset, test_dataset = get_dataset(p)
    # implement DDP later on
    data = DataLoader(dataset, 
                      batch_size=p['training']['bs'], 
                      shuffle=True, 
                      num_workers=p['data']['num_workers'])
    test_data = DataLoader(test_dataset,
                            batch_size=p['training']['bs'],
                            shuffle=True,
                            num_workers=p['data']['num_workers'])
    return data, test_data

def init_diffusion_by_parameter(p):
    #gammas = Diffusion.LevyDiffusion.gen_noise_schedule(p['diffusion']['diffusion_steps']).to(p['device'])
    diff = Diffusion.LevyDiffusion(alpha = p['diffusion']['alpha'],
                                   device = p['device'],
                                   diffusion_steps = p['diffusion']['diffusion_steps'],
                                   model_mean_type = p['diffusion']['mean_predict'],
                                   model_var_type = p['diffusion']['var_predict'],
                                   loss_type = p['training']['loss_type'],
                                   rescale_timesteps = p['diffusion']['rescale_timesteps'],
                                   isotropic = p['diffusion']['isotropic'],
                                   clamp_a=p['diffusion']['clamp_a'],
                                   clamp_eps=p['diffusion']['clamp_eps'],
                                   #LIM = p['LIM'],
                                   #config = p['LIM_config'] if p['LIM'] else None
                                   )
    return diff


def init_optimizer_by_parameter(model, p):
    # training manager
    optimizer = optim.AdamW(model.parameters(), 
                            lr=p['optim']['lr'], 
                            betas=(0.9, 0.99)) # beta_2 0.95 instead of 0.999
    return optimizer

def init_ls_by_parameter(optim, p):
    lr_scheduler = get_scheduler(
            p['optim']['schedule'],
            # "cosine",
            # "cosine_with_restarts",
            optimizer=optim,
            num_warmup_steps=p['optim']['warmup'],
            num_training_steps=p['optim']['lr_steps'],
        )
    return lr_scheduler

def init_eval_by_parameter(model, diffusion, data, logger, p):
    eval = Eval.Eval(model, 
            diffusion, 
            data,
            verbose=True, 
            logger = logger,
            ddim = p['eval']['ddim'],
            eval_eta = p['eval']['eval_eta'], 
            reduce_timesteps = p['eval']['reduce_timesteps'],
            data_to_generate = p['eval']['data_to_generate'],
            is_image = is_image_dataset(p['data']['dataset']),
            clip_denoised = p['eval']['clip_denoised'])
    return eval

def init_manager_by_parameter(model, 
                              data,
                              diffusion, 
                              optimizer,
                              learning_schedule,
                              eval, 
                              logger,
                              p):
    # training manager
    if logger is not None:
        logger.initialize(p)
    manager = Manager(model, 
                data,
                diffusion,
                optimizer,
                learning_schedule,
                eval,
                p['training']['ema_rates'],
                logger,
                lploss = p['training']['lploss'],
                monte_carlo_steps = p['training']['monte_carlo_steps'],
                loss_monte_carlo = p['training']['loss_monte_carlo'],
                grad_clip = p['training']['grad_clip'],
                )
    return manager

def prepare_experiment(p, logger = None):
    model = init_model_by_parameter(p)
    data, test_data = init_data_by_parameter(p)
    diffusion = init_diffusion_by_parameter(p)
    optim = init_optimizer_by_parameter(model, p)
    learning_schedule = init_ls_by_parameter(optim, p)
    # run it on test data
    eval = init_eval_by_parameter(model, diffusion, test_data, logger, p)
    # run it on training data
    manager = init_manager_by_parameter(model, 
                                        data, 
                                        diffusion, 
                                        optim,
                                        learning_schedule,
                                        eval,
                                        logger, 
                                        p)
    return model, data, test_data, manager





    '''if unet:
        def tmp_func(x):
            return nn.functional.pad(x, (2, 2, 2, 2), value=x[0][0][0][0])
        data = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data', train=True, download=True,
                                 transform=torchvision.transforms.Compose([
                                   torchvision.transforms.Resize((32, 32)),
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                     (0.1307,), (0.3081,))
                                 ])),
        batch_size=p['training']['bs'], shuffle=True, num_workers=p['data']['num_workers'])
    else:
        data_dict = p['data']
        data_gen = Data.Generator(data_dict['dataset'], 
                                    n = int(np.sqrt(data_dict['n_mixture'])), 
                                    std = data_dict['std'], 
                                    normalize = data_dict['normalized'],
                                    weights = data_dict['weights'],
                                    theta = data_dict['theta'],
                                    alpha = data_dict['data_alpha'])
        data_gen.generate(n_samples = data_dict['nsamples'])
        # possibly remove a dimension if nfeatures == 1
        if p['data']['nfeatures'] == 1:
            data_gen.samples = data_gen.samples[:, 0].unsqueeze(1)
        # add channel
        data_gen.samples = data_gen.samples.unsqueeze(1) # add channel
        dataset = TensorDataset(data_gen.samples, torch.tensor([0.]).repeat(data_gen.samples.shape[0]))
        #dataset = TensorDataset(data_gen.samples)
        data = DataLoader(dataset, batch_size=p['training']['bs'], shuffle=True)'''