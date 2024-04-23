import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.optim as optim
import torch.nn as nn

import matplotlib.pyplot as plt

import PDMP.models.Model as Model
from PDMP.manage.Manager import Manager
import PDMP.compute.pdmp as PDMP
import PDMP.pdmp_utils.Data as Data

import PDMP.models.unet as unet
import PDMP.evaluate.Eval as Eval

import torchvision
from transformers import get_scheduler

from PDMP.datasets import get_dataset, is_image_dataset
from PDMP.manage.exp_utils.file_utils import hash_parameters, hash_parameters_eval

import zuko

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
        # model
        if p['pdmp']['sampler_name'] == 'ZigZag':
            model = Model.LevyDiffusionModel(p)
        else:
            # Neural spline flow (NSF) with dim sample features (V_t) and dim + 1 context features (X_t, t)
            model = zuko.flows.NSF(p['data']['dim'], 
                                   p['data']['dim'] + 1, 
                                   transforms=p['model']['transforms'], #3, 
                                   hidden_features= [p['model']['hidden_width']] * p['model']['hidden_depth'] ) #[128] * 3)
        model = model.to(p['device'])
    else:
        assert p['pdmp']['sampler_name'] == 'ZigZag', 'Normalizing flows not yet implemented for image data.'
        model = _unet_model(p)
        model = model.to(p['device'])
    return model


def init_data_by_parameter(p):
    # get the dataset
    dataset_files, test_dataset_files = get_dataset(p)
    # implement DDP later on
    data = DataLoader(dataset_files, 
                      batch_size=p['training']['bs'], 
                      shuffle=True, 
                      num_workers=p['data']['num_workers'])
    test_data = DataLoader(test_dataset_files,
                            batch_size=p['training']['bs'],
                            shuffle=True,
                            num_workers=p['data']['num_workers'])
    return data, test_data, dataset_files, test_dataset_files

def init_pdmp_by_parameter(p):
    #gammas = Diffusion.LevyDiffusion.gen_noise_schedule(p['diffusion']['diffusion_steps']).to(p['device'])
    pdmp = PDMP.PDMP(
                    device = p['device'],
                    time_horizon = p['pdmp']['time_horizon'],
                    reverse_steps = p['pdmp']['reverse_steps'],
                    sampler_name = p['pdmp']['sampler_name'],
                    refresh_rate = p['pdmp']['refresh_rate'],
                    dim = p['data']['dim'],
                    )
    return pdmp


def init_optimizer_by_parameter(model, p):
    # training manager
    optimizer = optim.AdamW(model.parameters(), 
                            lr=p['optim']['lr'], 
                            betas=(0.9, 0.99)) # beta_2 0.95 instead of 0.999
    return optimizer

def init_ls_by_parameter(optim, p):
    if p['optim']['schedule'] == None:
        return None
    
    if p['optim']['schedule'] == 'steplr':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optim,
                                                       step_size= p['optim']['lr_step_size'], 
                                                       gamma= p['optim']['lr_gamma'], 
                                                       last_epoch=-1)
    else: 
        lr_scheduler = get_scheduler(
            p['optim']['schedule'],
            # "cosine",
            # "cosine_with_restarts",
            optimizer=optim,
            num_warmup_steps=p['optim']['warmup'],
            num_training_steps=p['optim']['lr_steps'],
        )
    return lr_scheduler

def init_eval_by_parameter(model, pdmp, data, dataset_files, logger, p):
    eval = Eval.Eval(model, 
            pdmp, 
            data,
            dataset_files,
            verbose=True, 
            logger = logger,
            dataset = p['data']['dataset'], # for saving images
            hash_params = '_'.join([hash_parameters(p), hash_parameters_eval(p)]), # for saving images. We want a hash specific to the training, and to the sampling
            reduce_timesteps = p['eval']['reduce_timesteps'],
            data_to_generate = p['eval']['data_to_generate'],
            is_image = is_image_dataset(p['data']['dataset']),
            remove_existing_eval_files = False if p['eval']['data_to_generate'] == 0 else True,
            clip_denoised = p['eval']['clip_denoised'])
    return eval

def init_manager_by_parameter(model, 
                              data,
                              pdmp, 
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
                pdmp,
                optimizer,
                learning_schedule,
                eval,
                p['training']['ema_rates'],
                logger,
                grad_clip = p['training']['grad_clip'],
                )
    return manager

def prepare_experiment(p, logger = None, do_not_load_data=False):

    model = init_model_by_parameter(p)
    if do_not_load_data:        
        data, test_data, dataset_files, test_dataset_files = None, None, None, None
    else:
        data, test_data, dataset_files, test_dataset_files = init_data_by_parameter(p)
    pdmp = init_pdmp_by_parameter(p)
    optim = init_optimizer_by_parameter(model, p)
    learning_schedule = init_ls_by_parameter(optim, p)
    # run evaluation on train or test data
    eval = init_eval_by_parameter(model, pdmp, data, dataset_files, logger, p)
    # run training
    manager = init_manager_by_parameter(model, 
                                        data, 
                                        pdmp, 
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