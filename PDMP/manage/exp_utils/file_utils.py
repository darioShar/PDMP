from pathlib import Path
import os
import hashlib
import re
import torch

''''''''''' FILE MANIPULATION '''''''''''

# we only want to hash for model parameters and data type.
# so this is a training only hash
def hash_parameters(p):
    # save only dataset (with channels and image_size)
    # diffusion, model, optim, training
    to_hash = {'data': {k:v for k, v in p['data'].items() if k in ['dataset', 'channels', 'image_size']},
               'pdmp': {k:v for k, v in p['pdmp'].items() if k in \
                             ['time_horizon', 
                              'sampler_name']},
               'model': p['model'],
               #'optim': p['optim'],
               #'training': p['training']
               }
    res = hashlib.sha256(str(to_hash).encode('utf-8')).hexdigest()
    res = str(res)[:16]
    #res = str(hex(abs(hash(tuple(p)))))[2:]
    return res

# this is an evaluation only hash
def hash_parameters_eval(p):
    to_hash = {'eval': {k:v for k, v in p['eval'].items() if k in ['reduce_timesteps', 'clip_denoised']}}
    res = hashlib.sha256(str(to_hash).encode('utf-8')).hexdigest()
    res = str(res)[:8]
    #res = str(hex(abs(hash(tuple(p)))))[2:]
    return res

# returns new save folder and parameters hash
def get_hash_path_from_param(p, 
                             folder_path, 
                             make_new_dir = False):
    
    h = hash_parameters(p)
    save_folder_path = os.path.join(folder_path, p['data']['dataset'])
    if make_new_dir:
        Path(save_folder_path).mkdir(parents=True, exist_ok=True)
    return save_folder_path, h

# returns eval folder given save folder, and eval hash
def get_hash_path_eval_from_param(p, 
                             save_folder_path, 
                             make_new_dir = False):
    h = hash_parameters(p)
    h_eval = hash_parameters_eval(p)
    eval_folder_path = os.path.join(save_folder_path, '_'.join(('new_eval', h, h_eval)))
    if make_new_dir:
        Path(eval_folder_path).mkdir(parents=True, exist_ok=True)
    return eval_folder_path, h, h_eval

# returns paths for model and param
# from a base folder. base/data_distribution/
def get_paths_from_param(p, 
                         folder_path, 
                         make_new_dir = False, 
                         curr_epoch = None, 
                         new_eval_subdir=False): # saves eval and param in a new subfolder
    save_folder_path, h = get_hash_path_from_param(p, folder_path, make_new_dir)
    if new_eval_subdir:
        eval_folder_path, h, h_eval = get_hash_path_eval_from_param(p, save_folder_path, make_new_dir)

    names = ['model', 'parameters', 'eval']
    # create path for each name
    # in any case, model get saved in save_folder_path
    if curr_epoch is not None:
        L = {'model': '_'.join([os.path.join(save_folder_path, 'model'), h, str(curr_epoch)])}
    else:
        L = {'model': '_'.join([os.path.join(save_folder_path, 'model'), h])}
    # then depending on save_new_eval, save either in save_folder or eval_folder
    if new_eval_subdir:
        if curr_epoch is not None:
            L.update({name: '_'.join([os.path.join(eval_folder_path, name), h, h_eval, str(curr_epoch)]) for name in names[1:]})
        else:
            L.update({name: '_'.join([os.path.join(eval_folder_path, name), h, h_eval]) for name in names[1:]})
    else:
        # we consider the evaluation to be made all along the epochs, in order to get a list of evaluations.s
        # so we do not append curr_epoch here. 
        L.update({name: '_'.join([os.path.join(save_folder_path, name), h]) for name in names[1:]})
    
    return tuple(L[name] +'.pt' for name in L.keys()) # model, param, eval

