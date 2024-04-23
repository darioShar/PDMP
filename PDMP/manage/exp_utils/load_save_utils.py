import torch
import PDMP.manage.exp_utils.prepare_utils as prepare_utils
import PDMP.manage.exp_utils.file_utils as file_utils
from pathlib import Path
import yaml
import os

''''''''''' LOADING/SAVING '''''''''''
def load_param_from_config(config_path, config_file):
    with open(os.path.join(config_path, config_file), "r") as f:
        config = yaml.safe_load(f)
    return config

# loads all params from a specific folder
def load_params_from_folder(folder_path):
    return [torch.load(path) for path in Path(folder_path).glob("parameters*")]


def _load_experiment(p, 
                     model_path, 
                     eval_path, 
                     logger,
                     do_not_load_model = False,
                     do_not_load_data = False):
    model, data, test_data, manager = prepare_utils.prepare_experiment(p, logger, do_not_load_data)
    if not do_not_load_model:
        manager.load(model_path)
    manager.load_eval_metrics(eval_path)
    #manager.losses = torch.load(eval_path)
    return model, data, test_data, manager

# loads a model from some param as should be contained in folder_path.
# Specify the training epoch at which to load; defaults to latest
def load_experiment_from_param(p, 
                               folder_path, 
                               logger=None,
                               curr_epoch = None,
                               do_not_load_model = False,
                               do_not_load_data=False,
                               load_eval_subdir=False):
    model_path, _, eval_path = file_utils.get_paths_from_param(p, 
                                                   folder_path, 
                                                   curr_epoch=curr_epoch,
                                                   new_eval_subdir = load_eval_subdir)
    model, data, test_data, manager = _load_experiment(p, 
                                            model_path, 
                                            eval_path, 
                                            logger,
                                            do_not_load_model=do_not_load_model,
                                            do_not_load_data=do_not_load_data)
    return model, data, test_data, manager


# unique hash of parameters, append training epochs
# simply separate folder by data distribution and alpha value
def save_experiment(p, 
                    base_path, 
                    manager, 
                    curr_epoch = None,
                    files = 'all',
                    save_new_eval=False): # will save eval and param in a subfolder.
    if isinstance(files, str):
        files = [files]
    for f in files:
        assert f in ['all', 'model', 'eval', 'param'], 'files must be one of all, model, eval, param'
    model_path, param_path, eval_path = file_utils.get_paths_from_param(p, 
                                                             base_path, 
                                                             make_new_dir = True, 
                                                             curr_epoch=curr_epoch,
                                                             new_eval_subdir=save_new_eval)
    #model_path = '_'.join([model_path, str(manager.training_epochs())]) 
    #losses_path = '_'.join([model_path, 'losses']) + '.pt'
    if 'all' in files:
        manager.save(model_path)
        manager.save_eval_metrics(eval_path)
        torch.save(p, param_path)
        return model_path, param_path, eval_path
    
    # else, slightly more complicated logic
    objects_to_save = {name: {'path': p, 'saved':False} for name, p in zip(['model', 'eval', 'param'],
                                                                     [model_path, eval_path, param_path])}
    for name, obj in objects_to_save.items():
        if name in files:
            obj['saved'] = True
            if name == 'model':
                manager.save(obj['path'])
            elif name == 'eval':
                manager.save_eval_metrics(obj['path'])
            elif name == 'param':
                torch.save(p, obj['path'])
    
    # return values in the right order
    return tuple(objects_to_save[name]['path'] if objects_to_save[name]['saved'] else None for name in ['model', 'eval', 'param'])

    if 'model' in files:
        manager.save(model_path)
        eval_path = None
        param_path = None
    if 'eval' in files:
        manager.save_eval_metrics(eval_path)
        model_path = None
        param_path = None
    if 'param' in files:
        torch.save(p, param_path)
        model_path = None
        eval_path = None
    return model_path, param_path, eval_path
