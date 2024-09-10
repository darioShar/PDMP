import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.optim as optim
from transformers import get_scheduler
import torchvision.utils as tvu

from pathlib import Path
import yaml
import os
import hashlib
import zuko

from PDMP.datasets import get_dataset, is_image_dataset
import PDMP.compute.Diffusion as Diffusion
import PDMP.models.Model as Model
from PDMP.manage.Training import Manager
import PDMP.compute.pdmp as PDMP
import PDMP.pdmp_utils.Data as Data
import PDMP.models.unet as unet
import PDMP.evaluate.Eval as Eval
import PDMP.manage.Generate as Gen
from PDMP.datasets import inverse_affine_transform
import PDMP.models.NormalizingFlow as NormalizingFLow
import PDMP.models.VAE as VAE
import PDMP.compute.NF as NF

''''''''''' FILE MANIPULATION '''''''''''

class FileUtils:
    '''
    p: parameters dictionnary
    '''
    def __init__(self, p):
        self.p = p


    # we only want to hash for model parameters and data type.
    # so this is a training only hash
    def hash_parameters(p):
        # save only dataset (with channels and image_size)
        # diffusion, model, optim, training
        # hash depends on wether we are using pdmp or diffusion.
        # check that different samplers give different hashes!!!!
        # wtf
        model_param = model_param_to_use(p)
        # print('attention: retro-compatibility with normalizing flow in hash parameter: not discrimnating model_type and model_vae_type')
        
        #retro_compatibility = ['x_emb_type', 'x_emb_size']
        retro_compatibility = ['model_type', 
                            'model_vae_type', 
                            'model_vae_t_hidden_width',
                            'model_vae_t_emb_size',
                            'model_vae_x_emb_size'
                            ]
        to_hash = {'data': {k:v for k, v in p['data'].items() if k in ['dataset', 'channels', 'image_size']},
                p['noising_process']: {k:v for k, v in p[p['noising_process']].items()},
                'model':  {k:v for k, v in model_param.items() if not k in retro_compatibility}, # here retro-compatibility
                #'optim': p['optim'],
                #'training': p['training']
                }
        res = hashlib.sha256(str(to_hash).encode('utf-8')).hexdigest()
        res = str(res)[:16]
        #res = str(hex(abs(hash(tuple(p)))))[2:]
        return res

# this is an evaluation only hash
def hash_parameters_eval(p):
    to_hash = {'eval': p['eval'][p['noising_process']]}
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
                         new_eval_subdir=False,
                         do_not_load_model=False): # saves eval and param in a new subfolder
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
        if not do_not_load_model:
            # checks if model is there. otherwise, loads latest model. also checks equality of no_iteration model and latest iteration one
            # list all model iterations
            model_paths = list(Path(save_folder_path).glob('_'.join(('model', h)) + '*'))
            assert len(model_paths) > 0, 'no models to load in {}, with hash {}'.format(save_folder_path, h)
            max_model_iteration = 0
            max_model_iteration_path = None
            for i, x in enumerate(model_paths):
                if str(x)[:-3].split('_')[-1].isdigit() and (len(str(x)[:-3].split('_')[-1]) < 8): # if it is digit, and not hash
                    model_iter = int(str(x)[:-3].split('_')[-1])
                    if max_model_iteration< model_iter:
                        max_model_iteration = model_iter
                        max_model_iteration_path = str(x)
            if max_model_iteration_path is not None:
                if Path(L['model'] + '.pt').exists():
                    print('Found another save with no specified iteration alonside others with specified iterations. Will not load it')
                print('Loading trained model at iteration {}'.format(max_model_iteration))
                L = {'model': '_'.join([os.path.join(save_folder_path, 'model'), h, str(max_model_iteration)])}
            elif Path(L['model']+ '.pt').exists():
                print('Found model with no specified iteration. Loading it')
                # L already holds the right name
                #L = {'model': '_'.join([os.path.join(save_folder_path, 'model'), h])}
            else:
                raise Exception('Did not find a model to load at location {} with hash {}'.format(save_folder_path, h))
            
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


    #save_folder_path, h = get_hash_path_from_param(p, folder_path, make_new_dir)
    #if new_eval_subdir:
    #    eval_folder_path, h, h_eval = get_hash_path_eval_from_param(p, save_folder_path, make_new_dir)
#
    #names = ['model', 'parameters', 'eval']
    ## create path for each name
    ## in any case, model get saved in save_folder_path
    #if curr_epoch is not None:
    #    L = {'model': '_'.join([os.path.join(save_folder_path, 'model'), h, str(curr_epoch)])}
    #else:
    #    L = {'model': '_'.join([os.path.join(save_folder_path, 'model'), h])}
    ## then depending on save_new_eval, save either in save_folder or eval_folder
    #if new_eval_subdir:
    #    if curr_epoch is not None:
    #        L.update({name: '_'.join([os.path.join(eval_folder_path, name), h, h_eval, str(curr_epoch)]) for name in names[1:]})
    #    else:
    #        L.update({name: '_'.join([os.path.join(eval_folder_path, name), h, h_eval]) for name in names[1:]})
    #else:
    #    # we consider the evaluation to be made all along the epochs, in order to get a list of evaluations.s
    #    # so we do not append curr_epoch here. 
    #    L.update({name: '_'.join([os.path.join(save_folder_path, name), h]) for name in names[1:]})
    #
    #return tuple(L[name] +'.pt' for name in L.keys()) # model, param, eval


def prepare_data_directories(dataset_name, dataset_files, remove_existing_eval_files, num_real_data, hash_params):

    if dataset_files is None:
        # do nothing, assume no data will be generated
        print('(prepare data directories) assuming no data will be generated.')
        return None, None

    # create directory for saving images
    folder_path = os.path.join('eval_files', dataset_name)
    generated_data_path = os.path.join(folder_path, 'generated_data', hash_params)
    if not is_image_dataset(dataset_name):
        # then we have various versions of the same dataset
        real_data_path = os.path.join(folder_path, 'original_data', hash_params)
    else:
        real_data_path = os.path.join(folder_path, 'original_data')
    
    #Path(generated_data_path).mkdir(parents=True, exist_ok=True)
    #Path(real_data_path).mkdir(parents=True, exist_ok=True)

    def remove_file_from_directory(dir):
        # remove the directory
        if not dir.is_dir():
            raise ValueError(f'{dir} is not a directory')
        # print('removing files in directory', dir)
        for file in dir.iterdir():
            file.unlink()

    def save_images(path):
            print('storing dataset in', path)
            # now saving the original data
            assert dataset_name.lower() in ['mnist', 'cifar10', 'celeba'], 'only mnist, cifar10, celeba datasets are supported for the moment. \
                For the moment we are loading {} data points. We may need more for the other datasets, \
                    and anyway we should implement somehting more systematic'.format(num_real_data)
            #data = gen_model.load_original_data(evaluation_files) # load all the data. Number of datapoints specific to mnist and cifar10
            data_to_store = num_real_data
            print('saving {} original images from pool of {} datapoints'.format(data_to_store, len(dataset_files)))
            for i in range(data_to_store):
                if (i%500) == 0:
                    print(i, end=' ')
                tvu.save_image(inverse_affine_transform(dataset_files[i][0]), os.path.join(path, f"{i}.png"))
    
    path = Path(generated_data_path)
    if path.exists():
        if remove_existing_eval_files:
            remove_file_from_directory(path)
    else:
        path.mkdir(parents=True, exist_ok=True)

    path = Path(real_data_path)
    if is_image_dataset(dataset_name):
        if path.exists():
            print('found', path)
            assert path.is_dir(), (f'{path} is not a directory')
            # check that there are the right number of image files, else remove and regenerate
            if len(list(path.iterdir())) != num_real_data:
                remove_file_from_directory(path)
                save_images(path)
        else:
            path.mkdir(parents=True, exist_ok=True)
            save_images(path)
    else:
        if path.exists():
            remove_file_from_directory(path)
        else:
            path.mkdir(parents=True, exist_ok=True)

    return generated_data_path, real_data_path




''''''''''' PREPARE FROM PARAMETER DICT '''''''''''

# for the moment, only unconditional models
def _unet_model(p, p_model_unet, bin_input_zigzag=False):
    assert bin_input_zigzag == False, 'bin_input_zigzag nyi for unet'
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

    channels = p['data']['channels']
    if p['pdmp']['sampler'] == 'ZigZag':
        out_channels = 2*channels #(channels if (not learn_gamma) or (not (p['pdmp']['sampler'] == 'ZigZag')) else 2*channels)
    else:
        out_channels = channels
    
    model = unet.UNetModel(
            in_channels=channels,
            model_channels=p_model_unet['model_channels'],
            out_channels= out_channels,
            num_res_blocks=p_model_unet['num_res_blocks'],
            attention_resolutions=p_model_unet['attn_resolutions'],# tuple([2, 4]), # adds attention at image_size / 2 and /4
            dropout= p_model_unet['dropout'],
            channel_mult= p_model_unet['channel_mult'], # divides image_size by two at each new item, except first one. [i] * model_channels
            dims = 2, # for images
            num_classes= None,#(NUM_CLASSES if class_cond else None),
            use_checkpoint=False,
            num_heads=p_model_unet['num_heads'],
            num_heads_upsample=-1, # same as num_heads
            use_scale_shift_norm=True,
            beta = p_model_unet['beta'] if p['pdmp']['sampler'] == 'ZigZag' else None,
            threshold = p_model_unet['threshold'] if p['pdmp']['sampler'] == 'ZigZag' else None,
            denoiser=p['pdmp']['denoiser'],
        )
    return model

def model_param_to_use(p):
    if p['noising_process'] == 'diffusion':
        if is_image_dataset(p['data']['dataset']):
            return p['model']['unet']
        else:
            return p['model']['mlp']
    elif p['noising_process'] == 'nf':
        return p['model']['nf']
    elif p['pdmp']['sampler'] == 'ZigZag':
        if is_image_dataset(p['data']['dataset']):
            return p['model']['unet']
        else:
            return p['model']['mlp']
    else:
        return p['model']['normalizing_flow']

def init_model_by_parameter(p):
    # model
    model_param = model_param_to_use(p)
    method = p['noising_process'] if p['noising_process'] in ['diffusion', 'nf'] else p['pdmp']['sampler']
    if not is_image_dataset(p['data']['dataset']):
        # model
        if method in ['diffusion', 'ZigZag']:
            model = Model.MLPModel(nfeatures = p['data']['dim'],
                                    device=p['device'], 
                                    p_model_mlp=model_param,
                                    noising_process=method,
                                    bin_input_zigzag=p['additional']['bin_input_zigzag'])
        elif method == 'nf':
            type = model_param['model_type']
            nfeatures = p['data']['dim']
            if type == 'NSF':
                model = zuko.flows.NSF(nfeatures,
                               0,
                               transforms=model_param['transforms'], #3
                                hidden_features= [model_param['hidden_width']] * model_param['hidden_depth'] ) #[128] * 3)
            elif type == 'MAF':
                model = zuko.flows.MAF(nfeatures,
                               0,
                               transforms=model_param['transforms'], #3
                                hidden_features= [model_param['hidden_width']] * model_param['hidden_depth'] ) #[128] * 3)
            else:
                raise Exception('NF type {} not yet implement'.format(type))
        else:
            # Neural spline flow (NSF) with dim sample features (V_t) and dim + 1 context features (X_t, t)
            #print('retro_compatibility: default values for 2d data when loading model')
            #p['model']['normalizing_flow']['x_emb_type'] = 'concatenate'
            #p['model']['normalizing_flow']['x_emb_size'] = 2
            if p['pdmp']['learn_jump_time']:
                model = NormalizingFLow.NormalizingFlowModelJumpTime(nfeatures=p['data']['dim'], 
                                                            device=p['device'], 
                                                            p_model_normalizing_flow=p['model']['normalizing_flow'])
            else:
                model = NormalizingFLow.NormalizingFlowModel(nfeatures=p['data']['dim'], 
                                                            device=p['device'], 
                                                            p_model_normalizing_flow=p['model']['normalizing_flow'])
    else:
        if method in ['diffusion', 'ZigZag']:
            model = _unet_model(p, p_model_unet = model_param, bin_input_zigzag=p['additional']['bin_input_zigzag'])
        else:
            # Neural spline flow (NSF) with dim sample features (V_t) and dim + 1 context features (X_t, t)
            data_dim = p['data']['image_size']**2 * p['data']['channels']
            if p['pdmp']['learn_jump_time']:
                model_vae_type = p['model']['normalizing_flow']['model_vae_type']
                if model_vae_type == 'VAE_1':
                    model = VAE.VAEJumpTime(nfeatures=data_dim, p_model_nf=p['model']['normalizing_flow'])
                elif model_vae_type == 'VAE_16': 
                    model = VAE.MultiVAEJumpTime(nfeatures=data_dim, n_vae=16, time_horizon=p['pdmp']['time_horizon'], p_model_nf=p['model']['normalizing_flow'])
                    
                #model = NormalizingFLow.NormalizingFlowModelJumpTime(nfeatures=p['data']['dim'], 
                #                                            device=p['device'], 
                #                                            p_model_normalizing_flow=p['model']['normalizing_flow'],
                #                                            unet=_unet_model(p, p_model_unet=p['model']['unet']))
            else:
                model = NormalizingFLow.NormalizingFlowModel(nfeatures=data_dim, 
                                                            device=p['device'], 
                                                            p_model_normalizing_flow=p['model']['normalizing_flow'],
                                                            unet=_unet_model(p, p_model_unet=p['model']['unet']))

    return model.to(p['device'])

def init_model_vae_by_parameter(p):
    # model
    if not p['model']['vae']:
        return None
    method = p['noising_process'] if p['noising_process'] in ['diffusion', 'nf'] else p['pdmp']['sampler']
    if not is_image_dataset(p['data']['dataset']):
        if method == 'diffusion':
            model = NormalizingFLow.NormalizingFlowModel(nfeatures=p['data']['dim'], 
                                                        device=p['device'], 
                                                        p_model_normalizing_flow=p['model']['normalizing_flow'])
        else:
            model = VAE.VAESimpleND(nfeatures=p['data']['dim'], device=p['device'])
    else:
        data_dim = p['data']['image_size']**2 * p['data']['channels']
        model_vae_type = p['model']['normalizing_flow']['model_vae_type']
        if method == 'nf':
            model = VAE.VAESimple(nfeatures=data_dim, p_model_nf=p['model']['normalizing_flow'])
        else:
            if model_vae_type == 'VAE_1':
                model = VAE.VAE(nfeatures=data_dim, p_model_nf=p['model']['normalizing_flow'])
            elif model_vae_type == 'VAE_16': 
                model = VAE.MultiVAE(nfeatures=data_dim, n_vae=16, time_horizon=p['pdmp']['time_horizon'], p_model_nf=p['model']['normalizing_flow'])
    return model.to(p['device'])

def init_data_by_parameter(p):
    # get the dataset
    dataset_files, test_dataset_files = get_dataset(p)
    
    # implement DDP later on
    data = DataLoader(dataset_files, 
                      batch_size=p['data']['bs'], 
                      shuffle=True, 
                      num_workers=p['data']['num_workers'])
    test_data = DataLoader(test_dataset_files,
                            batch_size=p['data']['bs'],
                            shuffle=True,
                            num_workers=p['data']['num_workers'])
    return data, test_data, dataset_files, test_dataset_files

def init_noising_process_by_parameter(p):
    #gammas = Diffusion.LevyDiffusion.gen_noise_schedule(p['diffusion']['diffusion_steps']).to(p['device'])
    if p['noising_process'] == 'pdmp':
        noising_process = PDMP.PDMP(
                        device = p['device'],
                        time_horizon = p['pdmp']['time_horizon'],
                        reverse_steps = p['eval']['pdmp']['reverse_steps'],
                        sampler = p['pdmp']['sampler'],
                        refresh_rate = p['pdmp']['refresh_rate'],
                        add_losses= p['pdmp']['add_losses'] if p['pdmp']['add_losses'] is not None else [],
                        use_softmax= p['additional']['use_softmax'],
                        learn_jump_time=p['pdmp']['learn_jump_time'],
                        bin_input_zigzag = p['additional']['bin_input_zigzag'],
                        denoiser = p['pdmp']['denoiser']
                        )
    elif p['noising_process'] == 'diffusion':
        noising_process = Diffusion.LevyDiffusion(alpha = p['diffusion']['alpha'],
                                   device = p['device'],
                                   diffusion_steps = p['diffusion']['reverse_steps'],
                                   model_mean_type = p['diffusion']['mean_predict'],
                                   model_var_type = p['diffusion']['var_predict'],
                                   loss_type = p['diffusion']['loss_type'],
                                   rescale_timesteps = p['diffusion']['rescale_timesteps'],
                                   isotropic = p['diffusion']['isotropic'],
                                   clamp_a=p['diffusion']['clamp_a'],
                                   clamp_eps=p['diffusion']['clamp_eps'],
                                   LIM = p['diffusion']['LIM'],
                                   diffusion_settings=p['diffusion'],
                                   #config = p['LIM_config'] if p['LIM'] else None
        )
    elif p['noising_process'] == 'nf':
        noising_process = NF.NF(reverse_steps = 1,
                                device = p['device'])
    
    return noising_process


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


def init_generation_manager_by_parameter(noising_process, dataloader, p):
    # here kwargs is passed to the underlying Generation Manager.
    kwargs = p['eval'][p['noising_process']]

    return Gen.GenerationManager(noising_process, 
                                 #reverse_steps=p['eval'][p['noising_process']]['reverse_steps'], 
                                 dataloader=dataloader, 
                                 is_image = is_image_dataset(p['data']['dataset']),
                                 **kwargs)


def init_eval_by_parameter(noising_process, gen_manager, data, logger, gen_data_path, real_data_path, p):

    eval = Eval.Eval( 
            noising_process=noising_process,
            gen_manager=gen_manager,
            dataloader=data,
            verbose=True, 
            logger = logger,
            data_to_generate = p['eval']['data_to_generate'],
            batch_size = p['eval']['batch_size'],
            is_image = is_image_dataset(p['data']['dataset']),
            gen_data_path=gen_data_path,
            real_data_path=real_data_path
    )
    return eval

def reset_model(p):
    model = init_model_by_parameter(p)
    optim = init_optimizer_by_parameter(model, p)
    learning_schedule = init_ls_by_parameter(model, p)
    return model, optim, learning_schedule

def reset_vae(p):
    model_vae = init_model_vae_by_parameter(p)
    optim_vae = init_optimizer_by_parameter(model_vae, p) if model_vae is not None else None
    learning_schedule_vae = init_ls_by_parameter(optim_vae, p) if model_vae is not None else None
    return model_vae, optim_vae, learning_schedule_vae

def init_manager_by_parameter(model,
                              model_vae,
                              data,
                              noising_process, 
                              optimizer,
                              optimizer_vae,
                              learning_schedule,
                              learning_schedule_vae,
                              eval, 
                              logger,
                              p):
    
    # here kwargs goes to manager (ema_rates), train_loop (grad_clip), and eventually to training_losses (monte_carlo...)
    kwargs = p['training'][p['noising_process']]
    manager = Manager(model,
                model_vae,
                data,
                noising_process,
                optimizer,
                optimizer_vae,
                learning_schedule,
                learning_schedule_vae,
                eval,
                logger,
                reset_vae=reset_vae,
                p = p,
                eval_freq = p['run']['eval_freq'],
                checkpoint_freq = p['run']['checkpoint_freq'],
                # ema_rate, grad_clip
                **kwargs
                )
    return manager

def prepare_data_directories_from_param(dataset_files, p):
    # prepare the evaluation directories
    return prepare_data_directories(dataset_name=p['data']['dataset'],
                             dataset_files = dataset_files, 
                             remove_existing_eval_files = False if p['eval']['data_to_generate'] == 0 else True,
                             num_real_data = p['eval']['real_data'],
                             hash_params = '_'.join([hash_parameters(p), hash_parameters_eval(p)]), # for saving images. We want a hash specific to the training, and to the sampling
                             )

def prepare_experiment(p, logger = None, do_not_load_data=False):

    # intialize logger
    if logger is not None:
        logger.initialize(p)

    model = init_model_by_parameter(p)
    model_vae = init_model_vae_by_parameter(p)

    if do_not_load_data:
        data, test_data, dataset_files, test_dataset_files = None, None, None, None
    else:
        data, test_data, dataset_files, test_dataset_files = init_data_by_parameter(p)
    
    # prepare the evaluation directories
    gen_data_path, real_data_path = prepare_data_directories_from_param(dataset_files, p)

    noising_process = init_noising_process_by_parameter(p)
    optim = init_optimizer_by_parameter(model, p)
    learning_schedule = init_ls_by_parameter(optim, p)
    optim_vae = init_optimizer_by_parameter(model_vae, p) if model_vae is not None else None
    learning_schedule_vae = init_ls_by_parameter(optim_vae, p) if model_vae is not None else None

    # get generation manager
    gen_manager = init_generation_manager_by_parameter(noising_process, data, p)

    # run evaluation on train or test data
    eval = init_eval_by_parameter(noising_process, gen_manager, data, logger, gen_data_path, real_data_path, p)
    
    # run training
    manager = init_manager_by_parameter(model,
                                        model_vae,
                                        data, 
                                        noising_process, 
                                        optim,
                                        optim_vae,
                                        learning_schedule,
                                        learning_schedule_vae,
                                        eval,
                                        logger,
                                        p)
    return model, data, test_data, manager



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
    model, data, test_data, manager = prepare_experiment(p, logger, do_not_load_data)
    if not do_not_load_model:
        print('loading from model file {}'.format(model_path))
        manager.load(model_path)
    print('loading from eval file {}'.format(eval_path))
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
    model_path, _, eval_path = get_paths_from_param(p, 
                                                   folder_path, 
                                                   curr_epoch=curr_epoch,
                                                   new_eval_subdir = load_eval_subdir,
                                                   do_not_load_model=do_not_load_model)
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
    model_path, param_path, eval_path = get_paths_from_param(p, 
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



