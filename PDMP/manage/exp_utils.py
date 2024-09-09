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
from PDMP.manage.Manager import Manager
import PDMP.compute.pdmp as PDMP
import PDMP.pdmp_utils.Data as Data
import PDMP.models.unet as unet
import PDMP.evaluate.Eval as Eval
import PDMP.manage.Generate as Gen
from PDMP.datasets import inverse_affine_transform
import PDMP.models.NormalizingFlow as NormalizingFLow
import PDMP.models.VAE as VAE
import PDMP.compute.NF as NF
from datetime import datetime

''''''''''' FILE MANIPULATION '''''''''''

class FileHandler:
    '''
    p: parameters dictionnary
    '''
    def __init__(self, p):
        self.p = p
    
    def get_file_name(self):
        # tmp = 'exp_{}_{}_{}'.format(self.p['noising_process'], self.p['data']['dataset'], datetime.now().strftime('%d_%m_%y_%H_%M_%S'))
        # return tmp
        model_param = model_param_to_use(self.p)
        # print('attention: retro-compatibility with normalizing flow in hash parameter: not discrimnating model_type and model_vae_type')
        
        #retro_compatibility = ['x_emb_type', 'x_emb_size']
        retro_compatibility = ['model_type', 
                            'model_vae_type', 
                            'model_vae_t_hidden_width',
                            'model_vae_t_emb_size',
                            'model_vae_x_emb_size'
                            ]
        to_hash = {'data': {k:v for k, v in self.p['data'].items() if k in ['dataset', 'channels', 'image_size']},
                self.p['noising_process']: {k:v for k, v in self.p[self.p['noising_process']].items()},
                'model':  {k:v for k, v in model_param.items() if not k in retro_compatibility}, # here retro-compatibility
                #'optim': p['optim'],
                #'training': p['training']
                }
        res = hashlib.sha256(str(to_hash).encode('utf-8')).hexdigest()
        res = str(res)[:16]
        #res = str(hex(abs(hash(tuple(p)))))[2:]
        return res

    # this is an evaluation only hash
    def hash_parameters_eval(self):
        to_hash = {'eval': self.p['eval'][self.p['noising_process']]}
        res = hashlib.sha256(str(to_hash).encode('utf-8')).hexdigest()
        res = str(res)[:8]
        #res = str(hex(abs(hash(tuple(p)))))[2:]
        return res

    # returns new save folder and parameters hash
    def get_file_path_from_param(self, 
                                folder_path, 
                                make_new_dir = False):
        
        h = self.get_file_name()
        save_folder_path = os.path.join(folder_path, self.p['data']['dataset'])
        if make_new_dir:
            Path(save_folder_path).mkdir(parents=True, exist_ok=True)
        return save_folder_path, h

    # returns eval folder given save folder, and eval hash
    def get_hash_path_eval_from_param(self, 
                                save_folder_path, 
                                make_new_dir = False):
        h = self.get_file_name()
        h_eval = self.hash_parameters_eval()
        eval_folder_path = os.path.join(save_folder_path, '_'.join(('new_eval', h, h_eval)))
        if make_new_dir:
            Path(eval_folder_path).mkdir(parents=True, exist_ok=True)
        return eval_folder_path, h, h_eval

    # returns paths for model and param
    # from a base folder. base/data_distribution/
    def get_paths_from_param(self, 
                            folder_path, 
                            make_new_dir = False, 
                            curr_epoch = None, 
                            new_eval_subdir=False,
                            do_not_load_model=False): # saves eval and param in a new subfolder
        save_folder_path, h = self.get_file_path_from_param(folder_path, make_new_dir)
        if new_eval_subdir:
            eval_folder_path, h, h_eval = self.get_hash_path_eval_from_param(save_folder_path, make_new_dir)

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


    def _prepare_data_directories(self, dataset_name, dataset_files, remove_existing_eval_files, num_real_data, hash_params):

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

    def prepare_data_directories(self, dataset_files):
        # prepare the evaluation directories
        return self._prepare_data_directories(dataset_name=self.p['data']['dataset'],
                                dataset_files = dataset_files, 
                                remove_existing_eval_files = False if self.p['eval']['data_to_generate'] == 0 else True,
                                num_real_data = self.p['eval']['real_data'],
                                hash_params = '_'.join([self.get_file_name(), self.hash_parameters_eval()]), # for saving images. We want a hash specific to the training, and to the sampling
                                )
    @staticmethod
    def get_param_from_config(config_path, config_file):
        with open(os.path.join(config_path, config_file), "r") as f:
            config = yaml.safe_load(f)
        return config

    @staticmethod
    # loads all params from a specific folder
    def get_params_from_folder(folder_path):
        return [torch.load(path) for path in Path(folder_path).glob("parameters*")]


''''''''''' PREPARE FROM PARAMETER DICT '''''''''''
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
    

class InitUtils:

    def __init__(self, p):
        self.p = p 

    # for the moment, only unconditional models
    def _unet_model(self, p_model_unet, bin_input_zigzag=False):
        assert bin_input_zigzag == False, 'bin_input_zigzag nyi for unet'
        image_size = self.p['data']['image_size']
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

        channels = self.p['data']['channels']
        if self.p['pdmp']['sampler'] == 'ZigZag':
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
                beta = p_model_unet['beta'] if self.p['pdmp']['sampler'] == 'ZigZag' else None,
                threshold = p_model_unet['threshold'] if self.p['pdmp']['sampler'] == 'ZigZag' else None,
                denoiser=self.p['pdmp']['denoiser'],
            )
        return model

    def init_model_by_parameter(self):
        # model
        model_param = model_param_to_use(self.p)
        method = self.p['noising_process'] if self.p['noising_process'] in ['diffusion', 'nf'] else self.p['pdmp']['sampler']
        if not is_image_dataset(self.p['data']['dataset']):
            # model
            if method in ['diffusion', 'ZigZag']:
                model = Model.MLPModel(nfeatures = self.p['data']['dim'],
                                        device=self.p['device'], 
                                        p_model_mlp=model_param,
                                        noising_process=method,
                                        bin_input_zigzag=self.p['additional']['bin_input_zigzag'])
            elif method == 'nf':
                type = model_param['model_type']
                nfeatures = self.p['data']['dim']
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
                #self.p['model']['normalizing_flow']['x_emb_type'] = 'concatenate'
                #self.p['model']['normalizing_flow']['x_emb_size'] = 2
                if self.p['pdmp']['learn_jump_time']:
                    model = NormalizingFLow.NormalizingFlowModelJumpTime(nfeatures=self.p['data']['dim'], 
                                                                device=self.p['device'], 
                                                                p_model_normalizing_flow=self.p['model']['normalizing_flow'])
                else:
                    model = NormalizingFLow.NormalizingFlowModel(nfeatures=self.p['data']['dim'], 
                                                                device=self.p['device'], 
                                                                p_model_normalizing_flow=self.p['model']['normalizing_flow'])
        else:
            if method in ['diffusion', 'ZigZag']:
                model = self._unet_model(p_model_unet = model_param, bin_input_zigzag=self.p['additional']['bin_input_zigzag'])
            else:
                # Neural spline flow (NSF) with dim sample features (V_t) and dim + 1 context features (X_t, t)
                data_dim = self.p['data']['image_size']**2 * self.p['data']['channels']
                if self.p['pdmp']['learn_jump_time']:
                    model_vae_type = self.p['model']['normalizing_flow']['model_vae_type']
                    if model_vae_type == 'VAE_1':
                        model = VAE.VAEJumpTime(nfeatures=data_dim, p_model_nf=self.p['model']['normalizing_flow'])
                    elif model_vae_type == 'VAE_16': 
                        model = VAE.MultiVAEJumpTime(nfeatures=data_dim, n_vae=16, time_horizon=self.p['pdmp']['time_horizon'], p_model_nf=self.p['model']['normalizing_flow'])
                        
                    #model = NormalizingFLow.NormalizingFlowModelJumpTime(nfeatures=self.p['data']['dim'], 
                    #                                            device=self.p['device'], 
                    #                                            p_model_normalizing_flow=self.p['model']['normalizing_flow'],
                    #                                            unet=_unet_model(p, p_model_unet=self.p['model']['unet']))
                else:
                    model = NormalizingFLow.NormalizingFlowModel(nfeatures=data_dim, 
                                                                device=self.p['device'], 
                                                                p_model_normalizing_flow=self.p['model']['normalizing_flow'],
                                                                unet=self._unet_model(p_model_unet=self.p['model']['unet']))

        return model.to(self.p['device'])

    def init_model_vae_by_parameter(self):
        # model
        if not self.p['model']['vae']:
            return None
        method = self.p['noising_process'] if self.p['noising_process'] in ['diffusion', 'nf'] else self.p['pdmp']['sampler']
        if not is_image_dataset(self.p['data']['dataset']):
            if method == 'diffusion':
                model = NormalizingFLow.NormalizingFlowModel(nfeatures=self.p['data']['dim'], 
                                                            device=self.p['device'], 
                                                            p_model_normalizing_flow=self.p['model']['normalizing_flow'])
            else:
                model = VAE.VAESimpleND(nfeatures=self.p['data']['dim'], device=self.p['device'])
        else:
            data_dim = self.p['data']['image_size']**2 * self.p['data']['channels']
            model_vae_type = self.p['model']['normalizing_flow']['model_vae_type']
            if method == 'nf':
                model = VAE.VAESimple(nfeatures=data_dim, p_model_nf=self.p['model']['normalizing_flow'])
            else:
                if model_vae_type == 'VAE_1':
                    model = VAE.VAE(nfeatures=data_dim, p_model_nf=self.p['model']['normalizing_flow'])
                elif model_vae_type == 'VAE_16': 
                    model = VAE.MultiVAE(nfeatures=data_dim, n_vae=16, time_horizon=self.p['pdmp']['time_horizon'], p_model_nf=self.p['model']['normalizing_flow'])
        return model.to(self.p['device'])

    def init_data_by_parameter(self):
        # get the dataset
        dataset_files, test_dataset_files = get_dataset(self.p)
        
        # implement DDP later on
        data = DataLoader(dataset_files, 
                        batch_size=self.p['data']['bs'], 
                        shuffle=True, 
                        num_workers=self.p['data']['num_workers'])
        test_data = DataLoader(test_dataset_files,
                                batch_size=self.p['data']['bs'],
                                shuffle=True,
                                num_workers=self.p['data']['num_workers'])
        return data, test_data, dataset_files, test_dataset_files

    def init_noising_process_by_parameter(self):
        #gammas = Diffusion.LevyDiffusion.gen_noise_schedule(self.p['diffusion']['diffusion_steps']).to(self.p['device'])
        if self.p['noising_process'] == 'pdmp':
            noising_process = PDMP.PDMP(
                            device = self.p['device'],
                            time_horizon = self.p['pdmp']['time_horizon'],
                            reverse_steps = self.p['eval']['pdmp']['reverse_steps'],
                            sampler = self.p['pdmp']['sampler'],
                            refresh_rate = self.p['pdmp']['refresh_rate'],
                            add_losses= self.p['pdmp']['add_losses'] if self.p['pdmp']['add_losses'] is not None else [],
                            use_softmax= self.p['additional']['use_softmax'],
                            learn_jump_time=self.p['pdmp']['learn_jump_time'],
                            bin_input_zigzag = self.p['additional']['bin_input_zigzag'],
                            denoiser = self.p['pdmp']['denoiser']
                            )
        elif self.p['noising_process'] == 'diffusion':
            noising_process = Diffusion.LevyDiffusion(alpha = self.p['diffusion']['alpha'],
                                    device = self.p['device'],
                                    diffusion_steps = self.p['diffusion']['reverse_steps'],
                                    model_mean_type = self.p['diffusion']['mean_predict'],
                                    model_var_type = self.p['diffusion']['var_predict'],
                                    loss_type = self.p['diffusion']['loss_type'],
                                    rescale_timesteps = self.p['diffusion']['rescale_timesteps'],
                                    isotropic = self.p['diffusion']['isotropic'],
                                    clamp_a=self.p['diffusion']['clamp_a'],
                                    clamp_eps=self.p['diffusion']['clamp_eps'],
                                    LIM = self.p['diffusion']['LIM'],
                                    diffusion_settings=self.p['diffusion'],
                                    #config = self.p['LIM_config'] if self.p['LIM'] else None
            )
        elif self.p['noising_process'] == 'nf':
            noising_process = NF.NF(reverse_steps = 1,
                                    device = self.p['device'])
        
        return noising_process


    def init_optimizer_by_parameter(self, model):
        # training manager
        optimizer = optim.AdamW(model.parameters(), 
                                lr=self.p['optim']['lr'], 
                                betas=(0.9, 0.99)) # beta_2 0.95 instead of 0.999
        return optimizer

    def init_ls_by_parameter(self, optim):
        if self.p['optim']['schedule'] == None:
            return None
        
        if self.p['optim']['schedule'] == 'steplr':
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optim,
                                                        step_size= self.p['optim']['lr_step_size'], 
                                                        gamma= self.p['optim']['lr_gamma'], 
                                                        last_epoch=-1)
        else: 
            lr_scheduler = get_scheduler(
                self.p['optim']['schedule'],
                # "cosine",
                # "cosine_with_restarts",
                optimizer=optim,
                num_warmup_steps=self.p['optim']['warmup'],
                num_training_steps=self.p['optim']['lr_steps'],
            )
        return lr_scheduler


    def init_generation_manager_by_parameter(self, noising_process, dataloader):
        # here kwargs is passed to the underlying Generation Manager.
        kwargs = self.p['eval'][self.p['noising_process']]

        return Gen.GenerationManager(noising_process, 
                                    #reverse_steps=self.p['eval'][self.p['noising_process']]['reverse_steps'], 
                                    dataloader=dataloader, 
                                    is_image = is_image_dataset(self.p['data']['dataset']),
                                    **kwargs)


    def init_eval_by_parameter(self, noising_process, gen_manager, data, logger, gen_data_path, real_data_path):

        eval = Eval.Eval( 
                noising_process=noising_process,
                gen_manager=gen_manager,
                dataloader=data,
                verbose=True, 
                logger = logger,
                data_to_generate = self.p['eval']['data_to_generate'],
                batch_size = self.p['eval']['batch_size'],
                is_image = is_image_dataset(self.p['data']['dataset']),
                gen_data_path=gen_data_path,
                real_data_path=real_data_path
        )
        return eval

    def reset_model(self):
        model = self.init_model_by_parameter()
        optim = self.init_optimizer_by_parameter(model)
        learning_schedule = self.init_ls_by_parameter(model)
        return model, optim, learning_schedule

    def reset_vae(self):
        model_vae = self.init_model_vae_by_parameter()
        optim_vae = self.init_optimizer_by_parameter(model_vae) if model_vae is not None else None
        learning_schedule_vae = self.init_ls_by_parameter(optim_vae) if model_vae is not None else None
        return model_vae, optim_vae, learning_schedule_vae

    def init_manager_by_parameter(self,
                                  model,
                                model_vae,
                                data,
                                noising_process, 
                                optimizer,
                                optimizer_vae,
                                learning_schedule,
                                learning_schedule_vae,
                                eval, 
                                logger):
        
        # here kwargs goes to manager (ema_rates), train_loop (grad_clip), and eventually to training_losses (monte_carlo...)
        kwargs = self.p['training'][self.p['noising_process']]
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
                    reset_vae=self.reset_vae,
                    p = self.p,
                    eval_freq = self.p['run']['eval_freq'],
                    checkpoint_freq = self.p['run']['checkpoint_freq'],
                    # ema_rate, grad_clip
                    **kwargs
                    )
        return manager

    def init_experiment(self,
                        data,
                        test_data,
                        gen_data_path, 
                        real_data_path, 
                        logger = None):
         # intialize logger
        if logger is not None:
            logger.initialize(self.p)

        model = self.init_model_by_parameter()
        model_vae = self.init_model_vae_by_parameter()

        noising_process = self.init_noising_process_by_parameter()
        optim = self.init_optimizer_by_parameter(model)
        learning_schedule = self.init_ls_by_parameter(optim)
        optim_vae = self.init_optimizer_by_parameter(model_vae) if model_vae is not None else None
        learning_schedule_vae = self.init_ls_by_parameter(optim_vae) if model_vae is not None else None

        # get generation manager
        gen_manager = self.init_generation_manager_by_parameter(noising_process, data)

        # run evaluation on train or test data
        eval = self.init_eval_by_parameter(noising_process, gen_manager, data, logger, gen_data_path, real_data_path)
        
        # run training
        manager = self.init_manager_by_parameter(model,
                                            model_vae,
                                            data, 
                                            noising_process, 
                                            optim,
                                            optim_vae,
                                            learning_schedule,
                                            learning_schedule_vae,
                                            eval,
                                            logger,
                                            )
        return model, manager



''''''''''' FULL CLASS + LOADING/SAVING '''''''''''

class ExpUtils:

    def __init__(self, p, config_path = None) -> None:
        self.set_parameter(p, config_path = config_path)
    
    '''
        Set the parameters of the experiment. p can be a path to a config file or a dictionary of parameters
    '''
    def set_parameter(self, p, config_path = None):
        if isinstance(p, str): # config file
            self.p = FileHandler.get_param_from_config(config_path, p)
        elif isinstance(p, dict): # dictionnary
            self.p = p
        else:
            raise Exception('p should be a path to a config file or a dictionary of parameters. Got {}'.format(p))
        self.file_handler = FileHandler(self.p)
        self.init_utils = InitUtils(self.p)


    def prepare_experiment(self, logger = None, do_not_load_data=False):
        # prepare the evaluation directories
        if do_not_load_data:
            data, test_data, dataset_files, test_dataset_files = None, None, None, None
        else:
            data, test_data, dataset_files, test_dataset_files = self.init_utils.init_data_by_parameter()
        gen_data_path, real_data_path = self.file_handler.prepare_data_directories(dataset_files)
        model, manager =  self.init_utils.init_experiment(data, test_data, gen_data_path, real_data_path, logger)
        return model, data, test_data, manager

    def _load_experiment(self, 
                        model_path, 
                        eval_path, 
                        logger,
                        do_not_load_model = False,
                        do_not_load_data = False):
        model, data, test_data, manager = self.prepare_experiment(logger, do_not_load_data)
        if not do_not_load_model:
            print('loading from model file {}'.format(model_path))
            manager.load(model_path)
        print('loading from eval file {}'.format(eval_path))
        manager.load_eval_metrics(eval_path)
        #manager.losses = torch.load(eval_path)
        return model, data, test_data, manager

    # loads a model from some param as should be contained in folder_path.
    # Specify the training epoch at which to load; defaults to latest
    def load_experiment(self, 
                        folder_path, 
                        logger=None,
                        curr_epoch = None,
                        do_not_load_model = False,
                        do_not_load_data=False,
                        load_eval_subdir=False):
        model_path, _, eval_path = self.file_handler.get_paths_from_param(
                                                    folder_path, 
                                                    curr_epoch=curr_epoch,
                                                    new_eval_subdir = load_eval_subdir,
                                                    do_not_load_model=do_not_load_model)
        model, data, test_data, manager = self._load_experiment( 
                                                model_path, 
                                                eval_path, 
                                                logger,
                                                do_not_load_model=do_not_load_model,
                                                do_not_load_data=do_not_load_data)
        return model, data, test_data, manager


    # unique hash of parameters, append training epochs
    # simply separate folder by data distribution and alpha value
    def save_experiment(self, 
                        base_path, 
                        manager,
                        curr_epoch = None,
                        files = 'all',
                        save_new_eval=False): # will save eval and param in a subfolder.
        if isinstance(files, str):
            files = [files]
        for f in files:
            assert f in ['all', 'model', 'eval', 'param'], 'files must be one of all, model, eval, param'
        model_path, param_path, eval_path = self.file_handler.get_paths_from_param( 
                                                                base_path, 
                                                                make_new_dir = True, 
                                                                curr_epoch=curr_epoch,
                                                                new_eval_subdir=save_new_eval)
        #model_path = '_'.join([model_path, str(manager.training_epochs())]) 
        #losses_path = '_'.join([model_path, 'losses']) + '.pt'
        if 'all' in files:
            manager.save(model_path)
            manager.save_eval_metrics(eval_path)
            torch.save(self.p, param_path)
            return model_path, param_path, eval_path
        
        # else, slightly more complicated logic
        objects_to_save = {name: {'path': path, 'saved':False} for name, path in zip(['model', 'eval', 'param'],
                                                                        [model_path, eval_path, param_path])}
        for name, obj in objects_to_save.items():
            if name in files:
                obj['saved'] = True
                if name == 'model':
                    manager.save(obj['path'])
                elif name == 'eval':
                    manager.save_eval_metrics(obj['path'])
                elif name == 'param':
                    torch.save(self.p, obj['path'])
        
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



