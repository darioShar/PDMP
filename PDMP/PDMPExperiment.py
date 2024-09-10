import zuko

from bem.datasets import get_dataset, is_image_dataset
import PDMP.methods.Diffusion as Diffusion
import PDMP.models.Model as Model
import PDMP.methods.pdmp as PDMP
import PDMP.models.unet as unet
import PDMP.models.NormalizingFlow as NormalizingFLow
import PDMP.models.VAE as VAE
import PDMP.methods.NF as NF
from bem.utils.exp import InitUtils




def model_param_to_use(p):
    if p['method'] == 'diffusion':
        if is_image_dataset(p['data']['dataset']):
            return p['model']['unet']
        else:
            return p['model']['mlp']
    elif p['method'] == 'nf':
        return p['model']['nf']
    elif p['pdmp']['sampler'] == 'ZigZag':
        if is_image_dataset(p['data']['dataset']):
            return p['model']['unet']
        else:
            return p['model']['mlp']
    else:
        return p['model']['normalizing_flow']


''''''''''' FILE MANIPULATION '''''''''''
def exp_hash(p):
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
            p['method']: {k:v for k, v in p[p['method']].items()},
            'model':  {k:v for k, v in model_param.items() if not k in retro_compatibility}, # here retro-compatibility
            #'optim': p['optim'],
            #'training': p['training']
            }
    return to_hash
    

''' INITIALIZATION UTILS '''

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

def init_model_default_by_parameter(p):
    # model
    model_param = model_param_to_use(p)
    method = p['method'] if p['method'] in ['diffusion', 'nf'] else p['pdmp']['sampler']
    if not is_image_dataset(p['data']['dataset']):
        # model
        if method in ['diffusion', 'ZigZag']:
            model = Model.MLPModel(nfeatures = p['data']['dim'],
                                    device=p['device'], 
                                    p_model_mlp=model_param,
                                    method=method,
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
    method = p['method'] if p['method'] in ['diffusion', 'nf'] else p['pdmp']['sampler']
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
                    batch_size=p['data']['batch_size'], 
                    shuffle=True, 
                    num_workers=p['data']['num_workers'])
    test_data = DataLoader(test_dataset_files,
                            batch_size=p['data']['batch_size'],
                            shuffle=True,
                            num_workers=p['data']['num_workers'])
    return data, test_data, dataset_files, test_dataset_files

def init_method_by_parameter(p):
    #gammas = Diffusion.LevyDiffusion.gen_noise_schedule(p['diffusion']['diffusion_steps']).to(p['device'])
    if p['method'] == 'pdmp':
        method = PDMP.PDMP(
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
    elif p['method'] == 'diffusion':
        method = Diffusion.LevyDiffusion(alpha = p['diffusion']['alpha'],
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
    elif p['method'] == 'nf':
        method = NF.NF(reverse_steps = 1,
                                device = p['device'])
    
    return method


def init_optimizer_by_parameter(p, model):
    return InitUtils.init_default_optimizer(p, model)

def init_ls_by_parameter(p, optim):
    return InitUtils.init_default_ls(p, optim)

def init_models_by_parameter(p):
    models = {
        'default': init_model_default_by_parameter(p)
        }
    optimizers = {
        'default': init_optimizer_by_parameter(p, models['default']),
        }
    learning_schedules = {
        'default': init_ls_by_parameter(p, optimizers['default'])
        }
    
    model_vae = init_model_vae_by_parameter(p)
    if model_vae is not None:
        models['vae'] = model_vae
        optimizers['vae'] = init_optimizer_by_parameter(p, models['vae'])
        learning_schedules['vae'] = init_ls_by_parameter(p, optimizers['vae'])

    return models, optimizers, learning_schedules

def reset_models(p):
    return init_models_by_parameter(p)

