import argparse

from PDMP.manage.exp_utils import init_ls_by_parameter, hash_parameters, hash_parameters_eval
from PDMP.datasets import is_image_dataset

CONFIG_PATH = 'PDMP/configs'


# These parameters should be changed for this specific run, before objects are loaded
def update_parameters_before_loading(p, args):
    if args.noising_process is not None:
        p['noising_process'] = args.noising_process
        
    # if alpha is specified, change the parameter, etc.
    if args.alpha is not None:
        p['diffusion']['alpha'] = args.alpha
    
    if args.epochs is not None:
        p['run']['epochs'] = args.epochs
    
    if args.eval is not None:
        p['run']['eval_freq'] = args.eval

    if args.check is not None:
        p['run']['checkpoint_freq'] = args.check

    if args.LIM:
        p['diffusion']['LIM'] = True
    
    if args.non_iso:
        p['diffusion']['isotropic'] = False 

    if args.non_iso_data:
        p['data']['isotropic'] = False
    
    if args.set_seed is not None:
        p['seed'] = args.set_seed

    if args.random_seed is not None:
        p['seed'] = None

    if args.lploss is not None:
        p['training']['lploss'] = args.lploss

    if args.variance:
        p['diffusion']['var_predict'] = 'GAMMA'
        p['training']['loss_type'] = 'VAR_KL'
        p['model']['compute_gamma'] = True

    if args.median is not None:
        assert len(args.median) == 2
        p['training']['loss_monte_carlo'] = 'median'
        p['training']['monte_carlo_steps'] = int(args.median[0])
        p['training']['monte_carlo_groups'] = int(args.median[1])
    
    if args.ddim:
        p['eval']['ddim'] = True
        p['eval']['eval_eta'] = 0.

    if args.clip:
        p['eval']['clip_denoised'] = True

    if args.generate is not None:
        #assert False, 'NYI. eval_files are stored in some folder, and the prdc and fid functions consider all the files in a folder. So if a previous run had generated more data, there is a contamination. To be fixed'
        p['eval']['data_to_generate'] = args.generate
    
    # will do the neceassary changes after loading
    if args.lr is not None:
        p['optim']['lr'] = args.lr
    
    if args.lr_steps is not None:
        p['optim']['lr_steps'] = args.lr_steps

    if args.reverse_steps is not None:
        p['eval'][p['noising_process']]['reverse_steps'] = args.reverse_steps


    if args.dataset is not None:
        p['data']['dataset'] = args.dataset

    # Now for pdmp
    if args.sampler is not None:
        p['pdmp']['sampler'] = args.sampler
    
    if args.time_horizon is not None:
        p['pdmp']['time_horizon'] = args.time_horizon

    if args.refresh_rate is not None:
        assert args.refresh_rate >= 0.
        p['pdmp']['refresh_rate'] = args.refresh_rate

    if args.scheme is not None:
        p['eval']['pdmp']['backward_scheme'] = args.scheme
    
    add_losses = set(p['pdmp']['add_losses'] if p['pdmp']['add_losses'] is not None else [])
    for l in args.loss:
        add_losses.add(l)
    p['pdmp']['add_losses'] = sorted(list(add_losses))

    if args.exponent is not None:
        p['eval']['pdmp']['exponent'] = args.exponent


    if args.vae is not None:
        p['model']['vae'] = args.vae
    
    if args.train_type is not None:
        p['training']['pdmp']['train_type'] = args.train_type
    
    if args.train_alternate is not None:
        p['training']['pdmp']['train_alternate'] = args.train_alternate
    
    # model
    if args.blocks is not None:
        p['model']['mlp']['nblocks'] = args.blocks
    
    if args.units is not None:
        p['model']['mlp']['nunits'] = args.units
    
    if args.transforms is not None:
        p['model']['normalizing_flow']['transforms'] = args.transforms
    
    if args.depth is not None:
        p['model']['normalizing_flow']['hidden_depth'] = args.depth
    
    if args.width is not None:
        p['model']['normalizing_flow']['hidden_width'] = args.width

    if args.t_embedding_type is not None:
        p['model']['normalizing_flow']['time_emb_type'] = args.t_embedding_type
        p['model']['mlp']['time_emb_type'] = args.t_embedding_type

    if args.t_embedding_size is not None:
        p['model']['normalizing_flow']['time_emb_size'] = args.t_embedding_size
        p['model']['mlp']['time_emb_size'] = args.t_embedding_size

    if args.x_embedding_type is not None:
        p['model']['normalizing_flow']['x_emb_type'] = args.x_embedding_type

    if args.x_embedding_size is not None:
        p['model']['normalizing_flow']['x_emb_size'] = args.x_embedding_size
    
    if args.nf_model_type is not None:
        p['model']['normalizing_flow']['model_type'] = args.nf_model_type

    if args.beta is not None:
        if not is_image_dataset(p['data']['dataset']):
            p['model']['mlp']['beta'] = args.beta
        else:
            p['model']['unet']['beta'] = args.beta


    if args.use_softmax:
        p['additional']['use_softmax'] = args.use_softmax
    
    # model vae
    if args.model_vae_type is not None:
        p['model']['normalizing_flow']['model_vae_type'] = args.model_vae_type

    if args.vae_t_embedding_hidden_width is not None:
        p['model']['normalizing_flow']['vae_t_emb_hidden_width'] = args.vae_t_embedding_hidden_width

    if args.vae_t_embedding_size is not None:
        p['model']['normalizing_flow']['vae_t_emb_size'] = args.vae_t_embedding_size

    if args.vae_x_embedding_size is not None:
        p['model']['normalizing_flow']['vae_x_emb_size'] = args.vae_x_embedding_size

    return p


# change some parameters for the run.
# These parameters should act on the objects already loaded from the previous runs
def update_experiment_after_loading(exp, args):
    # scheduler
    schedule_reset = False 
    if args.lr is not None:
        schedule_reset = True
        for param_group in exp.manager.optimizer.param_groups:
            param_group['lr'] = args.lr
        exp.p['optim']['lr'] = args.lr
    if args.lr_steps is not None:
        schedule_reset = True
        exp.p['optim']['lr_steps'] = args.lr_steps
    
    if schedule_reset:
        lr_scheduler = init_ls_by_parameter(exp.manager.optimizer, exp.p)
        exp.manager.learning_schedule = lr_scheduler

# some additional logging 
def additional_logging(exp, args):
    # logging job id
    if (exp.manager.logger is not None) and (args.job_id is not None):
        exp.manager.logger.log('job_id', args.job_id)
    
    # logging hash parameter
    if (exp.manager.logger is not None):
        exp.manager.logger.log('hash_parameter', hash_parameters(exp.p))
    
    # logging hash eval
    if (exp.manager.logger is not None):
        exp.manager.logger.log('hash_eval', hash_parameters_eval(exp.p))
    
    # starting epoch and batch
    if (exp.manager.logger is not None):
        exp.manager.logger.log('starting_epoch', exp.manager.training_epochs())
        exp.manager.logger.log('starting_batch', exp.manager.training_batches())


# define and parse the arguments
def parse_args():
    parser = argparse.ArgumentParser()

    # processes to choose from. Either diffusion, pdmp, or 'nf' to use a normal normalizing flow.
    parser.add_argument("--noising_process", help='noising process to use', default=None, type=str, choices=['diffusion', 'pdmp', 'nf'])

    # EXPERIMENT parameters, specific to TRAINING
    parser.add_argument("--config", help='config file to use', type=str, required=True)
    parser.add_argument("--name", help='name of the experiment. Defines save location: ./models/name/', type=str, required=True)
    parser.add_argument('--epochs', help='epochs', default=None, type = int)
    parser.add_argument('-r', "--resume", help="resume existing experiment", action='store_true', default=False)
    parser.add_argument('--resume_epoch', help='epoch from which to resume', default = None, type=int)
    parser.add_argument('--eval', help='evaluation frequency', default=None, type = int)
    parser.add_argument('--check', help='checkpoint frequency', default=None, type = int)
    parser.add_argument('--n_max_batch', help='max batch per epoch (to speed up testing)', default=None, type = int)

    parser.add_argument('--set_seed', help='set random seed', default = None, type=int)
    parser.add_argument('--random_seed', help='set random seed to a random number', action = 'store_true', default=None)

    parser.add_argument('--log', help='activate logging to neptune', action='store_true', default=False)
    parser.add_argument('--job_id', help='slurm job id', default=None, type = str)

    # if training together with a VAE
    parser.add_argument('--train_type', help='Which training to use', default = None, type=str)
    parser.add_argument('--train_alternate', help='altenrate normal and normal_with_vae training', default = None, action='store_true')


    # EXPERIMENT parameters, specific to EVALUATION
    parser.add_argument('--ema_eval', help='evaluate all ema models', action='store_true', default = False)
    parser.add_argument('--no_ema_eval', help='dont evaluate ema models', action='store_true', default = False)
    parser.add_argument('--generate', help='how many images/datapoints to generate', default = None, type = int)
    parser.add_argument('--reverse_steps', help='choose number of reverse_steps', default = None, type = int)
    parser.add_argument('--exponent', help='exponent in reverse_steps', default = None, type = float)
    parser.add_argument('--reset_eval', help='reset evaluation metrics', action='store_true', default = False)

    parser.add_argument('--ddim', help='use ddim for sampling (diffusion)', default = False, action='store_true')
    parser.add_argument('--clip', help='use clip denoised (diffusion)', default = False, action='store_true')

    # DATA
    parser.add_argument('--dataset', help='choose specific dataset', default = None, type = str)

    # OPTIMIZER
    parser.add_argument('--lr', help='reinitialize learning rate', type=float, default = None)
    parser.add_argument('--lr_steps', help='reinitialize learning rate steps', type=int, default = None)

    # MODEL
    parser.add_argument('--blocks', help='choose number of blocks in mlp', default = None, type = int)
    parser.add_argument('--units', help='choose number of units in mlp', default = None, type = int)
    parser.add_argument('--transforms', help='choose number of transforms in neural spline flow', default = None, type = int)
    parser.add_argument('--depth', help='choose depth in neural spline flow', default = None, type = int)
    parser.add_argument('--width', help='choose width in neural spline flow', default = None, type = int)
    parser.add_argument('--t_embedding_type', help='choose time embedding type', default = None, type = str)
    parser.add_argument('--t_embedding_size', help='choose time embedding size', default = None, type = int)
    parser.add_argument('--x_embedding_type', help='choose x embedding type', default = None, type = str)
    parser.add_argument('--x_embedding_size', help='choose x embedding size', default = None, type = int)
    parser.add_argument('--nf_model_type', help='Choose normalizing_flow model type', default = None, type = str)
    parser.add_argument('--beta', help='for softplus zigzag', default = None, type = float)
    
    parser.add_argument('--use_softmax', help='use softmax in ZigZag', default = False, action='store_true')

    # VAE MODEL
    parser.add_argument('--vae', help='Use vae', default = None, action='store_true')
    parser.add_argument('--model_vae_type', help='Choose VAE normalizing_flow model type (1 or 16)', default = None, type = str)
    parser.add_argument('--vae_t_embedding_hidden_width', help='Choose VAE normalizing_flow time embedding hidden layer size', default = None, type = int)
    parser.add_argument('--vae_t_embedding_size', help='Choose VAE normalizing_flow time embedding output size', default = None, type = int)
    parser.add_argument('--vae_x_embedding_size', help='Choose VAE normalizing_flow x embedding output size', default = None, type = int)

    # PDMP
    parser.add_argument('--sampler', help='choose sampler for PDMP', default = None, type = str)
    parser.add_argument('--time_horizon', help='choose time horizon for PDMP', default = None, type = int)
    parser.add_argument('--refresh_rate', help='refresh rate for pdmp', default = None, type = float)
    parser.add_argument('--scheme', help='choose scheme', default = None, type = str)
    parser.add_argument('--loss', help='Choose the losses to use (will be added to each other if multiple ones are given)', required=True, type = str, nargs='+',
                        choices = ['square', 'kl', 'logistic', 'hyvarinen', 'ml', 'hyvarinen_simple', 'kl_simple'])
    
    # DIFFUSION
    parser.add_argument('--alpha', help='alpha value for diffusion', default=None, type = float)
    parser.add_argument('--LIM', help='activate LIM training/sampling', action='store_true', default = False)
    parser.add_argument('--non_iso', help='use non isotropic noise in the diffusion', action='store_true', default = False)
    parser.add_argument('--non_iso_data', help='use non isotropic data', action='store_true', default = False)
    parser.add_argument('--median', help='use median of mean. Specify (total samples, number of groups). Must have total%groups==0', nargs ='+', default = None)
    parser.add_argument('--lploss', help='set p in lploss', default = None, type = float)
    parser.add_argument('--variance', help='learn variance', default = False, action='store_true')

    # not using this anymore
    #parser.add_argument('--subsamples', help='subsampling for ZigZag', default = None, type=int)

    # PARSE AND RETURN
    args = parser.parse_args()
    assert (args.no_ema_eval and args.ema_eval) == False, 'No possible evaluation to make'
    return args