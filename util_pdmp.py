import argparse

from PDMP.manage.exp_utils import init_ls_by_parameter, hash_parameters, hash_parameters_eval

CONFIG_PATH = 'PDMP/configs'

def update_parameters_before_loading(p, args):
    # These parameters should be changed for this specific run, before objects are loaded

    if args.noising_process is not None:
        p['noising_process'] = args.noising_process

    # DIFFUSION
        
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

    if args.gmm is not None:
        p['data']['dataset'] = 'gmm_grid'
    
    if args.stable is not None:
        p['data']['dataset'] = 'sas_grid'

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

    # Now for pdmp
    if args.sampler is not None:
        p['pdmp']['sampler'] = args.sampler
    
    if args.time_horizon is not None:
        p['pdmp']['time_horizon'] = args.time_horizon
    
    return p

def update_experiment_after_loading(exp, args):
    # change some parameters for the run.
    # These parameters should act on the objects already loaded from  the previous runs
    
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

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--noising_process", help='noising process to use', default=None, type=str)

    # diffusion
    parser.add_argument('-r', "--resume", help="resume existing experiment", action='store_true', default=False)
    parser.add_argument("--config", help='config file', type=str, required=True)
    parser.add_argument("--name", help='name of the experiment', type=str, required=True)
    parser.add_argument('--lr', help='reinitialize learning rate', type=float, default = None)
    parser.add_argument('--lr_steps', help='reinitialize learning rate steps', type=int, default = None)
    parser.add_argument('--log', help='activate logging', action='store_true', default=False)
    parser.add_argument('--job_id', help='slurm job id', default=None, type = str)
    parser.add_argument('--alpha', help='alpha value to train for', default=None, type = float)
    parser.add_argument('--epochs', help='epochs', default=None, type = int)
    parser.add_argument('--eval', help='evaluation frequency', default=None, type = int)
    parser.add_argument('--check', help='checkpoint frequency', default=None, type = int)
    parser.add_argument('--n_max_batch', help='max batch per epoch (speed up testing)', default=None, type = int)
    parser.add_argument('--no_ema_eval', help='dont evaluate ema models', action='store_true', default = False)
    parser.add_argument('--LIM', help='activate LIM training/sampling', action='store_true', default = False)
    parser.add_argument('--non_iso', help='use non isotropic noise in the diffusion', action='store_true', default = False)
    parser.add_argument('--non_iso_data', help='use non isotropic data', action='store_true', default = False)
    parser.add_argument('--median', help='use median of mean. Specify (total samples, number of groups). Must have total%groups==0', nargs ='+', default = None)
    parser.add_argument('--set_seed', help='set random seed', default = None, type=int)
    parser.add_argument('--random_seed', help='set random seed to a random number', action = 'store_true', default=None)
    parser.add_argument('--lploss', help='set p in lploss', default = None, type = float)
    parser.add_argument('--variance', help='learn variance', default = False, action='store_true')
    parser.add_argument('--resume_epoch', help='epoch from which to resume', default = None, type=int)
    parser.add_argument('--generate', help='how many images/datapoints to generate', default = None, type = int)
    parser.add_argument('--gmm', help='if 2d data, loads a gmm', default = None, action='store_true')
    parser.add_argument('--stable', help='if 2d data, loads a stable mm', default = None, action='store_true')
    parser.add_argument('--reverse_steps', help='choose number of reverse_steps', default = None, type = int)

    # now for pdmp
    parser.add_argument('--sampler', help='choose sampler for PDMP', default = None, type = str)
    parser.add_argument('--time_horizon', help='choose time horizon for PDMP', default = None, type = int)


    # specific to evaluation
    parser.add_argument('--reset_eval', help='reset eval dictionnary', action='store_true', default = False)
    parser.add_argument('--ema_eval', help='evaluate only ema models', action='store_true', default = False)
    parser.add_argument('--ddim', help='use ddim for sampling', default = False, action='store_true')
    parser.add_argument('--clip', help='use clip denoised', default = False, action='store_true')

    args = parser.parse_args()

    assert not (args.gmm and args.stable), 'Cannot load both a gmm and a stable mm'
    assert (args.no_ema_eval and args.ema_eval) == False, 'No evaluation to make'

    return args