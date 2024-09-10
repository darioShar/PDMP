import dem.manage.Experiments as Exp
import dem.manage.Logger as Logger
import dem.manage.exp_utils as exp_utils
import os
import yaml

from dem.utils import *

def run_exp(config_folder):
    args = parse_args()

    # open and get parameters from file
    p = exp_utils.FileHandler.get_param_from_config(args.config + '.yml', config_folder)

    update_parameters_before_loading(p, args)

    # create experiment object. Specify directory to save and load checkpoints, experiment parameters, and potential logger object
    exp = Exp.Experiment(os.path.join('models', args.name), 
                         p, 
                         logger = Logger.NeptuneLogger() if args.log else None)
    

    # load if necessary. Must be done here in case we have different hashes afterward
    if args.resume:
        if args.resume_epoch is not None:
            exp.load(epoch=args.resume_epoch)
        else:
            exp.load()
    else:
        exp.prepare()
    
    update_experiment_after_loading(exp, args)
    additional_logging(exp, args)
    exp.print_parameters()
    
    # run the experiment
    exp.run( 
        progress=p['run']['progress'],
        max_batch_per_epoch= args.n_max_batch,
        no_ema_eval=args.no_ema_eval, # to speed up testing
        )
    
    # in any case, save last models.
    print(exp.save(curr_epoch=p['run']['epochs']))
    
    # close everything
    exp.terminate()

