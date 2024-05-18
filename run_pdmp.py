import PDMP.manage.Experiments as Exp
import PDMP.manage.Logger as Logger
import os
import yaml

from util_pdmp import *

if __name__ == '__main__':
    args = parse_args()

    # open and get parameters from file
    with open(os.path.join(CONFIG_PATH, args.config + '.yml'), "r") as f:
        p = yaml.safe_load(f)
    
    update_parameters_before_loading(p, args)

    # create experiment object. Specify directory to save and load checkpoints,
    # experiment parameters, and potential logger object
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
    
    # if config file has changed, update that
    # TODO: check if the config file has changed and update run parameters
    # ex: will not change optimizer for the moment, will use the checkpointed one.
    # change some parameters before the run.
    update_experiment_after_loading(exp, args)
    
    additional_logging(exp, args)

    # print parameters
    exp.print_parameters()
    
    # run the experiment
    exp.run( 
        progress=p['run']['progress'],
        max_batch_per_epoch= args.n_max_batch,
        no_ema_eval=args.no_ema_eval, # to speed up testing
        )
    
    # in any case, save last models.
    print(exp.save(curr_epoch=p['run']['epochs'])) #curr_epoch=p['run']['epochs']
    
    # close everything
    exp.terminate()

