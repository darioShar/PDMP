import os
import yaml
import dem.manage.Experiments as Exp
import manage.Logger as Logger
import dem.manage.exp_utils as exp_utils
import os
import yaml
import matplotlib.pyplot as plt

from dem.utils import *


SAVE_ANIMATION_PATH = './animation'


def eval_exp(config_folder):
    args = parse_args()

    # open and get parameters from file
    p = exp_utils.FileHandler.get_param_from_config(args.config + '.yml', config_folder)
    
    update_parameters_before_loading(p, args)

    # create experiment object. Specify directory to save and load checkpoints, experiment parameters, and potential logger object
    exp = Exp.Experiment(os.path.join('models', args.name), 
                         p, 
                         logger = Logger.NeptuneLogger() if args.log else None)

    if args.reset_eval:
        print('Resetting eval dictionnary')
        exp.load()
        exp.manager.eval.reset(keep_losses=True, keep_evals=False)
        exp.save(files='eval')
        print('Eval dictionnary reset and saved.')
    else:
        exp.prepare()
        
    additional_logging(exp, args)

    # print parameters
    exp.print_parameters()

    # evlauate at different checkpointed epochs
    for epoch in range(args.eval, args.epochs + 1, args.eval):
        print('Evaluating epoch {}'.format(epoch))
        exp.load(epoch=epoch)
        # change some parameters before the run.
        update_experiment_after_loading(exp, args)
        if not args.ema_eval:
            exp.manager.evaluate(evaluate_emas=False)
        if not args.no_ema_eval:
            exp.manager.evaluate(evaluate_emas=True)
        tmp = exp.save(files=['eval', 'param'], save_new_eval=True, curr_epoch=epoch)
        print('Saved (model, eval, param) in ', tmp)

    # close everything
    exp.terminate()

