import os
import bem.Experiments as Exp
from bem.utils_exp import *
from script_utils import *
import PDMP.PDMPExperiment as pdmp_exp
from PDMP.NeptuneLogger import NeptuneLogger

SAVE_ANIMATION_PATH = './animation'


def eval_exp(config_path):
    args = parse_args()
    # open and get parameters from file
    p = FileHandler.get_param_from_config(config_path, args.config + '.yml')

    update_parameters_before_loading(p, args)

    # create experiment object. Specify directory to save and load checkpoints, experiment parameters, and potential logger object
    checkpoint_dir = os.path.join('models', args.name)
    # the ExpUtils class specifies how to hash the parameter dict, and what and how to initiliaze methods and models
    exp = Exp.Experiment(checkpoint_dir=checkpoint_dir, 
                        p=p,
                        logger = NeptuneLogger() if args.log else None,
                        exp_hash= pdmp_exp.exp_hash, 
                        eval_hash=None, # will use default function
                        init_method_by_parameter= pdmp_exp.init_method_by_parameter,
                        init_models_by_parameter= pdmp_exp.init_models_by_parameter,
                        reset_models= pdmp_exp.reset_models)

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

if __name__ == '__main__':
    config_path = 'PDMP/configs/'
    eval_exp(config_path)