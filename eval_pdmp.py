import os
import yaml
import PDMP.manage.Experiments as Exp
import PDMP.manage.Logger as Logger
import os
import yaml

from util_pdmp import *

from PDMP.datasets import is_image_dataset


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

    for epoch in range(args.eval, args.epochs + 1, args.eval):
        print(f'Evaluating epoch {epoch}')
        exp.load(epoch=epoch)
        # change some parameters before the run.
        update_experiment_after_loading(exp, args)
        if not args.ema_eval:
            exp.manager.evaluate()
        if not args.no_ema_eval:
            exp.manager.evaluate_emas()
        # if is image, we would rather have a separate folder per evaluation, since we won't be looking
        # at multiple evaluations during a single run, and doing operation on such a time series.
        if is_image_dataset(exp.p['data']['dataset']):
            tmp = exp.save(files=['eval', 'param'], save_new_eval=True, curr_epoch=epoch)
            print('Saved ', tmp)
        else:
            tmp = exp.save(files=['eval', 'param'], save_new_eval=False, curr_epoch=None)
            print('Saved ', tmp)
    
    # close everything
    exp.terminate()

