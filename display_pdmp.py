import os
import yaml
import PDMP.manage.Experiments as Exp
import PDMP.manage.Logger as Logger
import os
import yaml
import matplotlib.pyplot as plt

from util_pdmp import *


SAVE_ANIMATION_PATH = './animation'


if __name__ == '__main__':
    args = parse_args()

    # open and get parameters from file
    with open(os.path.join(CONFIG_PATH, args.config + '.yml'), "r") as f:
        p = yaml.safe_load(f)
    
    update_parameters_before_loading(p, args)

    # create experiment object. Specify directory to save and load checkpoints, experiment parameters, and potential logger object
    exp = Exp.Experiment(os.path.join('models', args.name), 
                         p, 
                         logger = Logger.NeptuneLogger() if args.log else None)

    exp.prepare()
        
    additional_logging(exp, args)

    # print parameters
    exp.print_parameters()

    print('Loading latest model')
    exp.load()
    
    update_experiment_after_loading(exp, args)

    # some information
    run_info = [exp.p['data']['dataset'], exp.manager.noising_process.reverse_steps, exp.manager.total_steps]
    title = '{}, reverse_steps={}, training_steps={}'.format(*run_info[:3])
    
    # display plot and animation, for a specific model
    anim = exp.manager.display_plots(ema_mu=None, # can specify ema rate, if such a model has been trained
                                plot_original_data=False, 
                                title=title,
                                nb_datapoints=25000, # number of points to display.
                                marker='.', # '.' marker displays pixel-wide points.
                                color='blue', # color of the points
                                xlim = (-.5, 1.5), # x-axis limits
                                ylim = (-.5, 1.5), # y-axis limits
                                alpha = 1.0,
                                )
    # save animation
    path = os.path.join(SAVE_ANIMATION_PATH, '_'.join([str(x) for x in run_info]))
    anim.save(path + '.mp4')
    print('Animation saved in {}'.format(path))

    # stops the thread from continuing
    plt.show()

    # close everything
    exp.terminate()

