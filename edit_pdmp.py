import PDMP.manage.Experiments as Exp
import PDMP.manage.Logger as Logger
import os
import yaml

from util_pdmp import *
from PDMP.manage.exp_utils import *

if __name__ == '__main__':

    dataset = 'mnist'
    exp_name = 'mnist_vae'
    # open and get parameters from file
    with open(os.path.join(CONFIG_PATH, dataset + '.yml'), "r") as f:
        p = yaml.safe_load(f)
    
    to_replace = {
        'transforms': 24,
        'hidden_width': 1024, # 2048
        'hidden_depth': 4,
        'time_emb_size': 32,
        'time_emb_type': 'learnable',
        'x_emb_size': 32,
        'x_emb_type': 'mlp', # mlp, unet
    }
    for k, v in to_replace.items():
        p['model']['normalizing_flow'][k] = v
    p['model']['vae'] = True

    # create experiment object. Specify directory to save and load checkpoints,
    # experiment parameters, and potential logger object
    exp = Exp.Experiment(os.path.join('models', exp_name), 
                         p, 
                         logger = None)
    


    # load if necessary. Must be done here in case we have different hashes afterward
    load_epoch = 100
    exp.load(epoch=load_epoch)
    
    to_replace = {
        'transforms': 24,
        'hidden_width': 128, # 2048
        'hidden_depth': 3,
        'time_emb_size': 16,
        'time_emb_type': 'learnable',
        'x_emb_size': 16,
        'x_emb_type': 'mlp', # mlp, unet
    }
    for k, v in to_replace.items():
        exp.p['model']['normalizing_flow'][k] = v
    exp.print_parameters()

    new_model, _, _, new_manager = prepare_experiment(exp.p)
    #exp.manager.model, exp.manager.optimizer, exp.manager.learning_schedule = reset_model(exp.p)
    new_manager.model_vae = exp.manager.model_vae
    new_manager.optimizer_vae = exp.manager.optimizer_vae
    new_manager.learning_schedule_vae = exp.manager.learning_schedule_vae
    exp.manager = new_manager
    exp.model = new_model

    print(exp.save())
    print(exp.save(curr_epoch = load_epoch))

    #exp.load()
    #exp.prepare()

    # print parameters
    #exp.print_parameters()
    
    # run the experiment
    #exp.run( 
    #    progress=p['run']['progress'],
    #    verbose = True,
    #    max_batch_per_epoch= args.n_max_batch,
    #    no_ema_eval=args.no_ema_eval, # to speed up testing
    #    )
    #
    ## in any case, save last models.
    #print(exp.save())
    #
    ## close everything
    #exp.terminate()

