import numpy as np
import torch
import DDLM.evaluate.Eval as Eval
import DDLM.manage.exp_utils.prepare_utils as prepare_utils
import DDLM.manage.exp_utils.load_save_utils as load_save_utils
from DDLM.datasets import is_image_dataset
from DDLM.manage.Generate import GenerationManager

CONFIG_PATH = 'DDLM/configs/'

def check_dict_eq(dic1, dic2):
    for k, v in dic1.items():
        if isinstance(v, dict):
            check_dict_eq(v, dic2[k])
        elif isinstance(v, torch.Tensor):
            if (v != dic2[k]).any():
                return False
        else:
            if v != dic2[k]:
                return False
    return True

def _get_device():
    if torch.backends.mps.is_available():
        device = "mps"
        mps_device = torch.device(device)
    elif torch.cuda.is_available():
        device = "cuda"
        cuda_device = torch.device(device)
    else:
        device = 'cpu'
        print ("GPU device not found.")
    print ('using device {}'.format(device))
    return device

def _optimize_gpu(device):
    if device == 'cuda':
        #torch.set_default_tensor_type(torch.cuda.FloatTensor)
        torch.backends.cudnn.benchmark = True

def _set_seed(seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
        #torch.backends.cudnn.deterministic = True

class Experiment:

    @staticmethod
    def get_parameters_from_dir(dir):
        return load_save_utils.load_params_from_folder(dir)
    
    def _set_parameter(self, p):
        if isinstance(p, str): # config file
            self.p = load_save_utils.load_param_from_config(CONFIG_PATH, p)
        elif isinstance(p, dict): # dictionnary
            self.p = p
        else:
            raise Exception('p should be a path to a config file or a dictionary of parameters. Got {}'.format(p))

    def _reset_attributes(self,
                         p,
                         checkpoint_dir,
                         logger):
        if (self.logger is not None) and (self.logger != logger) and (logger is not None):
            self.logger.stop()
            self.logger = logger
        elif (self.logger is None) and (logger is not None):
            self.logger = logger
        if checkpoint_dir is not None:
            self.checkpoint_dir = checkpoint_dir
        if p is not None:
            self._set_parameter(p)
    
    def _initialize(self):
        device = _get_device()
        self.p['device'] = device
        _optimize_gpu(device)
        if self.p['seed'] is not None:
            _set_seed(self.p['seed'], device)

    # p is used to prepare/setup the experiment. p can be a directory or a path to a config file
    # we specify a directory path from where to load/save checkpoints
    # and a potential logger, e.g NeptuneLogger()
    def __init__(self,
                 checkpoint_dir,
                 p,
                 logger = None):
        self.logger = None
        self._reset_attributes(p, checkpoint_dir, logger)
        self._initialize()
        self.prepare()

    # this lets us change some parameters many time on the fly and run variations of experiments rapidly
    def prepare(self, 
                p = None, 
                checkpoint_dir = None, 
                logger = None):
        self._reset_attributes(p, checkpoint_dir, logger)
        self.model, self.data, self.test_data, self.manager = prepare_utils.prepare_experiment(self.p, self.logger)
    
    # load experiment values from checkpoint dir
    def load(self, 
             p = None,
             checkpoint_dir = None,
             logger = None,
             epoch = None):
        self._reset_attributes(p, checkpoint_dir, logger)
        self.model, self.data, self.test_data, self.manager = \
            load_save_utils.load_experiment_from_param(self.p, 
                                       self.checkpoint_dir, 
                                       self.logger,
                                       curr_epoch=epoch)

    # save to checkpoint dir
    def save(self, curr_epoch = None):
        return load_save_utils.save_experiment(self.p, self.checkpoint_dir, self.manager, curr_epoch) 

    # training, checkpoints, closing logger and saving.
    # attention: eval_freq corresponds to freq of eval in each ckeckpoint_freq loop
    def run(self, 
            epochs, 
            eval_freq, 
            checkpoint_freq = None,
            progress = False,
            progress_batch = False,
            **kwargs): # bs, lr, eval_freq, Lploss...
        
        assert eval_freq > 0 or (eval_freq is None)
        assert checkpoint_freq > 0 or (checkpoint_freq is None)
        # if they are none, set them above the number of requested epochs, so nothing happens
        if eval_freq is None:
            eval_freq = epochs + 1
        if checkpoint_freq is None:
            checkpoint_freq = epochs + 1
        # if checkpoint_freq is not None, epochs should be a multiple of it
        #assert (checkpoint_freq is None) or (epochs % checkpoint_freq == 0)
        import tqdm
        if progress:
            tqdm.tqdm._instances.clear()
            pbar = tqdm.tqdm(total = epochs)

        # in order to run a maximum number of epochs dún coup dún seul.
        to_next_eval = eval_freq
        to_next_checkpoint = checkpoint_freq

        import copy
        while epochs > 0:
            n_epochs = min(to_next_eval, to_next_checkpoint, epochs)
            self.manager.train(nepochs = n_epochs, 
                               progress_batch = progress_batch,
                               epoch_pbar = pbar if progress else None, # pbar for epochs
                               **kwargs)
            

            if to_next_checkpoint == n_epochs:
                to_next_checkpoint = checkpoint_freq
                print(self.save(curr_epoch=self.manager.training_epochs()))
            else:
                to_next_checkpoint -= n_epochs
            if to_next_eval == n_epochs:
                to_next_eval = eval_freq
                self.manager.evaluate()
                self.manager.evaluate_emas()
            else:
                to_next_eval -= n_epochs
            epochs -= n_epochs
        if progress:
            pbar.close()
            tqdm.tqdm._instances.clear()
        # in any case, save last models.
        print(self.save())

    def get_generator(self):
        return GenerationManager(self.exp.model, 
                                self.exp.manager.diffusion, 
                                self.exp.data)

    def terminate(self):
        if self.logger:
            self.logger.stop()



    '''def evaluate_all_emas(self,
                         ddim = None,
                         eval_eta = None,
                         reduce_timesteps = None,
                         data_to_generate = None, # will default to p
                         verbose = True):
        eval = Eval.Eval(self.model, 
            self.manager.diffusion, 
            self.data,
            verbose=verbose, 
            logger = None,#, self.manager.logger, DO NOT LOG THIS EVAL RUN
            ddim = self.p['eval']['ddim'] if ddim is not None else ddim,
            eval_eta = self.p['eval']['eval_eta'] if eval_eta is not None else eval_eta,
            reduce_timesteps = self.p['eval']['reduce_timesteps'] if reduce_timesteps is not None else reduce_timesteps,
            data_to_generate = self.p['eval']['data_to_generate'] if data_to_generate is not None else data_to_generate,
            is_image = is_image_dataset(self.p['data']['dataset']))
        self.manager.evaluate_emas(eval)
        return eval'''