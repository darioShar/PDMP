import numpy as np
import torch
import matplotlib.pyplot as plt
import PDMP.compute.TrainLoop as TrainLoop
import PDMP.models.Ema as Ema
import copy
#Wrapper around training functions


class Manager:
        
    def __init__(self, 
                 model, 
                 data,
                 pdmp, 
                 optimizer,
                 learning_schedule,
                 eval,
                 ema_rates,
                 logger = None,
                 **kwargs):
        self.train_loop = TrainLoop.TrainLoop()
        self.model = model
        self.data = data
        self.pdmp = pdmp
        self.optimizer = optimizer
        self.learning_schedule = learning_schedule
        if ema_rates is None:
            self.ema_models = None
            self.ema_evals = None
        else:
            self.ema_models = [Ema.EMAHelper(model, mu = mu) for mu in ema_rates]
            logger = eval.logger
            # need to set logger to None for the deepcopy
            eval.logger = None
            self.ema_evals = [(copy.deepcopy(eval), mu) for mu in ema_rates]
            eval.logger = logger 
            for ema_eval, mu in self.ema_evals:
                ema_eval.logger = logger
        self.eval = eval
        self.kwargs = kwargs
        self.logger = logger
    
    def __getitem__(self, key):
        return self.eval.evals[key]
    
    def __setitem(self, key, value):
        self.eval.evals[key] = value
            
    
    def train(self, **kwargs):
        tmp_kwargs = self.kwargs
        tmp_kwargs.update(kwargs)

        def epoch_callback(epoch_loss):
            self.eval.register_epoch_loss(epoch_loss)
            if self.logger is not None:
                self.logger.log('current_epoch', self.train_loop.epochs)
        
        def batch_callback(batch_loss):
            self.eval.register_batch_loss(batch_loss)
            if self.logger is not None:
                self.logger.log('current_batch', self.train_loop.total_steps)

        self.train_loop.epoch(
            dataloader = self.data,
            model = self.model,
            pdmp = self.pdmp,
            optimizer = self.optimizer,
            learning_schedule = self.learning_schedule,
            ema_models=self.ema_models,
            batch_callback = batch_callback,
            epoch_callback = epoch_callback,
            **tmp_kwargs)
    
    def evaluate(self, **kwargs):
        self.model.eval()
        with torch.inference_mode():
            self.eval.evaluate_model(**kwargs)
    
    def get_ema_model(self, mu):
        for ema in self.ema_models:
            if ema.mu == mu:
                import copy
                new_ema_model = copy.deepcopy(self.model)
                ema.ema(new_ema_model)
                return new_ema_model
        raise ValueError('No EMA model with mu = {}'.format(mu))

    # do this on another eval object
    def evaluate_emas(self,
                    ddim = None,
                    eval_eta = None,
                    reduce_timesteps = None,
                    data_to_generate = None):
        # get a copy of the model
        #tmp_model = copy.deepcopy(self.model)
        # assign it to eval 
        #self.eval.model = tmp_model
        if self.ema_models is None:
            return
        for ema, (eval_ema, mu) in zip(self.ema_models, self.ema_evals):
            ema.ema(eval_ema.model) # now model has ema parameters
            eval_ema.model.eval()
            with torch.inference_mode():
                def callback_on_logging(logger, key, value):
                    if not (key in ['losses', 'losses_batch']):
                        logger.log('_'.join(('ema', str(mu), str(key))), value)
                eval_ema.evaluate_model(ddim = ddim if ddim is not None else self.eval.kwargs['ddim'],
                                    eval_eta = eval_eta if eval_eta is not None else self.eval.kwargs['eval_eta'],
                                    reduce_timesteps = reduce_timesteps if reduce_timesteps is not None else self.eval.kwargs['reduce_timesteps'],
                                    data_to_generate = data_to_generate if data_to_generate is not None else self.eval.kwargs['data_to_generate'],
                                    callback_on_logging = callback_on_logging) 
                # all other parameters are left unchanged
            

    def training_epochs(self):
        return self.train_loop.epochs
    
    def training_batches(self):
        return self.train_loop.total_steps
    
    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=torch.device(self.pdmp.device))
        self.model.load_state_dict(checkpoint['model_parameters'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.learning_schedule is not None:
            self.learning_schedule.load_state_dict(checkpoint['learning_schedule'])
        if self.ema_models is not None:
            for ema, ema_state in zip(self.ema_models, checkpoint['ema_models']):
                ema.load_state_dict(ema_state)
        self.train_loop.total_steps = checkpoint['steps']
        self.train_loop.epochs = checkpoint['epoch']
            
    def save(self, filepath):
        checkpoint = {
            'model_parameters': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'ema_models': [ema.state_dict() for ema in self.ema_models] \
                        if self.ema_models is not None else None,
            'epoch': self.train_loop.epochs,
            'steps': self.train_loop.total_steps,
            'learning_schedule': None if self.learning_schedule is None else self.learning_schedule.state_dict(),
        }
        torch.save(checkpoint, filepath)
    
    def save_eval_metrics(self, eval_path):
        eval_save = {'eval': self.eval.evals}
        if self.ema_evals is not None:
            eval_save.update({'ema_evals': [(ema_eval.evals, mu) for ema_eval, mu in self.ema_evals]})
        torch.save(eval_save, eval_path)
    
    def load_eval_metrics(self, eval_path):
        eval_save = torch.load(eval_path)
        if 'eval' in eval_save:
            self.eval.evals = eval_save['eval']
        else:
            self.eval.evals = eval_save
        self.eval.log_existing_eval_values(folder='eval')
        if not 'ema_evals' in eval_save:
            return
        assert self.ema_evals is not None
        #assert len(self.ema_evals) == len(eval_save['ema_evals']) # no thats fine, just have the same mu at least
        saved_ema_evals = [ema_eval_save for ema_eval_save, mu_save in eval_save['ema_evals']]
        saved_mus = [mu_save for ema_eval_save, mu_save in eval_save['ema_evals']]
        for ema_eval, mu in self.ema_evals:
            if mu not in saved_mus:
                continue
            idx = saved_mus.index(mu)
            ema_eval.evals = saved_ema_evals[idx]
            ema_eval.log_existing_eval_values(folder='eval_ema_{}'.format(mu))
        '''for (ema_eval, mu), (ema_eval_save, mu_save) in zip(self.ema_evals, eval_save['ema_evals']):
            assert mu == mu_save
            ema_eval.evals = ema_eval_save
            ema_eval.log_existing_eval_values(folder='eval_ema_{}'.format(mu))'''

    def __getitem__(self, key):
        import copy
        return copy.deepcopy(self.eval.evals[key])

    # also returns evals
    def display_evals(self,
                      key, 
                      rang = None,
                      xlim = None,
                      ylim = None,
                      log_scale = False):
        import copy
        metric = copy.deepcopy(self.eval.evals[key])
        if rang is not None:
            metric = metric[rang[0]:rang[1]]
        plt.plot(np.arange(len(metric)), metric)
        if xlim is not None:
            plt.xlim(xlim) 
        if ylim is not None:
            plt.ylim(ylim)
        if log_scale:
            plt.yscale('log')
        plt.show()





    #def reset_training(self, new_parameters = None):
    #    self.eval.reset()
    #    self.total_steps = 0
    #    self.epochs = 0
    #    self.learning_schedule.reset()
    #    if self.logger is not None:
    #        self.logger.stop()
    #    if new_parameters is not None:
    #        self.logger.initialize(new_parameters)