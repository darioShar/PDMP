import numpy as np
import torch
import matplotlib.pyplot as plt
import DDLM.compute.TrainLoop as TrainLoop
import DDLM.models.Ema as Ema
import copy
#Wrapper around training functions


class Manager:
        
    def __init__(self, 
                 model, 
                 data,
                 diffusion, 
                 optimizer,
                 learning_schedule,
                 eval,
                 ema_rates,
                 logger = None,
                 **kwargs):
        self.train_loop = TrainLoop.TrainLoop()
        self.model = model
        self.data = data
        self.diffusion = diffusion
        self.optimizer = optimizer
        self.learning_schedule = learning_schedule
        if ema_rates is None:
            self.ema_models = None
        else:
            self.ema_models = [Ema.EMAHelper(model, mu = mu) for mu in ema_rates]
            logger = eval.logger
            # need to set logger to None
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
        self.train_loop.epoch(
            dataloader = self.data,
            model = self.model,
            diffusion = self.diffusion,
            optimizer = self.optimizer,
            learning_schedule = self.learning_schedule,
            ema_models=self.ema_models,
            batch_callback = self.eval.register_batch_loss,
            epoch_callback = self.eval.register_epoch_loss,
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
                eval_ema.evaluate_model(ddim = ddim if ddim is not None else self.eval.kwargs['ddim'],
                                    eval_eta = eval_eta if eval_eta is not None else self.eval.kwargs['eval_eta'],
                                    reduce_timesteps = reduce_timesteps if reduce_timesteps is not None else self.eval.kwargs['reduce_timesteps'],
                                    data_to_generate = data_to_generate if data_to_generate is not None else self.eval.kwargs['data_to_generate']) 
                # all other parameters are left unchanged
                # log if not None
            if eval_ema.logger is not None:
                for k, v in eval_ema.evals.items():
                    if not (k in ['losses', 'losses_batch']):
                        eval_ema.logger.log('_'.join(('ema', str(mu), str(k))), v[-1])
            
        # restore initial model 
        #self.eval.model = self.model







    def training_epochs(self):
        return self.train_loop.epochs
    
    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=torch.device(self.diffusion.device))
        self.model.load_state_dict(checkpoint['model_parameters'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
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
            'learning_schedule': self.learning_schedule.state_dict(),
        }
        torch.save(checkpoint, filepath)
    
    def save_eval_metrics(self, eval_path):
        self.eval.save(eval_path)
    
    def load_eval_metrics(self, eval_path):
        self.eval.load(eval_path)
        if self.logger is not None:
            new_values = {'eval': self.eval.evals}
            self.logger.set_values(new_values)

    def display_evals(self,
                      key, 
                      rang = None,
                      xlim = None,
                      ylim = None):
        import copy
        metric = copy.deepcopy(self.eval.evals[key])
        if rang is not None:
            metric = metric[rang[0]:rang[1]]
        plt.plot(np.arange(len(metric)), metric)
        if xlim is not None:
            plt.xlim(xlim) 
        if ylim is not None:
            plt.ylim(ylim)
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