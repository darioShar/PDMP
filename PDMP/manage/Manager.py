import numpy as np
import torch
import matplotlib.pyplot as plt
import PDMP.compute.TrainLoop as TrainLoop
import PDMP.models.Ema as Ema
import copy

#Wrapper around training and evaluation functions


class Manager:
        
    def __init__(self, 
                 model,
                 model_vae,
                 data,
                 noising_process, 
                 optimizer,
                 optimizer_vae,
                 learning_schedule,
                 learning_schedule_vae,
                 eval,
                 logger = None,
                 ema_rates = None,
                 **kwargs):
        self.train_loop = TrainLoop.TrainLoop()
        self.model = model
        self.model_vae = model_vae
        self.data = data
        self.noising_process = noising_process
        self.optimizer = optimizer
        self.optimizer_vae = optimizer_vae
        self.learning_schedule = learning_schedule
        self.learning_schedule_vae = learning_schedule_vae
        self.eval = eval
        if ema_rates is None:
            self.ema_objects = None
        else:
            #self.ema_models = [Ema.EMAHelper(model, mu = mu) for mu in ema_rates]
            logger = eval.logger
            # need to set logger to None for the deepcopy
            eval.logger = None
            self.ema_objects = [{
                'model': Ema.EMAHelper(model, mu = mu),
                'eval': copy.deepcopy(eval),
            } for mu in ema_rates]
            #self.ema_evals = [(copy.deepcopy(eval), mu) for mu in ema_rates]
            eval.logger = logger 
            for ema_object in self.ema_objects:
                ema_object['eval'].logger = logger
        
        self.kwargs = kwargs
        self.logger = logger
     
    
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
            model_vae=self.model_vae,
            noising_process = self.noising_process,
            optimizer = self.optimizer,
            optimizer_vae = self.optimizer_vae,
            learning_schedule = self.learning_schedule,
            learning_schedule_vae = self.learning_schedule_vae,
            ema_models=[e['model'] for e in self.ema_objects] if self.ema_objects is not None else None,
            batch_callback = batch_callback,
            epoch_callback = epoch_callback,
            **tmp_kwargs)


    def get_ema_model(self, mu):
        for ema_obj in self.ema_objects:
            if ema_obj['model'].mu == mu:
                return ema_obj['model'].get_ema_model()
                #import copy
                #new_ema_model = copy.deepcopy(self.model)
                #ema.ema(new_ema_model)
                #return new_ema_model
        raise ValueError('No EMA model with mu = {}'.format(mu))


    def evaluate(self, evaluate_emas = False, **kwargs):
        def ema_callback_on_logging(logger, key, value):
            if not (key in ['losses', 'losses_batch']):
                logger.log('_'.join(('ema', str(ema_obj['model'].mu), str(key))), value)
        
        if not evaluate_emas:
            self.model.eval()
            with torch.inference_mode():
                self.eval.evaluate_model(self.model, self.model_vae, **kwargs)
        elif self.ema_objects is not None:
            for ema_obj in self.ema_objects:
                model = ema_obj['model'].get_ema_model()
                model.eval()
                with torch.inference_mode():
                    ema_obj['eval'].evaluate_model(model, self.model_vae, callback_on_logging = ema_callback_on_logging, **kwargs)


    def training_epochs(self):
        return self.train_loop.epochs
    
    def training_batches(self):
        return self.train_loop.total_steps
    
    def _safe_load_state_dict(self, dest, src):
        if dest is not None:
            dest.load_state_dict(src)

    def _safe_save_state_dict(self, src):
        return src.state_dict() if src is not None else None
    
    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=torch.device(self.noising_process.device))
        self.model.load_state_dict(checkpoint['model_parameters'])
        self._safe_load_state_dict(self.model_vae, checkpoint['model_vae_parameters'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self._safe_load_state_dict(self.optimizer_vae, checkpoint['optimizer_vae'])
        self._safe_load_state_dict(self.learning_schedule, checkpoint['learning_schedule'])
        self._safe_load_state_dict(self.learning_schedule_vae, checkpoint['learning_schedule_vae'])
        if self.ema_objects is not None:
            for ema_obj, ema_state in zip(self.ema_objects, checkpoint['ema_models']):
                ema_obj['model'].load_state_dict(ema_state)
        self.train_loop.total_steps = checkpoint['steps']
        self.train_loop.epochs = checkpoint['epoch']
            
    def save(self, filepath):
        checkpoint = {
            'model_parameters': self.model.state_dict(),
            'model_vae_parameters': self._safe_save_state_dict(self.model_vae),
            'optimizer': self.optimizer.state_dict(),
            'optimizer_vae': self._safe_save_state_dict(self.optimizer_vae),
            'ema_models': [ema_obj['model'].state_dict() for ema_obj in self.ema_objects] \
                        if self.ema_objects is not None else None,
            'epoch': self.train_loop.epochs,
            'steps': self.train_loop.total_steps,
            'learning_schedule': self._safe_save_state_dict(self.learning_schedule),
            'learning_schedule_vae': self._safe_save_state_dict(self.learning_schedule_vae),
        }
        torch.save(checkpoint, filepath)
    
    def save_eval_metrics(self, eval_path):
        eval_save = {'eval': self.eval.evals}
        if self.ema_objects is not None:
            eval_save.update({'ema_evals': [(ema_obj['eval'].evals, ema_obj['model'].mu) for ema_obj in self.ema_objects]})
        torch.save(eval_save, eval_path)
    
    def load_eval_metrics(self, eval_path):
        eval_save = torch.load(eval_path)
        assert 'eval' in eval_save, 'no eval subdict in eval file'
        # load eval metrics
        self.eval.evals = eval_save['eval']
        self.eval.log_existing_eval_values(folder='eval')

        # load ema eval metrics
        if not 'ema_evals' in eval_save:
            return
        assert self.ema_objects is not None
        # saved ema evaluation, in order
        saved_ema_evals = [ema_eval_save for ema_eval_save, mu_save in eval_save['ema_evals']]
        # saved ema mu , in order
        saved_mus = [mu_save for ema_eval_save, mu_save in eval_save['ema_evals']]
        
        for ema_obj in self.ema_objects:
            # if mu has not been run previously, no loading
            if ema_obj['model'].mu not in saved_mus:
                continue
            # find index of our mu of interest
            idx = saved_mus.index(ema_obj['model'].mu)
            # load the saved evaluation
            ema_obj['eval'].evals = saved_ema_evals[idx]
            # log the saved evaluation
            ema_obj['eval'].log_existing_eval_values(folder='eval_ema_{}'.format(ema_obj['model'].mu))

        '''saved_ema_evals = [ema_eval_save for ema_eval_save, mu_save in eval_save['ema_evals']]
        saved_mus = [mu_save for ema_eval_save, mu_save in eval_save['ema_evals']]
        for ema_eval, mu in self.ema_evals:
            if mu not in saved_mus:
                continue
            idx = saved_mus.index(mu)
            ema_eval.evals = saved_ema_evals[idx]
            ema_eval.log_existing_eval_values(folder='eval_ema_{}'.format(mu)) '''
        '''for (ema_eval, mu), (ema_eval_save, mu_save) in zip(self.ema_evals, eval_save['ema_evals']):
            assert mu == mu_save
            ema_eval.evals = ema_eval_save
            ema_eval.log_existing_eval_values(folder='eval_ema_{}'.format(mu))'''

    def __getitem__(self, key):
        import copy
        return copy.deepcopy(self.eval.evals[key])
    
    def __setitem(self, key, value):
        self.eval.evals[key] = value
    
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