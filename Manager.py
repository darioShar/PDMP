import numpy as np
import torch
import matplotlib.pyplot as plt
import Ema
import torch.nn as nn
from tqdm import tqdm
import copy
#Wrapper around training functions


class Manager:
        
    def __init__(self, 
                 model, 
                 dataloader,
                 pdmp, 
                 optimizer,
                 learning_schedule,
                 eval,
                 ema_rates,
                 logger = None,
                 **kwargs):
        
        self.epochs = 0
        self.total_steps = 0

        self.model = model
        self.dataloader = dataloader
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
                self.logger.log('current_epoch', self.epochs)
        
        def batch_callback(batch_loss):
            self.eval.register_batch_loss(batch_loss)
            if self.logger is not None:
                self.logger.log('current_batch', self.total_steps)

        self.train_loop(
            batch_callback = batch_callback,
            epoch_callback = epoch_callback,
            **tmp_kwargs)
    
    def train_loop(self, 
            nepochs = 1,
            grad_clip = None,
            batch_callback = None,
            epoch_callback = None,
            progress_epoch = True,
            epoch_pbar = None,
            max_batch_per_epoch = None,
            ):
        
        self.model.train()
        if progress_epoch:
            progress_epoch = lambda x : tqdm(x)
        else:
            progress_epoch = lambda x : x
        
        for epoch in progress_epoch(range(nepochs)):
            epoch_loss = steps = 0
            for i, (Xbatch, y) in (enumerate(self.dataloader)):
                if max_batch_per_epoch is not None:
                    if i >= max_batch_per_epoch:
                        break
                device = self.pdmp.device
                # generate random speed
                if self.pdmp.sampler == 'ZigZag':
                    Vbatch = torch.tensor([-1., 1.])[torch.randint(0, 2, (2*Xbatch.shape[0],))]
                    Vbatch = Vbatch.reshape(Xbatch.shape[0], 1, 2)
                elif self.pdmp.sampler == 'HMC':
                    Vbatch = torch.randn_like(Xbatch)
                elif self.pdmp.sampler == 'BPS':
                    Vbatch = torch.randn_like(Xbatch)
                

                # generate random time horizons
                time_horizons = self.pdmp.T * (torch.rand(Xbatch.shape[0])**2)

                # must be of the same shape as Xbatch for the pdmp forward process, since it will be applied component wise
                t = time_horizons.clone().detach().unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 2)
                
                # clone our initial data, since it will be modified by forward process
                x = Xbatch.clone()
                v = Vbatch.clone()
                
                # apply the forward process. Everything runs on the cpu.
                self.pdmp.forward(Xbatch, t, Vbatch)
                
                # check that the data has been modified
                #idx = (x == Xbatch)
                #print(idx, x[idx], Xbatch[idx])
                #idx = (v == Vbatch)
                #print(idx, v[idx], Vbatch[idx])
                assert ((x == Xbatch).logical_not()).any() and ((v == Vbatch).logical_not()).any()
                # check that the time horizon has been reached for all data
                assert not (t != 0.).any()
                
                loss = self.pdmp.training_losses(self.model, Xbatch, Vbatch, time_horizons)
                loss = loss.mean()
                
                #print('loss computed')
                # and finally gradient descent
                self.optimizer.zero_grad()
                loss.backward()
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                self.optimizer.step()
                if self.learning_schedule is not None:
                    self.learning_schedule.step()
                # update ema models
                if self.ema_models is not None:
                    for ema in self.ema_models:
                        ema.update(self.model)
                epoch_loss += loss.item()
                steps += 1
                self.total_steps += 1
                if batch_callback is not None:
                    batch_callback(loss.item())
                #print(loss.item())
            if epoch_pbar is not None:
                epoch_pbar.update(1)
            epoch_loss = epoch_loss / steps
            self.epochs += 1
            if epoch_callback is not None:
                epoch_callback(epoch_loss)
    
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
        return self.epochs
    
    def training_batches(self):
        return self.total_steps
    
    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=torch.device(self.pdmp.device))
        self.model.load_state_dict(checkpoint['model_parameters'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.learning_schedule is not None:
            self.learning_schedule.load_state_dict(checkpoint['learning_schedule'])
        if self.ema_models is not None:
            for ema, ema_state in zip(self.ema_models, checkpoint['ema_models']):
                ema.load_state_dict(ema_state)
        self.total_steps = checkpoint['steps']
        self.epochs = checkpoint['epoch']
            
    def save(self, filepath):
        checkpoint = {
            'model_parameters': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'ema_models': [ema.state_dict() for ema in self.ema_models] \
                        if self.ema_models is not None else None,
            'epoch': self.epochs,
            'steps': self.total_steps,
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
        assert len(self.ema_evals) == len(eval_save['ema_evals'])
        for (ema_eval, mu), (ema_eval_save, mu_save) in zip(self.ema_evals, eval_save['ema_evals']):
            assert mu == mu_save
            ema_eval.evals = ema_eval_save
            ema_eval.log_existing_eval_values(folder='eval_ema_{}'.format(mu))

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
        



        #speed_idx_0 = X_V_t[:, :, 2] + 1 //2 # 0 if speed_0 is -1, 1 if speed_0 is 1
        #speed_idx_1 = X_V_t[:, :, 3] + 1 //2 # 0 if speed_1 is -1, 1 if speed_1 is 1
        #loss = 0.5*output[:, :, 0]**2 - (1 / output[:,:,0])
        #loss += 0.5*output[:, :, 1]**2 - (1 / output[:,:,1])
        #loss = loss.mean()
        #normalize_mean = output.mean()
        #loss += (torch.log(normalize_mean))**2
        #loss  = (1 / (1 + output.mean()) - 1 / 2)**2
        #loss += - torch.log(output_inv_1).mean() - torch.log(output_inv_2).mean()
        #if i % 10 != 0:
        #    loss = 0.5*output[:, :, 0]**2 - (output_inv_1[:,:,0])
        #    loss += 0.5*output[:, :, 1]**2 - (output_inv_2[:,:,1])
        #    loss = loss.mean()
        #loss = 0.5*output[:, :, 0]**2 - (output[:,:,2 + 0])
        #loss += 0.5*output[:, :, 1]**2 - (output[:,:,2 + 1])
        #else:
        #    loss = .5*((1 / output[:,:,0]).mean() - 1)**2
        #    loss += .5*((1 / output[:,:,1]).mean() - 1)**2
        #    loss += .5*((output_inv_1[:,:,0]).mean() - 1)**2
        #    loss += .5*((output_inv_2[:,:,1]).mean() - 1)**2
        #    loss += .5*((output[:, :, 0] * output_inv_1[:,:,0] - 1).mean())**2
        #    loss += .5*((output[:, :, 1] * output_inv_2[:,:,1] - 1).mean())**2
        #    loss = loss.mean()
        #loss += (output.mean() - 1)**2
        #loss += (output_inv_1.mean() - 1)**2
        #loss += (output_inv_2.mean() - 1)**2
        #loss += (output[:, :, 0] * output_inv_1[:,:,0] - 1).mean()
        #loss += (output[:, :, 1] * output_inv_2[:,:,1] - 1).mean()
        # and finally gradient descent