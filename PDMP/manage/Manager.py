import numpy as np
import torch
import matplotlib.pyplot as plt
import PDMP.compute.TrainLoop_old as TrainLoop_old
import PDMP.models.Ema as Ema
import copy
from PDMP.datasets import is_image_dataset
import torch.nn as nn
from tqdm import tqdm
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
                reset_vae = None,
                 p = None,
                 **kwargs):
        self.epochs = 0
        self.total_steps = 0
        self.model = model
        self.model_vae = model_vae
        self.data = data
        self.noising_process = noising_process
        self.optimizer = optimizer
        self.optimizer_vae = optimizer_vae
        self.learning_schedule = learning_schedule
        self.learning_schedule_vae = learning_schedule_vae
        self.eval = eval
        self.reset_vae = reset_vae
        self.p = p
        if ema_rates is None:
            self.ema_objects = None
        else:
            #self.ema_models = [Ema.EMAHelper(model, mu = mu) for mu in ema_rates]
            logger = eval.logger
            # need to set logger to None for the deepcopy
            eval.logger = None
            self.ema_objects = [{
                'model': Ema.EMAHelper(self.model, mu = mu),
                'model_vae': Ema.EMAHelper(self.model_vae, mu = mu) if self.model_vae is not None else None,
                'eval': copy.deepcopy(eval),
            } for mu in ema_rates]
            #self.ema_evals = [(copy.deepcopy(eval), mu) for mu in ema_rates]
            eval.logger = logger 
            for ema_object in self.ema_objects:
                ema_object['eval'].logger = logger
        
        self.kwargs = kwargs
        self.logger = logger
     
    
    def train(self, total_epoch, **kwargs):
        tmp_kwargs = self.kwargs
        tmp_kwargs.update(kwargs)

        def epoch_callback(epoch_loss):
            self.eval.register_epoch_loss(epoch_loss)
            if self.logger is not None:
                self.logger.log('current_epoch', self.epochs)
        
        def batch_callback(batch_loss):
                        # batch_loss is nan, reintialize vae
            if np.isnan(batch_loss): 
                if self.model_vae is not None:
                    print('reinitializing model_vae, which is likely causing nan in batch loss')
                    m, o, l = self.reset_vae(self.p)
                    self.model_vae = m
                    self.optimizer_vae = o
                    self.learning_schedule_vae = l
                else:
                    raise Exception('loss is nan')
            
            self.eval.register_batch_loss(batch_loss)
            if self.logger is not None:
                self.logger.log('current_batch', self.total_steps)
        
        self._train_epochs(total_epoch, 
                           epoch_callback=epoch_callback, 
                           batch_callback=batch_callback,
                           **tmp_kwargs)
        #self.train_loop.epoch(
        #    dataloader = self.data,
        #    model = self.model,
        #    model_vae=self.model_vae,
        #    noising_process = self.noising_process,
        #    optimizer = self.optimizer,
        #    optimizer_vae = self.optimizer_vae,
        #    learning_schedule = self.learning_schedule,
        #    learning_schedule_vae = self.learning_schedule_vae,
        #    ema_models=[e['model'] for e in self.ema_objects] if self.ema_objects is not None else None,
        #    batch_callback = batch_callback,
        #    epoch_callback = epoch_callback,
        #    is_image=is_image_dataset(self.p['data']['dataset']),
        #    **tmp_kwargs)

    def _train_epochs(
                self,
                total_epochs,
                eval_freq = None,
                checkpoint_freq= None,
                checkpoint_callback=None,
                no_ema_eval = False,
                grad_clip = None,
                batch_callback = None,
                epoch_callback = None,
                max_batch_per_epoch = None,
                progress = False,
                train_type=None,
                train_alternate=False,
                **kwargs):
        

        self.model.train()
        if self.model_vae is not None:
            self.model_vae.train()
        
        ema_models=[e['model'] for e in self.ema_objects] if self.ema_objects is not None else None
        ema_models_vae=[e['model_vae'] for e in self.ema_objects] if self.ema_objects is not None else None
        print('training model to epoch {} from epoch {}'.format(total_epochs, self.epochs), '...')
        is_image = is_image_dataset(self.p['data']['dataset'])
        freeze_vae = False

        #train_procedure = [['VAE', 'NORMAL']]*10 + [['VAE', 'NORMAL_WITH_VAE']]
        #['VAE']*5 + ['NORMAL']*2 + ['NORMAL_WITH_VAE']*1 + ['NORMAL']*2

        while self.epochs < total_epochs:
            epoch_loss = steps = 0
            for i, (Xbatch, y) in enumerate(tqdm(self.data) if progress else self.data):
                if max_batch_per_epoch is not None:
                    if i >= max_batch_per_epoch:
                        break
                
                if is_image:
                    # for image datasets.
                    Xbatch += 0.5*torch.rand_like(Xbatch) / (256)
                if (self.model_vae is not None):
                    if not train_alternate:
                        train_type = ['VAE']
                    else:
                        freeze_vae = True
                        self.model_vae.eval()
                        #if (self.total_steps % 2) == 0:
                        #    train_type = ['NORMAL'] 
                        #else:
                        train_type = ['NORMAL_WITH_VAE'] 
                    if not is_image:
                        train_type = ['VAE', 'NORMAL']
                else:
                    train_type = None
                
                #print('train_type:', train_type)
                
                if train_type is not None:
                    loss = self.noising_process.training_losses(self.model, 
                                                       Xbatch, 
                                                       train_type=train_type,#train_procedure[self.total_steps % len(train_procedure)], 
                                                       model_vae=self.model_vae, 
                                                       **kwargs)
                else:
                    loss = self.noising_process.training_losses(self.model, Xbatch, **kwargs)
                

                #loss = pdmp.training_losses(model, Xbatch, Vbatch, time_horizons)
                #loss = loss.mean()
                #print('loss computed')
                # and finally gradient descent
                self.optimizer.zero_grad()
                if (self.optimizer_vae is not None) and (not freeze_vae):
                    self.optimizer_vae.zero_grad()
                loss.backward()
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                self.optimizer.step()
                if (self.optimizer_vae is not None) and (not freeze_vae):
                    self.optimizer_vae.step()
                if (self.learning_schedule is not None):
                    self.learning_schedule.step()
                if (self.learning_schedule_vae is not None) and (not freeze_vae):
                    self.learning_schedule_vae.step()
                # update ema models
                if self.ema_objects is not None:
                    for e in self.ema_objects:
                        e['model'].update(self.model)
                        if e['model_vae'] is not None and not freeze_vae:
                            e['model_vae'].update(self.model_vae)
                epoch_loss += loss.item()
                steps += 1
                self.total_steps += 1
                if batch_callback is not None:
                    batch_callback(loss.item())
            epoch_loss = epoch_loss / steps
            print('epoch_loss', epoch_loss)
            self.epochs += 1
            if epoch_callback is not None:
                epoch_callback(epoch_loss)

            print('Done training epoch {}/{}'.format(self.epochs, total_epochs))

            # now potentially checkpoint
            if (checkpoint_freq is not None) and (self.epochs % checkpoint_freq) == 0:
                checkpoint_callback(self.epochs)
                #print(self.save(curr_epoch=self.manager.training_epochs()))
            
            # now potentially eval
            if (eval_freq is not None) and  (self.epochs % eval_freq) == 0:
                self.evaluate()
                if not no_ema_eval:
                    self.evaluate(evaluate_emas=True)


    def get_ema_model(self, mu):
        assert False, 'deprecated'
        for ema_obj in self.ema_objects:
            if ema_obj['model'].mu == mu:
                return ema_obj['model'].get_ema_model()
                #import copy
                #new_ema_model = copy.deepcopy(self.model)
                #ema.ema(new_ema_model)
                #return new_ema_model
        raise ValueError('No EMA model with mu = {}'.format(mu))

    def display_plots(self, 
                   ema_mu = None, 
                   title = None,
                   nb_datapoints = 10000,
                   marker = '.',
                   color='blue',
                   plot_original_data=False,
                   xlim = (-0.5, 1.5),
                   ylim = (-0.5, 1.5),
                   alpha = 0.5):
        # loading the right model
        model = None
        if ema_mu is not None:
            if self.ema_objects is not None:
                for ema_obj in self.ema_objects:
                    if ema_obj['model'].mu == ema_mu:
                        model = ema_obj['model'].get_ema_model()
                        model.eval
                    if ema_obj['model_vae'] is not None:
                        model_vae = ema_obj['model_vae'].get_ema_model()
                        model_vae.eval()
                    else:
                        model_vae = None
        else:
            model = self.model
            model.eval()
        
        assert model is not None, 'ema_mu={} has not been found'.format(ema_mu)

        # number of samples to draw
        nsamples = 1 if self.eval.is_image else nb_datapoints
        print('Generating {} datapoints'.format(nsamples))

        # generate images
        gm = self.eval.gen_manager
        with torch.inference_mode():
            gm.generate(self.model, self.model_vae, nsamples, get_sample_history = True, print_progression = True)
        
        # get and display plots
        if self.eval.is_image:
            gm.get_image(black_and_white=True, title=title)
        else:
            gm.get_plot(plot_original_data=plot_original_data,
                                 limit_nb_datapoints = nb_datapoints,
                                 title=title, 
                                 marker = marker, 
                                 color=color, 
                                 xlim=xlim, 
                                 ylim=ylim,
                                 alpha=alpha)
        plt.show(block=False)
        anim = gm.get_animation(plot_original_data=plot_original_data,
                                  limit_nb_datapoints=nb_datapoints,
                                  title=title, 
                                  marker = marker, 
                                  color=color, 
                                  xlim=xlim, 
                                  ylim=ylim,
                                  alpha=alpha,
                                  pdmp = self.noising_process)
        plt.show(block=False)
        return anim
    
    def evaluate(self, evaluate_emas = False, **kwargs):
        def ema_callback_on_logging(logger, key, value):
            if not (key in ['losses', 'losses_batch']):
                logger.log('_'.join(('ema', str(ema_obj['model'].mu), str(key))), value)
        
        if not evaluate_emas:
            print('evaluating model')
            self.model.eval()
            with torch.inference_mode():
                self.eval.evaluate_model(self.model, self.model_vae, **kwargs)
        elif self.ema_objects is not None:
            for ema_obj in self.ema_objects:
                model = ema_obj['model'].get_ema_model()
                print('evaluating ema model with mu={}'.format(ema_obj['model'].mu))
                if ema_obj['model_vae'] is not None:
                    model_vae = ema_obj['model_vae'].get_ema_model()
                    model_vae.eval()
                else:
                    model_vae = None
                model.eval()
                with torch.inference_mode():
                    ema_obj['eval'].evaluate_model(model, model_vae, callback_on_logging = ema_callback_on_logging, **kwargs)

    def training_epochs(self):
        return self.epochs
    
    def training_batches(self):
        return self.total_steps
    
    # provide key rather than src[key] in case we load an acient run e.g that did not have vae implemented
    def _safe_load_state_dict(self, dest, src, key):
        if dest is not None:
            dest.load_state_dict(src[key])

    def _safe_save_state_dict(self, src):
        return src.state_dict() if src is not None else None
    
    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=torch.device(self.noising_process.device))
        self.model.load_state_dict(checkpoint['model_parameters'])
        self._safe_load_state_dict(self.model_vae, checkpoint, 'model_vae_parameters')
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self._safe_load_state_dict(self.optimizer_vae, checkpoint, 'optimizer_vae')
        self._safe_load_state_dict(self.learning_schedule, checkpoint, 'learning_schedule')
        self._safe_load_state_dict(self.learning_schedule_vae, checkpoint, 'learning_schedule_vae')
        if self.ema_objects is not None:
            if ('ema_models_vae' in checkpoint) and checkpoint['ema_models_vae'] is not None:
                for ema_obj, ema_state, ema_vae_state in zip(self.ema_objects, checkpoint['ema_models'], checkpoint['ema_models_vae']):
                    ema_obj['model'].load_state_dict(ema_state)
                    if ema_obj['model_vae'] is not None:
                        ema_obj['model_vae'].load_state_dict(ema_vae_state)
            else:
                for ema_obj, ema_state, ema_vae_state in zip(self.ema_objects, checkpoint['ema_models'], checkpoint['ema_models_vae']):
                    ema_obj['model'].load_state_dict(ema_state)
                    if ema_obj['model_vae'] is not None:
                        raise Exception('VAE is in EMA mode, but there are no EMA VAE checkpoint to load')
        self.total_steps = checkpoint['steps']
        self.epochs = checkpoint['epoch']
            
    def save(self, filepath):
        checkpoint = {
            'model_parameters': self.model.state_dict(),
            'model_vae_parameters': self._safe_save_state_dict(self.model_vae),
            'optimizer': self.optimizer.state_dict(),
            'optimizer_vae': self._safe_save_state_dict(self.optimizer_vae),
            'ema_models': [ema_obj['model'].state_dict() for ema_obj in self.ema_objects] \
                        if self.ema_objects is not None else None,
            'ema_models_vae': [ema_obj['model_vae'].state_dict() if ema_obj['model_vae'] is not None else None 
                                for ema_obj in self.ema_objects] \
                                if self.ema_objects is not None else None,
            'epoch': self.epochs,
            'steps': self.total_steps,
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