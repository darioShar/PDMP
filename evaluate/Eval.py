import numpy as np
import torch
import DDLM.manage.Generate as Gen
from .fid_score import fid_score, prdc
import torchvision.utils as tvu
from .wasserstein import compute_wasserstein_distance
from .prd_legacy import compute_precision_recall_curve, compute_f_beta


class Eval:

    def __init__(self,
                 model, 
                 diffusion,
                 dataloader,
                 verbose = True,
                 logger = None,
                 **kwargs):
        self.model = model 
        self.diffusion = diffusion
        self.dataloader = dataloader
        self.verbose = verbose
        self.logger = logger
        self.kwargs = kwargs
        self.reset()

    def reset(self):
        self.evals = {
            'losses': np.array([], dtype = np.float32),
            'losses_batch': np.array([], dtype = np.float32),
            'wass': [],
            'precision': [],
            'recall': [],
            'density': [],
            'coverage': [],
            'f_1_pr': [],
            'f_1_dc': [],
            'fid': [],
            'fig': []
        }
    
    def save(self, eval_path):
        torch.save(self.evals, eval_path)

    def load(self, eval_path):
        self.evals = torch.load(eval_path)
        # log if logger is not None
        if self.logger is not None:
            new_values = {'eval': self.evals}
            self.logger.set_values(new_values)
    
    def register_batch_loss(self, batch_loss):
        self.evals['losses_batch'] = np.append(self.evals['losses_batch'], batch_loss)
        if self.logger is not None:
            self.logger.log('loss_batch', batch_loss)
    
    def register_epoch_loss(self, epoch_loss):
        self.evals['losses'] = np.append(self.evals['losses'], epoch_loss)
        if self.logger is not None:
            self.logger.log('loss_epoch', epoch_loss)

    # little workaround to enable arbitrary number of kwargs to be specified beforehand
    def evaluate_model(self, **kwargs):
        tmp_kwargs = self.kwargs
        tmp_kwargs.update(kwargs)
        self._evaluate_model(**tmp_kwargs)

    # compute evaluatin metrics
    def _evaluate_model(self,
                       is_image = True, # is the dataset composed of images
                       ddim = False,
                       eval_eta = 1., 
                       reduce_timesteps = 1., # divide by reduce_timesteps
                       data_to_generate = 5000,
                       fig_lim = 3,
                       clip_denoised = False):
        gen_model = Gen.GenerationManager(self.model, 
                                          self.diffusion, 
                                          self.dataloader)
        # generate data to evaluate
        gen_model.generate(data_to_generate, 
                           ddim=ddim, 
                           eta = eval_eta, 
                           reduce_timesteps = reduce_timesteps,
                           print_progression= True,
                           clip_denoised = clip_denoised)
        
        # prepare data. REMOVE CHANNEL FOR THE MOMENT REMOVE CHANNEL FOR THE MOMENT REMOVE CHANNEL FOR THE MOMENT REMOVE CHANNEL FOR THE MOMENT REMOVE CHANNEL FOR THE MOMENT
        data = gen_model.load_original_data(data_to_generate) # this function also removes channel'
        gen_samples = gen_model.samples.squeeze(1) # squeeze initial data

        eval_results = {}
        #wasserstein
        if data_to_generate <= 256:
            eval_results['wass'] = compute_wasserstein_distance(data, 
                                                gen_samples,
                                                manual_compute=True)
        else:
            eval_results['wass'] = compute_wasserstein_distance(data, 
                                                gen_samples, 
                                                bins = 250)
        #print('wasserstein ok')

        if is_image: 
            # save data in folder
            import os
            for i in range(data_to_generate):
                tvu.save_image(
                    gen_samples[i], os.path.join('eval_files/generated_data', f"{i}.png")
                )
                tvu.save_image(
                    data[i], os.path.join('eval_files/original_data', f"{i}.png")
                )
            # fid score
            eval_results['fid'] = fid_score('eval_files/original_data', 
                                            'eval_files/generated_data', 
                                            data_to_generate, 
                                            self.diffusion.device, 
                                            num_workers=0)
            # precision, recall density, coverage
            prdc_value = prdc('eval_files/original_data', 
                            'eval_files/generated_data', 
                            data_to_generate, 
                            self.diffusion.device, 
                            num_workers=0)
        else:
            # scatter plot, simple 2d data.
            # precision/recall
            pr_curve = compute_precision_recall_curve(data, 
                                                    gen_samples, 
                                                    num_clusters=100 if data_to_generate > 2500 else 20)
            #print('prd ok')
            f_beta = compute_f_beta(*pr_curve)
            prdc_value = {
                'precision': f_beta[0],
                'recall': f_beta[1],
                'density': 0.,
                'coverage': 0.
            }
        for k, v in prdc_value.items():
            eval_results[k] = v

        # compute f_1 score
        if prdc_value['precision'] + prdc_value['recall'] > 0:
            eval_results['f_1_pr'] = (2 * prdc_value['precision'] * prdc_value['recall']) / (prdc_value['precision'] + prdc_value['recall'])
        else:
            eval_results['f_1_pr'] = 0.
        if prdc_value['density'] + prdc_value['coverage'] > 0:
            eval_results['f_1_dc'] = (2 * prdc_value['density'] * prdc_value['coverage']) / (prdc_value['density'] + prdc_value['coverage'])
        else:
            eval_results['f_1_dc'] = 0.

        # figure
        if not is_image:
            fig = gen_model.get_plot(xlim = (-fig_lim, fig_lim), ylim = (-fig_lim, fig_lim))
        else:
            fig = gen_model.get_image()
        
        eval_results['fig'] = fig

        # append results to self.evals dictionnary
        for k in eval_results.keys():
            self.evals[k].append(eval_results[k])

        if self.logger is not None:
                for k, v in eval_results.items():
                    self.logger.log(k, v)
        
        # print them if necessary
        if self.verbose:
            # last loss, if computed
            if self.evals['losses_batch'].any():
                print(f"\tlosses_batch = {self.evals['losses_batch'][-1]}")
            # and the valuation metrics
            for k, v in eval_results.items():
                print('\t{} = {}'.format(k, v))
        
        return eval_results
