import numpy as np
import torch
import Generate as Gen
from .fid_score import fid_score, prdc
import torchvision.utils as tvu
from .wasserstein import compute_wasserstein_distance
from .prd_legacy import compute_precision_recall_curve, compute_f_beta
import os 
from pathlib import Path


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
class Eval:

    def __init__(self,
                 model, 
                 pdmp,
                 dataloader,
                 verbose = True,
                 logger = None,
                 dataset = 'undefined_dataset', # for saving images
                 hash_params = '', # for saving images
                 **kwargs):
        self.model = model 
        self.pdmp = pdmp
        self.dataloader = dataloader
        self.verbose = verbose
        self.logger = logger
        self.dataset = dataset
        self.hash_params = hash_params
        self.kwargs = kwargs
        self.reset()
        # create directory for saving images
        folder_path = os.path.join('eval_files', self.dataset)
        self.generated_data_path = os.path.join(folder_path, 'generated_data', self.hash_params)
        self.original_data_path = os.path.join(folder_path, 'original_data', self.hash_params)
        Path(self.generated_data_path).mkdir(parents=True, exist_ok=True)
        Path(self.original_data_path).mkdir(parents=True, exist_ok=True)

    def reset(self,
              keep_losses = False,
              keep_evals = False):
        self.evals = {
            'losses': self.evals['losses'] if keep_losses else np.array([], dtype = np.float32),
            'losses_batch': self.evals['losses_batch'] if keep_losses else np.array([], dtype = np.float32),
            'wass': self.evals['wass'] if keep_evals else [],
            'precision': self.evals['precision'] if keep_evals else [],
            'recall': self.evals['recall'] if keep_evals else [],
            'density': self.evals['density'] if keep_evals else [],
            'coverage': self.evals['coverage'] if keep_evals else [],
            'f_1_pr': self.evals['f_1_pr'] if keep_evals else [],
            'f_1_dc': self.evals['f_1_dc'] if keep_evals else [],
            'fid': self.evals['fid'] if keep_evals else [],
            'fig': self.evals['fig'] if keep_evals else [],
        }
    
    #def save(self, eval_path):
    #    torch.save(self.evals, eval_path)

    def log_existing_eval_values(self, folder='eval'):
        #self.evals = torch.load(eval_path)
        if self.logger is not None:
            new_values = {folder: self.evals}
            self.logger.set_values(new_values)
    
    def register_batch_loss(self, batch_loss):
        self.evals['losses_batch'] = np.append(self.evals['losses_batch'], batch_loss)
        if self.logger is not None:
            self.logger.log('losses_batch', batch_loss)
    
    def register_epoch_loss(self, epoch_loss):
        self.evals['losses'] = np.append(self.evals['losses'], epoch_loss)
        if self.logger is not None:
            self.logger.log('losses', epoch_loss)

    # little workaround to enable arbitrary number of kwargs to be specified beforehand
    def evaluate_model(self, **kwargs):
        tmp_kwargs = self.kwargs
        tmp_kwargs.update(kwargs)
        self.model.eval()
        with torch.inference_mode():
            self._evaluate_model(**tmp_kwargs)

    # compute evaluatin metrics
    def _evaluate_model(self,
                       is_image = False, # is the dataset made of images
                       reduce_timesteps = 1., # divide by reduce_timesteps
                       data_to_generate = 5000,
                       fig_lim = 3,
                       clip_denoised = False,
                       callback_on_logging = None):
        
        gen_model = Gen.GenerationManager(self.model, 
                                          self.pdmp, 
                                          self.dataloader,
                                          is_image = is_image)
        
        # appart from fid and prdc, maximum of 512 datapoint.
        data_to_gen_wass = min(data_to_generate, 512)

        # generate data to evaluate
        gen_model.generate(data_to_gen_wass, 
                           reduce_timesteps = reduce_timesteps,
                           print_progession = True,
                           clip_denoised = clip_denoised)
        # prepare data. REMOVE CHANNEL FOR THE MOMENT REMOVE CHANNEL FOR THE MOMENT REMOVE CHANNEL FOR THE MOMENT REMOVE CHANNEL FOR THE MOMENT REMOVE CHANNEL FOR THE MOMENT

        data = gen_model.load_original_data(data_to_gen_wass)
        gen_samples = gen_model.samples

        eval_results = {}
        #wasserstein
        print('wasserstein')
        if is_image:
            eval_results['wass'] = compute_wasserstein_distance(data, 
                                                gen_samples,
                                                manual_compute=True)
        else:
            eval_results['wass'] = compute_wasserstein_distance(data, 
                                                gen_samples, 
                                                bins = 250 if data_to_gen_wass > 512 else 'auto')
        #print('wasserstein ok')
            
        if not is_image:
            # scatter plot, simple 2d data.
            # precision/recall
            pr_curve = compute_precision_recall_curve(data, 
                                                    gen_samples, 
                                                    num_clusters=100 if data_to_gen_wass > 2500 else 20)
            #print('prd ok')
            f_beta = compute_f_beta(*pr_curve)
            prdc_value = {
                'precision': f_beta[0],
                'recall': f_beta[1],
                'density': 0.,
                'coverage': 0.,
                'fid': 0.
            }
        else:
            # save data in folder
            import os
            # need at least 2048 samples for fid score, otherwise imaginary component
            # its ok I changed the linalg sqrt matrix function

            # generate more data if necessary
            fid_data_to_generate = max(data_to_gen_wass, data_to_generate) # 2048
            data = gen_model.load_original_data(fid_data_to_generate)
            remaining = max(data_to_generate - data_to_gen_wass, 0)
            print('generating {} images for fid computation'.format(remaining))
            while remaining > 0:
                print(remaining, end = ' ')
                gen_model.generate(min(data_to_gen_wass, remaining), 
                                reduce_timesteps = reduce_timesteps,
                                print_progression= False,
                                clip_denoised = clip_denoised)
                gen_samples = torch.cat((gen_samples, gen_model.samples), dim = 0)
                remaining = remaining - data_to_gen_wass
            
            assert fid_data_to_generate == gen_samples.shape[0]
            
            print('saving {} images'.format(gen_samples.shape[0]))
            for i in range(gen_samples.shape[0]):
                tvu.save_image(
                    gen_samples[i], os.path.join(self.generated_data_path, f"{i}.png")
                )
                tvu.save_image(
                    data[i], os.path.join(self.original_data_path, f"{i}.png")
                )
            # fid score
            print('fid')
            eval_results['fid'] = fid_score(self.original_data_path, 
                                            self.generated_data_path, 
                                            256, # batch size
                                            self.pdmp.device, 
                                            num_workers= 6 if is_image else 0)
            print('prdc')
            # precision, recall density, coverage
            prdc_value = prdc(self.original_data_path, 
                            self.generated_data_path, 
                            256, # batch size 
                            self.pdmp.device, 
                            num_workers= 6 if is_image else 0)
        
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
                if callback_on_logging is not None:
                    callback_on_logging(self.logger, k, v)
                else:
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
