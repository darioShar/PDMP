import numpy as np
import torch
import PDMP.manage.Generate as Gen
from .fid_score import fid_score, prdc
import torchvision.utils as tvu
from .wasserstein import compute_wasserstein_distance
from .prd_legacy import compute_precision_recall_curve, compute_f_beta
import os 
from pathlib import Path
from PDMP.datasets import inverse_affine_transform

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
                 noising_process,
                 dataloader,
                 dataset_files,
                 verbose = True,
                 logger = None,
                 dataset = 'undefined_dataset', # for saving images
                 hash_params = None, # for saving images
                 is_image = False,
                 remove_existing_eval_files = False,
                 **kwargs):
        
        # number of evaluation files required for fid and prdc calculation
        # these original files will be downloaded from the image dataset and stored locally.
        # see manage_directories for more details
        self.evaluation_files = 10000

        self.model = model 
        self.noising_process = noising_process
        self.dataloader = dataloader
        self.dataset_files = dataset_files
        self.verbose = verbose
        self.logger = logger
        self.dataset = dataset
        self.hash_params = hash_params
        self.is_image = is_image
        self.remove_existing_eval_files = remove_existing_eval_files
        self.kwargs = kwargs
        self.reset()
        self.gen_model = Gen.GenerationManager(model = self.model,
                                               noising_process = self.noising_process,
                                              dataloader=self.dataloader,
                                              is_image = self.is_image)
        # create directory for saving images
        folder_path = os.path.join('eval_files', self.dataset)
        self.generated_data_path = os.path.join(folder_path, 'generated_data', self.hash_params)
        if not self.is_image:
            # then we have various versions of the same dataset
            self.original_data_path = os.path.join(folder_path, 'original_data', self.hash_params)
        else:
            self.original_data_path = os.path.join(folder_path, 'original_data')
        
        self.manage_data_directories(self.remove_existing_eval_files)

        #Path(self.generated_data_path).mkdir(parents=True, exist_ok=True)
        #Path(self.original_data_path).mkdir(parents=True, exist_ok=True)

    def manage_data_directories(self, remove_existing_eval_files):

        def remove_file_from_directory(dir):
            # remove the directory
            if not dir.is_dir():
                raise ValueError(f'{dir} is not a directory')
            print('removing files in directory', dir)
            for file in dir.iterdir():
                file.unlink()

        def save_images(path):
                print('storing dataset in', path)
                # now saving the original data
                assert self.dataset.lower() in ['mnist', 'cifar10', 'celeba'], 'only mnist, cifar10, celeba datasets are supported for the moment. \
                    For the moment we are loading {} data points. We may need more for the other datasets, \
                        and anyway we should implement somehting more systematic'.format(self.evaluation_files)
                #data = self.gen_model.load_original_data(self.evaluation_files) # load all the data. Number of datapoints specific to mnist and cifar10
                data_to_store = 10000
                print('saving {} original images from pool of {} datapoints'.format(data_to_store, len(self.dataset_files)))
                for i in range(data_to_store):
                    if (i%500) == 0:
                        print(i, end=' ')
                    tvu.save_image(inverse_affine_transform(self.dataset_files[i][0]), os.path.join(path, f"{i}.png"))
        
        path = Path(self.generated_data_path)
        if path.exists():
            if remove_existing_eval_files:
                remove_file_from_directory(path)
        else:
            path.mkdir(parents=True, exist_ok=True)

        path = Path(self.original_data_path)
        if self.is_image:
            if path.exists():
                print('found', path)
                assert path.is_dir(), (f'{path} is not a directory')
                # check that there are the right number of image files, else remove and regenerate
                if len(list(path.iterdir())) != self.evaluation_files:
                    remove_file_from_directory(path)
                    save_images(path)
            else:
                path.mkdir(parents=True, exist_ok=True)
                save_images(path)
        else:
            if path.exists():
                remove_file_from_directory(path)
            else:
                path.mkdir(parents=True, exist_ok=True)

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
                        data_to_generate,
                       fig_lim = 3,
                       callback_on_logging = None,
                       **kwargs):
        
        eval_results = {}
        #wasserstein. Do not compute if image
        if not self.is_image:
            
            data_to_generate #min(data_to_generate, 128)

            # generate data to evaluate
            self.gen_model.generate(data_to_generate, 
                            print_progression= True,
                            **kwargs)
            # prepare data. REMOVE CHANNEL FOR THE MOMENT REMOVE CHANNEL FOR THE MOMENT REMOVE CHANNEL FOR THE MOMENT REMOVE CHANNEL FOR THE MOMENT REMOVE CHANNEL FOR THE MOMENT
            gen_samples = self.gen_model.samples
            
            data = self.gen_model.load_original_data(data_to_generate)

            print('wasserstein')
            if self.is_image:
                eval_results['wass'] = compute_wasserstein_distance(data, 
                                                    gen_samples,
                                                    manual_compute=True)
            else:
                eval_results['wass'] = compute_wasserstein_distance(data, 
                                                    gen_samples, 
                                                    bins = 250 if data_to_generate >=512 else 'auto')
            print('wasserstein:', eval_results['wass'])

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
                'coverage': 0.,
                'fid': 0.
            }
        else:
            # save data in folder
            import os
            # need at least 2048 samples for fid score, otherwise imaginary component
            # its ok I changed the linalg sqrt matrix function
            if self.remove_existing_eval_files:
                # generate more data if necessary
                #data_to_gen_wass : really is the batch size now. We use some approximation.
                data_batch_size = len(self.dataset_files) // len(self.dataloader) #1024

                gen_samples = torch.tensor([])
                fid_data_to_generate = max(data_batch_size, data_to_generate) # 2048
                remaining = data_to_generate
                print('generating {} images for fid computation'.format(remaining))
                total_generated_data = 0
                while remaining > 0:
                    print(remaining, end = ' ')
                    self.gen_model.generate(min(data_batch_size, remaining),
                                            print_progression= True,
                                            **kwargs)
                    # save data to file. We do that rather than concatenating to save on memory, 
                    # but really it is because I want to inspect the images while they are generated
                    print('saving {} generated images'.format(self.gen_model.samples.shape[0]))
                    for i in range(self.gen_model.samples.shape[0]):
                        tvu.save_image(self.gen_model.samples[i], os.path.join(self.generated_data_path, f"{i+total_generated_data}.png"))
                    total_generated_data += self.gen_model.samples.shape[0]
                    #gen_samples = torch.cat((gen_samples, self.gen_model.samples), dim = 0)
                    remaining = remaining - self.gen_model.samples.shape[0]
                    
                assert fid_data_to_generate == total_generated_data
                print('saved generated data in {}.'.format(self.generated_data_path))
            
            # fid score
            print('fid')
            eval_results['fid'] = fid_score(self.original_data_path, 
                                            self.generated_data_path, 
                                            128, # batch size
                                            self.pdmp.device, 
                                            num_workers= 4 if self.is_image else 0)
            print('prdc')
            # precision, recall density, coverage
            prdc_value = prdc(self.original_data_path, 
                            self.generated_data_path, 
                            128, # batch size 
                            self.pdmp.device, 
                            num_workers= 4 if self.is_image else 0,
                            max_num_files=total_generated_data)
        
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
        if not self.is_image:
            fig = self.gen_model.get_plot(xlim = (-fig_lim, fig_lim), ylim = (-fig_lim, fig_lim))
        else:
            if len(self.gen_model.samples) != 0:
                fig = self.gen_model.get_image() # todo: load an image from the folder
            else:
                fig = None
        
        # for the moment, do not save the figures. Compatibility issues between different versions of matplotlib,
        # and thus for different systems. 
        eval_results['fig'] = None # fig

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
