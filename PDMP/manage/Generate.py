import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from PDMP.datasets import inverse_affine_transform
#from IPython.display import HTML


SAVE_ANIMATION_PATH = './animation'

class GenerationManager:
    
    # same device as pdmp
    def __init__(self, 
                 noising_process,
                 dataloader,
                 is_image,
                 ):
        self.noising_process = noising_process
        self.original_data = dataloader
        self.is_image = is_image
        self.samples = []
        self.history = []

    def generate(self,
                 model,
                 nsamples,
                 time_spacing = None,
                 use_samples = None,
                 get_sample_history = False,
                 print_progression=True,
                 **kwargs
                 ):
        
        _, (data, y) = next(enumerate(self.original_data))
        size = list(data.size())
        size[0] = nsamples 
        x = self.noising_process.reverse_sampling(shape=size,
                                                  model = model,
                                                  time_spacing = time_spacing,
                                                  initial_data = use_samples,
                                                  print_progression = print_progression,
                                                  get_sample_history = get_sample_history,
                                                  **kwargs)
        # store samples and possibly history on cpu
        assert data.shape[-1] == 2, 'only 2d data supported for the moment.'
        if get_sample_history:
            hist = x
            samples = hist[-1, ..., :2]
            self.history = [h[..., :2].cpu() for h in hist]
            if self.is_image:
                self.history = torch.stack([inverse_affine_transform(h) for h in hist]) # apply inverse transform                
        else:
            samples = x[..., :2] # select positions in case of pdmp
            
        # as done in LIM, clamp to -1, 1
        if self.is_image:
            samples = samples.clamp(-1., 1.)
        else:
            samples = samples.clamp(-6., 6.)
        
        self.samples = samples.cpu()
        if self.is_image:
            self.samples = inverse_affine_transform(self.samples)



    def get_image(self, 
                  idx = -1, 
                  black_and_white=False, # in the case of single channel data
                  ):
        return self._get_image_from(self.samples, 
                                    idx=idx, 
                                    black_and_white=black_and_white)
    
    def _get_image_from(self, 
                        samples,
                        idx = -1,
                        black_and_white = False,
                       animated = False):
        img = samples[idx]
        img = torch.stack([img[i]for i in range(img.shape[0])], dim = -1)
        # potentially repeat last dimension for signle channel data to be black and white
        if black_and_white and img.shape[-1] == 1:
            img.repeat(1, 1, 3)
        fig = plt.figure()
        plt.imshow(img, animated = animated)
        plt.close(fig)
        return fig
    
    def get_plot(self, 
                 plot_original_data = True, 
                 limit_nb_orig_data = 2000,
                xlim = None, ylim = None,
                title= None):
        
        original_data = self.load_original_data(limit_nb_orig_data)
        original_data = original_data.squeeze(1) # remove channel since simple 2d data
        gen_data = self.samples
        gen_data = gen_data.squeeze(1) # remove channel
        fig = plt.figure()
        if plot_original_data:
            self._plot_data(original_data, limit_nb_orig_data)
        self._plot_data(gen_data[:limit_nb_orig_data], marker='+')
        if xlim is not None:
            plt.xlim(xlim) 
        if ylim is not None:
            plt.ylim(ylim)
        if plot_original_data:
            plt.legend(["Original data", "Generated data"])
        else:
            plt.legend('Generated data')
        if title is not None:
            plt.title(title)
        plt.close(fig)
        return fig
    
    def _plot_data(self, data, limit=None, marker=None, animated=False):
        if limit is not None:
            limit = data.shape[0]
        if data.shape[1] == 1:
            fig = plt.scatter(data[:limit, 0], torch.zeros(data.shape[0]), 
                        marker=marker, alpha = .5, animated=animated)
        else:
            fig = plt.scatter(data[:limit, 0], data[:limit, 1], 
                        marker=marker, alpha=0.5, animated = animated)
        return fig
    
    def load_original_data(self, nsamples):
        data_size = 0
        total_data = torch.tensor([])
        while data_size < nsamples:
            _, (data,y) = next(enumerate(self.original_data))
            if self.is_image:
                data = inverse_affine_transform(data)
            total_data = torch.concat([total_data, data])
            data_size += data.size()[0]
        return total_data[:nsamples]
    
    def animation(self,
                  generated_data_name = "undefined_distribution",
                plot_original_data = True, 
                filepath = SAVE_ANIMATION_PATH, 
                xlim = (-2.5, 2.5), ylim = (-2.5, 2.5),
                limit_nb_orig_data = 1000):
        if plot_original_data:
            original_data = self.load_original_data(limit_nb_orig_data)
            original_data = original_data.squeeze(1)
        
        def draw_frame(i):
            plt.clf()
            Xvis = self.history[i].cpu().squeeze(1)
            if self.is_image:
                # only plot successive images, no original data
                fig = self._get_image_from(Xvis, animated=True)
            else:
                if plot_original_data:
                    fig = self._plot_data(original_data, limit_nb_orig_data, 
                                    marker = '.', animated=True)
                
                    self._plot_data(Xvis, 
                                marker = '.', animated=True)
                else:
                    fig = self._plot_data(Xvis, 
                                marker = '.', animated=True)
                plt.xlim(xlim)
                plt.ylim(ylim)
            return fig,


        fig = plt.figure()
        # 30 fps, 33ms
        anim = animation.FuncAnimation(fig, draw_frame, frames=len(self.history), interval=33, blit=True)
        path = os.path.join(filepath, 
                            generated_data_name+ '.mp4')
        anim.save(path, fps=30)
        return path
    
    
    
    
    
    