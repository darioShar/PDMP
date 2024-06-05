import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os
import copy
from PDMP.datasets import inverse_affine_transform
#from IPython.display import HTML


SAVE_ANIMATION_PATH = './animation'

class GenerationManager:
    
    # same device as pdmp
    def __init__(self, 
                 noising_process,
                 dataloader,
                 is_image,
                 **kwargs
                 ):
        self.noising_process = noising_process
        self.original_data = dataloader
        self.is_image = is_image
        self.kwargs = kwargs
        self.samples = []
        self.history = []

    def generate(self,
                 model,
                 model_vae,
                 nsamples,
                 time_spacing = None,
                 use_samples = None,
                 get_sample_history = False,
                 print_progression=True,
                 **kwargs
                 ):
        tmp_kwargs = copy.deepcopy(self.kwargs)
        tmp_kwargs.update(kwargs)

        _, (data, y) = next(enumerate(self.original_data))
        size = list(data.size())
        size[0] = nsamples
        x = self.noising_process.reverse_sampling(shape=size,
                                                  model = model,
                                                  model_vae = model_vae,
                                                  time_spacing = time_spacing,
                                                  initial_data = use_samples,
                                                  print_progression = print_progression,
                                                  get_sample_history = get_sample_history,
                                                  **tmp_kwargs)
        # store samples and possibly history on cpu
        #assert data.shape[-1] == 2, 'only 2d data supported for the moment.'
        if get_sample_history:
            hist = x
            samples = hist[-1, ..., :data.shape[-1]]
            self.history = [h[..., :data.shape[-1]].cpu() for h in hist]
            if self.is_image:
                self.history = torch.stack([inverse_affine_transform(h) for h in self.history]) # apply inverse transform
        else:
            samples = x[..., :data.shape[-1]] # select positions in case of pdmp
        # as done in LIM, clamp to -1, 1
        if self.is_image:
            samples = samples.clamp(-1., 1.)
        else:
            samples = samples.clamp(-6., 6.)
        
        self.samples = samples.cpu()
        if self.is_image:
            self.samples = inverse_affine_transform(self.samples)

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
        return fig
    
    def get_plot(self, 
                 plot_original_data = True, 
                 limit_nb_datapoints = 10000,
                 title= None,
                 marker = '.',
                 color='blue',
                 xlim = None, 
                 ylim = None,
                 alpha = 0.5
                 ):
        
        original_data = self.load_original_data(limit_nb_datapoints)
        original_data = original_data.squeeze(1) # remove channel since simple 2d data
        gen_data = self.samples
        gen_data = gen_data.squeeze(1) # remove channel
        fig = plt.figure()
        if plot_original_data:
            self._plot_data(original_data[:limit_nb_datapoints], label='Original data', marker=marker, color='orange', alpha=alpha)
        self._plot_data(gen_data[:limit_nb_datapoints], label='Generated data', marker=marker, color=color, alpha=alpha)  # marker='+'
        if xlim is not None:
            plt.xlim(xlim) 
        if ylim is not None:
            plt.ylim(ylim)
        if title is not None:
            plt.title(title)
        plt.legend()
        return fig
    
    def _get_scatter_marker_specific_kwargs(self, marker):
        if marker == '.':
            return {'marker': marker, 'lw': 0, 's': 1}
        return {'marker': marker}

    
    def _plot_data(self, data, marker='.', animated=False, label=None, ax=None, color=None, alpha=0.5):
        assert data.shape[1] == 2, 'only supports plotting 2d data'
        canvas = plt if ax is None else ax 
        fig = canvas.scatter(data[:, 0], data[:, 1], alpha=alpha, animated = animated, label=label, color=color, **self._get_scatter_marker_specific_kwargs(marker))
        return fig

    def get_animation(self, 
                      plot_original_data = False,
                      limit_nb_datapoints = 10000,
                      title = None,
                      marker = '.',
                      color = 'blue',
                      xlim = (-1.1, 1.1), 
                      ylim = (-1.1, 1.1),
                      alpha = 0.5
                      ):
        if plot_original_data:
            original_data = self.load_original_data(limit_nb_datapoints)
            original_data = original_data.squeeze(1)
        
        fig, ax = plt.subplots()  # Create a figure and axes once.
        #if title is not None:
        #    plt.title(title)

        scatter = ax.scatter([], [], alpha=alpha, animated=True, color=color, **self._get_scatter_marker_specific_kwargs(marker))
        scatter_orig = ax.scatter([], [], alpha=alpha, animated=True, color='orange', **self._get_scatter_marker_specific_kwargs(marker))

        def init_frame():
            #ax.clear()  # Clear the current axes.
            ax.set_xlim(xlim)  # Set the limit for x-axis.
            ax.set_ylim(ylim)  # Set the limit for y-axis.
            ax.set_title(title)
            scatter.set_offsets(np.empty((0, 2)))  # Properly shaped empty array
            scatter_orig.set_offsets(np.empty((0, 2)))  # Properly shaped empty array
            return scatter, scatter_orig, 
        
        def draw_frame(i):
            #ax.clear()
            Xvis = self.history[i].cpu().squeeze(1)[:limit_nb_datapoints]
            if self.is_image:
                fig = self._get_image_from([Xvis], animated=True)
            else:
                scatter.set_offsets(Xvis)
                if plot_original_data:
                    scatter_orig.set_offsets(original_data[:limit_nb_datapoints])
                    return scatter, scatter_orig, 
            return scatter, 
        
        # 2500 ms per loop
        anim = animation.FuncAnimation(fig, draw_frame, frames=len(self.history), interval= 3000 / len(self.history), blit=True, init_func=init_frame)
        return anim

    def save_animation(self,
                       anim,
                       generated_data_name = "undefined_distribution",
                       filepath = SAVE_ANIMATION_PATH,
                        ):
        path = os.path.join(filepath, generated_data_name + '.mp4')
        anim.save(path)
        print('animation saved in {}'.format(path))
        return anim
    
    
    
    
    
    