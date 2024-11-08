import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os
import copy
from .datasets import inverse_affine_transform
#from IPython.display import HTML


SAVE_ANIMATION_PATH = './animation'

class GenerationManager:
    
    # same device as pdmp
    def __init__(self, 
                 method,
                 dataloader,
                 is_image,
                 **kwargs
                 ):
        self.method = method
        self.original_data = dataloader
        self.is_image = is_image
        self.kwargs = kwargs
        self.samples = []
        self.history = []

    def generate(self,
                 models,
                 nsamples,
                 get_sample_history = False, # if method supports it, get the history of the samples
                 print_progression=True,
                 **kwargs
                 ):
        assert nsamples > 0, 'nsamples must be greater than 0, got {}'.format(nsamples)
        tmp_kwargs = copy.deepcopy(self.kwargs)
        tmp_kwargs.update(kwargs)

        _, (data, y) = next(enumerate(self.original_data))
        size = list(data.size())
        size[0] = nsamples
        x = self.method.sample(shape=size,
                            models = models,
                            print_progression = print_progression,
                            get_sample_history = get_sample_history,
                            **tmp_kwargs)
        # store samples and possibly history on cpu
        # as done in LIM, clamp to -1, 1
        clamp = 1. if self.is_image else 6.
        if get_sample_history:
            # samples, hist = x
            hist = x
            self.samples = hist[-1, ..., :data.shape[-1]]
            self.history = hist[..., :data.shape[-1]].clamp(-clamp, clamp).cpu()
        else:
            self.samples = x[..., :data.shape[-1]] # select positions in case of pdmp
        self.samples = self.samples.clamp(-clamp, clamp).cpu()
        if self.is_image:
            # print('samples generated:', self.samples.mean(), self.samples.min(),self.samples.max())
            self.samples = inverse_affine_transform(self.samples)
            # print('samples generated, inverse affine transform:', self.samples.mean(), self.samples.min(),self.samples.max())
            if len(self.history) != 0:
                self.history = torch.stack([inverse_affine_transform(h) for h in self.history]) # apply inverse transform

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
                  title=None,
                  ):
        img = self._get_image_from(self.samples, 
                                    idx=idx, 
                                    black_and_white=black_and_white)
        fig = plt.figure()
        plt.imshow(img, animated = False)
        if title is not None:
            plt.title(title)
        return fig
    
    def _img_to_plt_img(self, tensor):
        return torch.stack([tensor[i]for i in range(tensor.shape[0])], dim = -1)

    def _get_image_from(self, 
                        samples,
                        idx = -1,
                        black_and_white = False):
        img = samples[idx]
        img = self._img_to_plt_img(img)
        # potentially repeat last dimension for signle channel data to be black and white
        if black_and_white and img.shape[-1] == 1:
            img = img.repeat(1, 1, 3)
        return img
    
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
                  plot_original_data=False,
                  limit_nb_datapoints=10000,
                  title=None,
                  marker='.',
                  color='blue',
                  xlim=None, 
                  ylim=None,
                  alpha=0.5,
                  method=None,
                  plot_type='line'  # 'scatter' or 'line'
                  ):
        assert method is not None, 'Must give method object to determine the time spacing'
        
        # Determine the timesteps we are working with
        timesteps = np.array([x for x in method.get_timesteps(len(self.history)-1, **self.kwargs)])
        timesteps = np.sort(timesteps)
        T = timesteps[-1]
        timesteps = torch.tensor(timesteps)

        num_frames = 60 * 6

        if plot_original_data:
            original_data = self.load_original_data(limit_nb_datapoints)
            original_data = original_data.squeeze(1)
        
        fig, ax = plt.subplots()
        if self.is_image:
            image_shape = self._img_to_plt_img(self.load_original_data(1)[0]).shape
            im = plt.imshow(np.random.random(image_shape), interpolation='none')
        else:
            if plot_type == 'scatter':
                scatter = ax.scatter([], [], alpha=alpha, animated=True, color=color,
                                    **self._get_scatter_marker_specific_kwargs(marker))
                if plot_original_data:
                    scatter_orig = ax.scatter([], [], alpha=alpha, animated=True, color='orange',
                                            **self._get_scatter_marker_specific_kwargs(marker))
            elif plot_type == 'line':
                num_samples = min(self.history.shape[1], limit_nb_datapoints)
                lines = []
                for _ in range(num_samples):
                    line, = ax.plot([], [], alpha=alpha, animated=True, color=color)
                    lines.append(line)
                if plot_original_data:
                    scatter_orig = ax.scatter([], [], alpha=alpha, color='orange',
                                            **self._get_scatter_marker_specific_kwargs(marker))
            else:
                raise ValueError("Invalid plot_type. Expected 'scatter' or 'line'.")
            # Compute xlim and ylim if not provided
            if xlim is None or ylim is None:
                if plot_type == 'line':
                    # Reshape history_stacked to (num_timesteps * num_samples, D)
                    all_data = self.history[:, :num_samples].reshape(-1, self.history.shape[-1])
                else:
                    # For scatter plot, use the data from all timesteps
                    all_history = self.history.cpu()
                    all_data = all_history.reshape(-1, all_history.shape[-1])[:limit_nb_datapoints]
                x_data = all_data[:, 0].numpy()
                y_data = all_data[:, 1].numpy()
                if plot_original_data:
                    orig_x = original_data[:limit_nb_datapoints, 0].numpy()
                    orig_y = original_data[:limit_nb_datapoints, 1].numpy()
                    x_data = np.concatenate([x_data, orig_x])
                    y_data = np.concatenate([y_data, orig_y])
                x_padding = (x_data.max() - x_data.min()) * 0.05  # 5% padding
                y_padding = (y_data.max() - y_data.min()) * 0.05
                xlim = (x_data.min() - x_padding, x_data.max() + x_padding)
                ylim = (y_data.min() - y_padding, y_data.max() + y_padding)

        def init_frame_2d():
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            # ax.set_aspect('equal', adjustable='box')  # Set aspect ratio to be equal
            ax.set_title(title)
            ax.set_xlabel(r'$x_1$')
            ax.set_ylabel(r'$x_2$')
            # ax.legend(loc='upper right')
            ax.grid(True)
            if plot_type == 'scatter':
                scatter.set_offsets(np.empty((0, 2)))
                if plot_original_data:
                    scatter_orig.set_offsets(original_data[:limit_nb_datapoints])
                    return [scatter, scatter_orig]
                else:
                    return [scatter]
            elif plot_type == 'line':
                for line in lines:
                    line.set_data([], [])
                if plot_original_data:
                    scatter_orig.set_offsets(original_data[:limit_nb_datapoints])
                    return lines + [scatter_orig]
                else:
                    return lines

        def init_frame_image():
            im.set_data(np.random.random(image_shape))
            return im, 

        def get_interpolation_values(i):
            t = T * (i / (num_frames - 1))
            k = torch.searchsorted(timesteps, t) - 1
            k = max(min(k.item(), len(timesteps) - 2), 0)
            l = (t - timesteps[k]) / (timesteps[k+1] - timesteps[k])
            return k, l

        def draw_frame_2d(i):
            k, l = get_interpolation_values(i)
            Xk1 = self.history[k+1].cpu()[:limit_nb_datapoints]
            Xk = self.history[k].cpu()[:limit_nb_datapoints]
            Xvis = (Xk1 * l + Xk * (1 - l)).squeeze(1)
            if plot_type == 'scatter':
                scatter.set_offsets(Xvis)
                if plot_original_data:
                    scatter_orig.set_offsets(original_data[:limit_nb_datapoints])
                    return scatter, scatter_orig
                else:
                    return scatter,
            elif plot_type == 'line':
                # Retrieve the trajectory data up to time k for each sample
                X_history = self.history[:k+1] # Shape: (k+1, num_samples, 1, D)
                X_history = X_history.cpu().squeeze(2) # Shape: (k+1, num_samples, D)
                # append Xvis
                X_history = torch.cat([X_history, Xvis.unsqueeze(0)], dim=0)
                for idx, line in enumerate(lines):
                    x_data = X_history[:, idx, 0].numpy()
                    y_data = X_history[:, idx, 1].numpy()
                    line.set_data(x_data, y_data)
                if plot_original_data:
                    return lines + [scatter_orig]
                else:
                    return lines

        def draw_frame_image(i):
            k, l = get_interpolation_values(i)
            Xk1 = self.history[k+1][0].cpu()
            Xk = self.history[k][0].cpu()
            Xvis = Xk1 * l + Xk * (1 - l)
            img = self._get_image_from([Xvis], black_and_white=True)
            im.set_data(img)
            return im,

        if self.is_image:
            anim = animation.FuncAnimation(fig, draw_frame_image, frames=num_frames,
                                        interval=6000 / num_frames, blit=True,
                                        init_func=init_frame_image)
        else:
            anim = animation.FuncAnimation(fig, draw_frame_2d, frames=num_frames,
                                        interval=6000 / num_frames, blit=True,
                                        init_func=init_frame_2d)
        return anim
    
    def save_animation(self,
                       anim = None,
                       generated_data_name = "undefined_distribution",
                       filepath = SAVE_ANIMATION_PATH,
                        ):    
        path = os.path.join(filepath, generated_data_name + '.mp4')
        anim.save(path)
        print('animation saved in {}'.format(path))
        return anim
    
    
    
    
    
    