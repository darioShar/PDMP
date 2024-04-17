import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
#from IPython.display import HTML


SAVE_ANIMATION_PATH = './animation'


''' 
This class is used to generate samples from a PDMP process.
It takes a model and a PDMP objects as input, and a dataloader to sample initial data from.
'''
class GenerationManager:
    
    # same device as diffusion
    def __init__(self, 
                 model, 
                 pdmp, 
                 dataloader,
                 is_image = False
                 ):
        self.model = model
        self.original_data = dataloader
        self.samples = []
        self.history = []
        self.pdmp = pdmp

    # generate nsamples from the reverse PDMP process and our model
    # we need to implement pdmp.reverse_sampling first
    def generate(self, 
                 nsamples,
                 num_timesteps = None, # keep default
                 clip_denoised = False,
                 print_progession = False,
                 use_samples = None,
                 get_sample_history = False,
                 reduce_timesteps = 1.):
        _, (data, _) = next(enumerate(self.original_data))
        size = list(data.size())
        size[0] = nsamples
        if num_timesteps is not None:
            default_reverse_steps = self.pdmp.reverse_steps
            self.pdmp.reverse_steps = num_timesteps
        chain = self.pdmp.reverse_sampling(
                        nsamples=nsamples,
                        model=self.model,
                        initial_data = use_samples, # sample from Gaussian, else use this initial_data
                        print_progession = print_progession,
                        )
        if num_timesteps is not None:
            self.pdmp.reverse_steps = default_reverse_steps
        # chain is [time, particle, channel, (position, speed)]
        self.samples = chain[-1, :, :, :2].cpu() # get positions
        # store history on cpu
        self.history = torch.stack([h.cpu() for h in chain])
    

    # if the data is image, return the last sample as a pyplot figure
    def get_image(self):
        img = self.samples[-1].squeeze(0)
        fig = plt.figure()
        plt.imshow(img)
        plt.close(fig)
        return fig
    
    # if the data is 2d, return a scatter plot of the samples.
    # Optionally, plot the original data as well
    def get_plot(self, 
                 plot_original_data = True, 
                 limit_nb_orig_data = 5000,
                xlim = None, ylim = None):
        
        original_data = self.load_original_data(limit_nb_orig_data)
        gen_data = self.samples
        # squeeze
        original_data = original_data.squeeze(1)
        gen_data = gen_data.squeeze(1)
        fig = plt.figure()
        if plot_original_data:
            self._plot_data(original_data, limit_nb_orig_data)
        self._plot_data(gen_data, marker='+')
        if xlim is not None:
            plt.xlim(xlim) 
        if ylim is not None:
            plt.ylim(ylim)
        if plot_original_data:
            plt.legend(["Original data", "Generated data"])
        else:
            plt.legend('Generated data')
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
    
    # load original data from the dataloader
    def load_original_data(self, nsamples):
        data_size = 0
        total_data = torch.tensor([])
        while data_size < nsamples:
            _, (data,_) = next(enumerate(self.original_data))
            total_data = torch.concat([total_data, data])
            data_size += data.size()[0]
        return total_data[:nsamples]
    
    # Create an animation of the generation of original data, saved in self.history,
    # and save it to a file.
    def animation(self,
                  generated_data_name = "undefined_distribution",
                plot_original_data = True, 
                filepath = SAVE_ANIMATION_PATH, 
                xlim = (-2.5, 2.5), ylim = (-2.5, 2.5),
                limit_nb_orig_data = 5000):
        
        original_data = self.load_original_data(limit_nb_orig_data)
        original_data = original_data.squeeze(1)
        
        def draw_frame(i):
            plt.clf()
            Xvis = self.history[i].cpu().squeeze(1)
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
        # 60 fps, 16.66ms
        anim = animation.FuncAnimation(fig, draw_frame, frames=len(self.history), interval=17, blit=True)
        path = os.path.join(filepath, 
                            generated_data_name+ '.mp4')
        anim.save(path, fps=60)
        return path
    
    
    
    
    
    