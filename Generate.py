import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
#from IPython.display import HTML


SAVE_ANIMATION_PATH = './animation'

class GenerationManager:
    
    # same device as diffusion
    def __init__(self, 
                 model, 
                 pdmp, 
                 dataloader
                 ):
        self.model = model
        self.original_data = dataloader
        self.samples = []
        self.history = []
        self.pdmp = pdmp

    def generate(self, 
                 nsamples,
                 use_samples = None,
                 get_sample_history = False):
        _, (data, y) = next(enumerate(self.original_data))
        size = list(data.size())
        size[0] = nsamples 
        x = self.pdmp.reverse_sampling(
                        nsamples,
                        initial_data = use_samples, # sample from Gaussian, else use this initial_data
                        get_history = get_sample_history # else store history of data points in a list
                        )
        # store samples and possibly history on cpu
        if get_sample_history:
            samples, hist = x
            self.history = [h.cpu() for h in hist]
        else:
            samples = x
        self.samples = samples.cpu()

    def get_image(self):
        img = self.samples[-1].squeeze(0)
        fig = plt.figure()
        plt.imshow(img)
        plt.close(fig)
        return fig
    
    def get_plot(self, 
                 plot_original_data = True, 
                 limit_nb_orig_data = 5000,
                xlim = None, ylim = None):
        
        original_data = self.load_original_data(limit_nb_orig_data)
        gen_data = self.samples
        # squeeze
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
    
    def load_original_data(self, nsamples):
        data_size = 0
        total_data = torch.tensor([])
        while data_size < nsamples:
            _, (data,y) = next(enumerate(self.original_data))
            total_data = torch.concat([total_data, data])
            data_size += data.size()[0]
        return total_data.squeeze(1)[:nsamples]
    
    def animation(self,
                  generated_data_name = "undefined_distribution",
                plot_original_data = True, 
                filepath = SAVE_ANIMATION_PATH, 
                xlim = (-2.5, 2.5), ylim = (-2.5, 2.5),
                limit_nb_orig_data = 5000):
        
        original_data = self.load_original_data(limit_nb_orig_data)

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
    
    
    
    
    
    