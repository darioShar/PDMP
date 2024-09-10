from tqdm import tqdm
import torch
import torch.nn as nn
import PDMP.pdmp_utils.Data as Data
from PDMP.pdmp_utils.Distributions import *
import enum
import torch as th
from PDMP.LIM.functions.sde import VPSDE
from PDMP.LIM.functions.sampler import LIM_sampler
from PDMP.LIM.torchlevy import LevyStable
import PDMP.LIM.functions.loss as lim_loss

class ModelMeanType():
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = 'PREVIOUS_X'  # the model predicts x_{t-1}
    START_X = 'START_X'  # the model predicts x_0
    EPSILON = 'EPSILON'  # the model predicts epsilon
    Z = 'Z' # the model predicts z_t
    SQRT_GAMMA_EPSILON = 'SQRT_GAMMA_EPSILON' # the model predicts (1 - sqrt(1 - gamma))eps


class ModelVarType():
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    FIXED = 'FIXED' # fixed variance
    GAMMA = 'GAMMA' # can learn gamma factor, with the true var being tilde_sigma = gamma * sigma
    #SIGMA_T_1_OVER_A = enum.auto() # learn directly Sigma_t_1 / a_t
    # TRUE_VAR = enum.auto()

class LossType():
    # can replace different means by normal loss and lambda loss, with some lambda to
    # pass as argument in training. should be better.
    LP_LOSS = 'LP_LOSS' # L_p loss between model output and target value
    LP_MEAN_LOSS = 'LP_MEAN_LOSS' # LP loss between model mean and true mean
    LP_EPS_LOSS = 'LP_EPS_LOSS' # lp loss between noise and model noise
    LP_LAMBDA_LOSS = 'LP_LAMBDA_LOSS' #  L_p loss between model output and target value, times some lambda factor true mean adn predicted mean (=eps * lambda factor)
    L1_SMOOTH = 'L1_SMOOTH' # L1 smooth loss between eps and model_eps
    VAR_KL = 'VAR_KL' # when mixing mean and var.
    VAR_LP_SUM = 'VAR_LP_SUM' # when mixing mean and var.

class LevyDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in (0 to 1).
    """

    def __init__(
        self,
        *,
        alpha,
        device,
        diffusion_steps,
        model_mean_type,
        model_var_type,
        loss_type,
        time_spacing = 'linear',
        rescale_timesteps=False,
        isotropic = True, # isotropic levy noise
        clamp_a = None,
        clamp_eps = None,
        LIM = False,
        diffusion_settings = None
    ):
        self.alpha = alpha
        self.device = device
        self.diffusion_steps = diffusion_steps
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.time_spacing = time_spacing
        self.rescale_timesteps = rescale_timesteps
        self.isotropic = isotropic
        self.LIM = LIM
        self.diffusion_settings = diffusion_settings # to remove
        #if config is not None:
        #    self.config.alpha = self.alpha
        #    self.config.nfe = self.diffusion_steps

        if self.LIM:
            assert (self.model_mean_type == ModelMeanType.EPSILON) \
            and (self.model_var_type == ModelVarType.FIXED) \
            and (self.rescale_timesteps == True), "LIM only supports epsilon prediction, fixed variance and rescaled timesteps"
            self.sde = VPSDE(alpha, 'cosine') #config.diffusion.beta_schedule)
            self.levy = LevyStable()
        
        # constants 
        self.gammas = self.gen_noise_schedule(self.diffusion_steps, self.time_spacing).to(device)
        self.bargammas = th.cumprod(self.gammas, dim = 0)
        self.skewed_levy_data = Data.Generator('skewed_levy', 
                                        alpha = self.alpha, 
                                        device=self.device,
                                        isotropic=isotropic,
                                        clamp_a = clamp_a)
        self.get_eps = Data.Generator('sas',
                                      alpha = self.alpha, 
                                      device=self.device,
                                      isotropic=isotropic,
                                      clamp_eps = clamp_eps)
    
    def rescale_noising(self, diffusion_steps, time_spacing = None):
        assert isinstance(diffusion_steps, int), "Diffusion steps must be an integer"
        assert self.rescale_timesteps, "Rescaling only works when rescale_timesteps is True" 
        # just update with this number of diffusion steps regenerate noise schedule
        if time_spacing is not None:
            self.time_spacing = time_spacing
        self.diffusion_steps = diffusion_steps
        self.gammas = self.gen_noise_schedule(self.diffusion_steps, self.time_spacing).to(self.device)
        self.bargammas = th.cumprod(self.gammas, dim = 0)

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        g, bg = self._get_schedule_to_match_last_dims(x_t)
        t = self._get_t_to_batch_size(x_t, t)
        xstart = x_t - eps*((1 - bg[t])**(1/self.alpha))
        return xstart / (bg[t]**(1/self.alpha))

    def _predict_eps_from_xstart(self, x_t, t, xstart):
        g, bg = self._get_schedule_to_match_last_dims(x_t)
        t = self._get_t_to_batch_size(x_t, t)
        eps = x_t - xstart* (bg[t]**(1/self.alpha))
        return eps / ((1 - bg[t])**(1/self.alpha))


    def _get_schedule_to_match_last_dims(self, Xbatch):
        g = match_last_dims(self.gammas, Xbatch.size())
        bg = match_last_dims(self.bargammas, Xbatch.size())
        return g, bg
    
    def _get_t_to_batch_size(self, Xbatch, t):
        if isinstance(t, int):
            return torch.full([Xbatch.size()[0]], t).to(self.device)
        return t
    
    def _get_noises_to_incoming_batch_dims(self, Xbatch, t):
        #if (isinstance(t, int) and t == 1) or (t[0] == 1) :
        #    a_t_1 = torch.zeros(Xbatch.size()).to(self.device)
        #else:
        a_t_1 = self.skewed_levy_data.generate(size = Xbatch.size())
        # zero variance when t == 1
        zero_loc = torch.where(t == 1)[0]
        if zero_loc is not tuple():
            a_t_1[zero_loc] = torch.zeros(
                                (zero_loc.shape[0], 
                                 *(a_t_1.shape[1:]))
                                 ).to(self.device)
        # ok
        a_t_prime = self.skewed_levy_data.generate(size = Xbatch.size())
        a_t = self.compute_a_t(t, a_t_prime, a_t_1)
        return a_t_1, a_t_prime, a_t
    
    def _get_constants_to_incoming_batch_dims(self, Xbatch, t):
        g, bg = self._get_schedule_to_match_last_dims(Xbatch)
        t = self._get_t_to_batch_size(Xbatch, t)
        a_t_1, a_t_prime, a_t = self._get_noises_to_incoming_batch_dims(Xbatch, t)
        return g, bg, t, a_t_1, a_t_prime, a_t

    def q_sample(self, x_start, t, eps = None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start, and eps
        """
        #g, bg = self._get_schedule_to_match_last_dims(x_start)
        #t = self._get_t_to_batch_size(x_start, t)
        #if eps is None: 
        #    eps = self.get_eps.generate(size = x_start.shape)
        x_coeff = (self.bargammas[t] ** (1/self.alpha))
        noise_coeff = ((1 - self.bargammas[t]) ** (1/self.alpha))
        #mean = (bg[t] ** (1/self.alpha)) * x_start 
        #noise = ((1 - bg[t]) ** (1/self.alpha)) * eps
        return x_coeff, noise_coeff

    def q_posterior_mean_variance(self, x_start, x_t, a_t_1, a_t_prime, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0, a_t_1, a_t')

        """
        assert x_start.shape == x_t.shape

        eps = self._predict_eps_from_xstart(x_t, t, x_start)
        a_t = self.compute_a_t(t, a_t_prime, a_t_1)
        Gamma = self.gamma_factor(t, a_t, a_t_1)
        posterior_mean = self.anterior_mean(t, x_t, eps, Gamma)
        posterior_variance = self.sigma_tilde(t, a_t, a_t_1)

        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance

    """
    Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
    the initial x, x_0.

    :param model: the model, which takes a signal and a batch of timesteps
                    as input.
    :param x: the [N x C x ...] tensor at time t.
    :param t: a 1-D Tensor of timesteps.
    :param clip_denoised: if True, clip the denoised signal into [-1, 1].
    :param denoised_fn: if not None, a function which applies to the
        x_start prediction before it is used to sample. Applies before
        clip_denoised.
    :param model_kwargs: if not None, a dict of extra keyword arguments to
        pass to the model. This can be used for conditioning.
    :return: a dict with the following keys:
                - 'mean': the model mean output.
                - 'gamma': the model gamma output, if learned
                -  'Sigma': the forward variance. 
                            The reverse variance is Sigma_tilde_t= Gamma_t*Sigma_{t-1}
                - 'xstart': the prediction for x_0.
    """
    def p_mean_variance(self, 
                        model, 
                        x, 
                        t, 
                        clip_denoised=False, 
                        denoised_fn=None, 
                        model_kwargs=None):
        # to process outpout of model
        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x
        
        # get constants
        if model_kwargs is None:
            model_kwargs = {}
        g, bg, t, a_t_1, a_t_prime, a_t =  \
            self._get_constants_to_incoming_batch_dims(x, t)
        B, C = x.shape[:2]
        assert t.shape == (B,)

        # run model
        noise_coeff = ((1 - self.bargammas[t]) ** (1/self.alpha))
        #model_output = model(x, self._scale_timesteps(t), **model_kwargs)
        model_output = model(x, self._scale_timesteps(t), **model_kwargs)

        # get gamma factor
        if self.model_var_type in [ModelVarType.GAMMA]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_gamma = th.split(model_output, C, dim=1)
        else:
            model_gamma = self.gamma_factor(t, a_t, a_t_1, (g, bg))

        # get variance
        # use this formula to make use of our computed gamma. important to obtain equivalence between ddim and ddpm in case of computed gamma.
        model_variance = model_gamma \
                        * self.compute_Sigma_t_1_over_a_t(t, model_gamma, (g, bg)) \
                        * a_t
        
        if (self.model_mean_type == ModelMeanType.EPSILON) and (not clip_denoised):
            # just bypass everyhting
            model_eps = model_output
        else:
            if self.model_mean_type == ModelMeanType.EPSILON:
                model_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            elif self.model_mean_type == ModelMeanType.START_X:
                model_xstart = process_xstart(model_output)
            elif self.model_mean_type == ModelMeanType.Z:
                eps = torch.sqrt(a_t) * model_output
                model_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=eps)
                )
            elif self.model_mean_type == ModelMeanType.SQRT_GAMMA_EPSILON:
                eps = model_output / (1 - torch.sqrt(1 - model_gamma))
                model_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=eps)
                )
            elif self.model_mean_type == ModelMeanType.PREVIOUS_X:
                # get xstart from x_previous. use model_gamma
                eps = self._eps_from_previous_x(t, x, model_output, model_gamma, (g, bg))
                model_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=eps)
                )
            else:
                raise NotImplementedError(self.model_mean_type)

            # now get the new values after potential clipping and denoising
            model_eps = self._predict_eps_from_xstart(x_t=x,
                                                        t=t,
                                                        xstart=model_xstart)
        

        # do not use q_posterior_mean_variance to get mean from eps or xstart, since it may use fixed gamma instead of computed one.
        model_mean = self.anterior_mean(t, x, model_eps, model_gamma, (g, bg))

        # some assertion
        assert model_mean.shape == x.shape
        return {
            #'output': model_output,
            'eps': model_eps,
            'mean': model_mean,
            #'gamma': model_gamma,
            'variance': model_variance,
            #'xstart': model_xstart,
            #'a_t_1': a_t_1,
            #'a_t': a_t
        }

    
    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1.0 / self.diffusion_steps)
        return t
    
    """
    Sample x_{t-1} from the model at the given timestep.

    :param model: the model to sample from.
    :param x: the current tensor at x_{t-1}.
    :param t: the value of t, starting at 0 for the first diffusion step.
    :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
    :param denoised_fn: if not None, a function which applies to the
        x_start prediction before it is used to sample.
    :param model_kwargs: if not None, a dict of extra keyword arguments to
        pass to the model. This can be used for conditioning.
    :return: a dict containing the following keys:
                - 'sample': a random sample from the model.
                - 'xstart': a prediction of x_0.
    """
    def p_sample(
        self, model, x, t, clip_denoised=False, denoised_fn=None, model_kwargs=None
    ):
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x)
        nonzero_mask = (
            (t != 1).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 1
        sample = out["mean"] + nonzero_mask * torch.sqrt(out["variance"]) * noise
        return {"sample": sample}#, "xstart": out["xstart"]}

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=False,
        denoised_fn=None,
        model_kwargs=None,
        progress=False,
        get_sample_history = False
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        if progress:
            tqdm._instances.clear()
            pbar = tqdm(total = self.diffusion_steps)
        model.eval()
        x_hist = []
        with th.inference_mode():
            final = None
            for sample in self.p_sample_loop_progressive(
                model,
                shape,
                noise=noise,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs
            ):
                final = sample
                if get_sample_history:
                    x_hist.append(sample['sample'])
                if progress:
                    pbar.update(1)
            if progress:
                pbar.close()
                tqdm._instances.clear()
            if get_sample_history:
                return torch.stack(x_hist)
            return final["sample"]

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=False,
        denoised_fn=None,
        model_kwargs=None
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        assert self.device is not None 
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = self.get_eps.generate(size = shape)
        indices = list(range(self.diffusion_steps-1, 0, -1))

        for i in indices:
            t = th.tensor([i] * shape[0], device=self.device)
            out = self.p_sample(
                model,
                img,
                t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
            )
            yield out
            img = out["sample"]

    def ddim_sample(
        self,
        model,
        x,
        t,
        clip_denoised=False,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """

        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # deterministic sampling, second version 
        g, bg = self._get_schedule_to_match_last_dims(x)
        t = self._get_t_to_batch_size(x, t)
        nonzero_mask = (t != 1).float().view(-1, *([1] * (len(x.shape) - 1))) # no noise when t == 1
        sigma_t = eta * (1 - bg[t-1])**(1/self.alpha)
        noise = self.get_eps.generate(size = x.shape)
        eps = out['eps']
        sample = (x - (1 - bg[t])**(1 / self.alpha)*eps) / g[t]**(1 / self.alpha)
        sample += (1 - bg[t-1] - sigma_t**(self.alpha))**(1 / self.alpha)*eps
        sample += nonzero_mask* sigma_t*noise


        #sample = anterior + nonzero_mask * sig_t * noise

        assert not torch.isnan(sample).any() # verify no nan
        return {"sample": sample}#, "xstart": out["xstart"]}

    def ddim_reverse_sample(
        self,
        model,
        x,
        t,
        clip_denoised=False,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        assert False
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["xstart"]
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        # Equation 12. reversed
        mean_pred = (
            out["xstart"] * th.sqrt(alpha_bar_next)
            + th.sqrt(1 - alpha_bar_next) * eps
        )

        return {"sample": mean_pred, "xstart": out["xstart"]}

    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=False,
        denoised_fn=None,
        model_kwargs=None,
        progress=False,
        eta=0.0,
        get_sample_history = False
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        if progress:
            tqdm._instances.clear()
            pbar = tqdm(total = self.diffusion_steps)

        model.eval()
        x_hist = []
        with th.inference_mode():
            final = None
            for sample in self.ddim_sample_loop_progressive(
                model,
                shape,
                noise=noise,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
                eta=eta,
            ):
                final = sample
                if get_sample_history:
                    x_hist.append(sample['sample'])
                if progress:
                    pbar.update(1)
            if progress:
                pbar.close()
                tqdm._instances.clear()
            if get_sample_history:
                return final['sample'], x_hist
            return final["sample"]

    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=False,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        assert self.device is not None
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = self.get_eps.generate(size = shape)
        yield {'sample': img}
        indices = list(range(self.diffusion_steps-1, 0, -1))
        for i in indices:
            t = th.tensor([i] * shape[0], device=self.device)
            out = self.ddim_sample(
                model,
                img,
                t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
                eta=eta,
            )
            yield out
            img = out["sample"]


    def lim_sample(self,
                   model,
                   shape,
                   ddim = False,
                   get_sample_history = False,
                   clip_denoised = False
                   ):
                    
        init_clamp = 20. #config.sampling.init_clamp # is 20 in LIM configs, can be set to None
        
        if self.sde.alpha == 2.0:
            # Gaussian noise
            x = torch.randn(shape).to(self.device)
        else:
            if self.isotropic:
                # isotropic
                x = self.levy.sample(alpha=self.sde.alpha, size=shape, is_isotropic=True, clamp=init_clamp).to(self.device)
            else:
                # non-isotropic
                x = torch.clamp(self.levy.sample(self.sde.alpha, size=shape, is_isotropic=False, clamp=None).to(self.device), 
                                min=-init_clamp, max=init_clamp)
            
        if False:#config.model.is_conditional:
            if config.sampling.cond_class is not None: 
                y = torch.ones(n) * config.sampling.cond_class
                y = y.to(self.device)
                y = torch.tensor(y, dtype=torch.int64)
            else:
                y = None
        else:
            y = None
            
        #x = LIM_sampler(args, config, x, y, model, self.sde, self.levy)
        samples = LIM_sampler(#args=self.config, 
                              #config=self.config,
                              ddim = ddim,
                              x = x,
                              y = None,
                              model = model,
                              sde = self.sde,
                              levy = self.levy,
                              isotropic=self.isotropic,
                              steps=self.diffusion_steps,
                              clamp_a = self.diffusion_settings['clamp_a'],
                              clamp_eps = self.diffusion_settings['clamp_eps'],
                              device = self.device,
                              get_sample_history = get_sample_history)
        # The clamping and inverse affine transform will be managed in the generation manager.
        #x = x.clamp(-1.0, 1.0)
        #x = (x + 1) / 2
        return samples


    def sample(self, 
                models,
                shape,
                reverse_steps,
                time_spacing = None,
                initial_data = None, 
                clip_denoised=False, 
                ddim=False, 
                eta=1.0, 
                print_progression=False, 
                get_sample_history=False):
        
        model = models['default']

        # rescale noising with the number of provided reverse_steps
        assert time_spacing is None, "Specific time spacing is not yet supported for diffusion reverse sampling"
        default_diffusion_steps = self.diffusion_steps # store original number to restore.
        default_time_spacing = self.time_spacing
        self.rescale_noising(reverse_steps, time_spacing=time_spacing)

        if self.LIM:
            x = self.lim_sample(model, 
                                shape = shape,
                                ddim = ddim,
                                get_sample_history = get_sample_history,
                                clip_denoised = clip_denoised)
        else:
            if ddim:
                x = self.ddim_sample_loop(model,
                                        shape = initial_data.shape if initial_data is not None else shape,
                                        noise = initial_data if initial_data is not None else None,
                                        eta = eta,
                                        progress=print_progression,
                                        get_sample_history = get_sample_history,
                                        clip_denoised = clip_denoised)
            else:
                x = self.p_sample_loop(model,
                                        shape = shape,
                                        progress=print_progression,
                                        get_sample_history = get_sample_history,
                                        clip_denoised = clip_denoised)
        # restore original diffusion steps
        self.rescale_noising(default_diffusion_steps, default_time_spacing)
        return x

    def training_losses_fixed_a(self, 
                        model, 
                        x_start, 
                        t, 
                        g, bg,
                        a_t_1, a_t_prime, a_t,
                        eps,
                        lploss =1.0,
                        model_kwargs=None):
        # compute resulting noised input
        #x_t, _ = self.q_sample(x_start, t, eps=eps) # x_t, eps
        x_coeff, noise_coeff = self.q_sample(x_start, t, eps=eps)
        x_t =  match_last_dims(x_coeff, x_start.shape) * x_start + match_last_dims(noise_coeff, eps.shape) * eps

        # compute true values
        true_mean, true_var = self.q_posterior_mean_variance(
                x_start=x_start, x_t=x_t, a_t_1=a_t_1, a_t_prime = a_t_prime, t=t
            )
        true_gamma = self.gamma_factor(t, a_t, a_t_1)
        
        # run model
        # give noise scale to the model
        #model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)
        model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)
        # to store model output
        terms = {}
        #torch.autograd.set_detect_anomaly(True)              
        if self.model_var_type in [ModelVarType.GAMMA]:
            # check shape
            B, C = x_t.shape[:2]
            assert model_output.shape == (B, C * 2, *x_t.shape[2:])
            # split output
            model_output, model_v = th.split(model_output, C, dim=1)
            # compute variance from gamma
            terms['v'] = model_v
            Sigma_t_1 = (1 - bg[t-1])**(2 / self.alpha) *a_t_1  #self.compute_Sigma_t_1_over_a_t(t, true_gamma, (g, bg)) * a_t
            #model_gamma = torch.pow(true_gamma, model_v)
            model_variance = torch.exp((1-model_v) * torch.log(Sigma_t_1) + model_v * torch.log(true_var))
            # true_var is Gamma_t * Sigma_t_1
            # model_var is equivalently model_gamma* Sigma_t_1

            #model_variance = model_gamma \
            #        * self.compute_Sigma_t_1_over_a_t(t, model_gamma, (g, bg)) \
            #        * a_t 
            terms['variance'] = model_variance

        value_target = {
            ModelMeanType.PREVIOUS_X: true_mean,
            ModelMeanType.START_X: x_start,
            ModelMeanType.EPSILON: eps,
            ModelMeanType.Z: eps / torch.sqrt(a_t),
            ModelMeanType.SQRT_GAMMA_EPSILON: (1 - torch.sqrt(1 - true_gamma))*eps
        }[self.model_mean_type]
        assert model_output.shape == value_target.shape == x_start.shape

        # return 1-d vector (num batches) of loss
        def compute_loss(tens):
            assert False, 'NYI'
            return torch.pow(torch.linalg.norm(tens, 
                                               ord = lploss, # always 2. here instead of lploss 
                                               dim = list(range(1, len(tens.shape)))), 
                            1 / lploss)
        def compute_loss_terms(x, y):
            if lploss == 2.:
                tmp = nn.functional.mse_loss(x, y, reduction='none')
                tmp = torch.sqrt(tmp.mean(dim = list(range(1, len(x.shape)))))
                return tmp
            elif lploss == 1.:
                tmp = nn.functional.smooth_l1_loss(x, y, beta=1, reduction='none')
                tmp = tmp.mean(dim = list(range(1, len(x.shape))))
                return tmp
            else:
                return compute_loss(x - y)

        # if computing variance
        if self.model_var_type == ModelVarType.GAMMA:
            model_mean = self.anterior_mean_from_model_output(t, 
                                                x_t, 
                                                model_output,
                                                true_gamma,#model_gamma, #model_gamma,
                                                a_t,
                                                (g, bg))
            if self.loss_type == LossType.VAR_KL:
                model_var = terms['variance'] + 1e-6 # to stabilize, for instance when using null init
                true_var += 1e-6
                first_term = .5*(torch.log(model_var) - torch.log(true_var))
                first_term += true_var / (2* model_var) - .5
                # detach for the second term
                v = model_var # .detach() # does not contribute to Gamma gradient.
                m = model_mean.detach()
                # factor = 1 if torch.mean(v) < 1e-4 else 1 / (2*v)
                second_term = (m - true_mean)**2 / (2*v)
            elif self.loss_type == LossType.VAR_LP_SUM:
                first_term = torch.log(model_var) - torch.log(true_var)
                # set terms at t==1 to zero
                second_term = model_output - value_target
            # for the record
            terms['true_mean'] = true_mean
            terms['model_mean'] = model_mean
            terms['true_var'] = true_var
            # compute losses
            terms['first_term'] = first_term
            terms['second_term'] = second_term
            idx = torch.stack(torch.where(torch.isnan(first_term)))
            assert not torch.isnan(first_term).any() # verify not nan
            assert not torch.isnan(second_term).any() # verify not nan
            # compute final loss
            terms['loss'] = (terms['first_term'] + terms['second_term']).mean(dim = list(range(1, len(first_term.shape))))
            # compute_loss(torch.sqrt(first_term + second_term))
            # add simple loss 
            model_eps = self.eps_from_model_output(t, 
                                                x_t, 
                                                model_output,
                                                true_gamma,
                                                a_t,
                                                (g, bg))
            lambda_vlb = 1
            terms['loss'] = compute_loss_terms(model_eps, eps) + lambda_vlb * terms['loss']
            return terms


        # lambda constant
        # lamb = torch.ones(value_target.shape, device = self.device)
        # lamb = self.compute_lambda(t, a_t, a_t_1)
        if self.loss_type == LossType.LP_EPS_LOSS:
            model_eps = self.eps_from_model_output(t, 
                                                x_t, 
                                                model_output,
                                                true_gamma,
                                                a_t,
                                                (g, bg))
            terms['loss'] = compute_loss_terms(model_eps, eps)
        elif self.loss_type == LossType.LP_LOSS:
            terms['loss'] = compute_loss_terms(model_output, value_target)
        elif self.loss_type == LossType.LP_MEAN_LOSS:
            model_mean = self.anterior_mean_from_model_output(t, 
                                                x_t, 
                                                model_output,
                                                true_gamma,
                                                a_t,
                                                (g, bg))
            terms['loss'] = compute_loss_terms(model_mean, true_mean)
        else:
            raise Exception('NYI: {}'.format(self.loss_type))
        return terms

    def training_losses_lim(self, 
                            model, 
                            x_start, 
                            t, 
                            eps, 
                            y = None):
        #assert self.rescale_timesteps, "Lim loss only works when rescale_timesteps is True (need time between 0 and 1)"
        #t_rescaled = self._scale_timesteps(t)
        return lim_loss.loss_fn(model, self.sde, x_start, t, eps, config = None, y = None)
        sigma = self.sde.marginal_std(t_rescaled)
        x_coeff = self.sde.diffusion_coeff(t_rescaled)
        
        x_t = x_start * match_last_dims(x_coeff, x_start.shape) \
            + eps * match_last_dims(sigma, x_start.shape)
        '''
        if self.sde.alpha == 2.0:
            score = - eps  
        else:
            score = - eps / self.sde.alpha
        '''
        target = eps

        if y is None:
            #output = model(x_t, t)
            output = model(x_t, t_rescaled)
        else:
            #output = model(x_t, t, y)
            output = model(x_t, t_rescaled, y)

        d = 1. #x0.shape[-1] * x0.shape[-2] * x0.shape[-3]
        
        return nn.functional.smooth_l1_loss(output, target, beta=1, reduction='none')

    def training_losses(self, 
                        models, 
                        x_start, 
                        lploss =1.0,
                        monte_carlo_steps = 1,
                        loss_monte_carlo  = 'mean',
                        monte_carlo_groups = 1,
                        model_kwargs=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, a dict of the noises at that step.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """

        model = models['default']

        if model_kwargs is None:
            model_kwargs = {}
        if model_kwargs is None:
            model_kwargs = {}

        x_start = x_start.to(self.device)

        if self.LIM:
            n = x_start.size(0)
            # Noise
            if self.sde.alpha == 2.0:
                # gaussian noise
                e = torch.randn_like(x_start).to(self.device)
            else:
                clamp = 20 #self.config.diffusion.clamp # set to 20 in their configs. Can be set to None
                if self.isotropic:
                    # isotropic
                    e = self.levy.sample(alpha=self.sde.alpha, size=x_start.shape, is_isotropic=True, clamp=clamp).to(self.device)
                else:
                    # non-isotropic
                    e = torch.clamp(self.levy.sample(self.sde.alpha, size=x_start.shape, is_isotropic=False, clamp=None).to(self.device), 
                                    min=-clamp, max=clamp)

            # time
            eps = 1e-5  
            t = torch.rand(n).to(self.device) * (self.sde.T - eps) + eps

            terms = {}
            losses = self.training_losses_lim(model, x_start, t, e, y = None)
        else:
            # get timesteps
            if self.model_var_type in [ModelVarType.GAMMA]:
                # we don't know how to handle t == 1
                t = torch.randint(2, self.diffusion_steps, size=[len(x_start)]).to(self.device)
            else:
                t = torch.randint(1, self.diffusion_steps, size=[len(x_start)]).to(self.device)
            x_start_extended = x_start.repeat(monte_carlo_steps, *([1]*len(x_start.shape[1:])))
            t_extended = t.repeat(monte_carlo_steps)
            g, bg, t_extended, a_t_1, a_t_prime, a_t = self._get_constants_to_incoming_batch_dims(x_start_extended, t_extended)
            z_t = torch.randn_like(x_start, device=self.device)
            z_t_extended = z_t.repeat(monte_carlo_steps,  *([1]*len(x_start.shape[1:])))
            eps = torch.sqrt(a_t)* z_t_extended
            terms = self.training_losses_fixed_a(model, 
                                            x_start_extended, 
                                            t_extended, 
                                            g,
                                            bg,
                                            a_t_1, 
                                            a_t_prime, 
                                            a_t, 
                                            eps,
                                            lploss = lploss,
                                            model_kwargs=model_kwargs)
            losses = terms['loss']
            assert monte_carlo_steps % monte_carlo_groups == 0
            if loss_monte_carlo == 'mean':
                losses = losses.mean()#(dim = 0)
            elif loss_monte_carlo == 'median':
                losses = losses.reshape(monte_carlo_steps // monte_carlo_groups, monte_carlo_groups, x_start.shape[0])
                losses = losses.mean(dim = 0) # now got a loss for each group
                # better to take median of mean than mean of median
                losses, _ = losses.median(dim = 0)
                losses = losses.mean()
        # assert not nan
        assert not torch.isnan(losses).any(), 'Nan in losses'
        return losses
        
        

    @staticmethod
    def gen_noise_schedule(steps,
                           time_spacing = 'linear'):
        # Set noising variances betas as in Nichol and Dariwal paper (https://arxiv.org/pdf/2102.09672.pdf)
        s = 0.008
        if time_spacing == 'linear':
            timesteps = torch.tensor(range(0, steps), dtype=torch.float32)
        elif time_spacing == 'quadratic':
            timesteps = steps* (torch.tensor(range(0, steps), dtype=torch.float32) / steps)**2
        else:
            raise NotImplementedError(time_spacing)
        schedule = torch.cos((timesteps / steps + s) / (1 + s) * torch.pi / 2)**2

        baralphas = schedule / schedule[0]
        betas = 1 - baralphas / torch.concatenate([baralphas[0:1], baralphas[0:-1]])
        alphas = 1 - betas

        # linear schedule for gamma
        # for the moment let's use the same schedule alphas
        gammas = alphas
    
        return gammas
    
    
    '''def compute_sigma(self, a_skewed):
        g, bg = self.gammas, self.bargammas
        alpha = self.alpha
        # torch.cumsum(a_skewed*((1 - gammas)/bargammas)**(2 / alpha)) * bargammas**(2 / alpha)
        return torch.cumsum(a_skewed*((1 - g)/bg)**(2 / alpha)) * bg**(2 / alpha)'''

    # in order not to compute the gammas, bargammas constants each time.
    def get_constants(self, constants, size):
        if constants is None:
            g = match_last_dims(self.gammas, size)
            bg = match_last_dims(self.bargammas, size)
        else:
            g, bg = constants
        return g, bg

    def compute_a_t(self, t, a_t_prime, a_t_1, constants = None):
        g, bg = self.get_constants(constants, a_t_1.size())
        return g[t]**(2/self.alpha)*a_t_1 \
                            + (1 - g[t])**(2 / self.alpha) * a_t_prime
    
    def Sigma_t(self, t, a_t, constants = None):
        g, bg = self.get_constants(constants, a_t.size())
        return a_t * (1 - bg[t])**(2/self.alpha)
    
    # should already have gammas[t] and bargammas[t] as input
    def gamma_factor(self, t, a_t, a_t_1, constants = None):
        g, bg = self.get_constants(constants, a_t.size())
        return (1 - (g[t]**(2 / self.alpha) * self.Sigma_t(t-1, a_t_1, (g, bg))) 
            / self.Sigma_t(t, a_t, (g, bg)))
    
    def sigma_tilde(self, t, a_t, a_t_1, constants = None):
        g, bg = self.get_constants(constants, a_t.size())
        return self.gamma_factor(t, a_t, a_t_1, (g, bg)) \
                    * self.Sigma_t(t-1, a_t_1, (g, bg))
    
    def anterior_mean(self, t, xt, pred_noise, Gamma, constants = None):
        g, bg = self.get_constants(constants, Gamma.size())
        factor = Gamma * (1 - bg[t])**(1/self.alpha)
        x = (1 / g[t]**(1 / self.alpha)) *(xt - factor* pred_noise)
        return x
    
    def reverse_factors(self, t, size):
        tmp = torch.ones(size)
        g, bg, t, a_t_1, a_t_prime, a_t = self._get_constants_to_incoming_batch_dims(tmp, t)
        Gamma = self.gamma_factor(t, a_t, a_t_1, (g, bg))
        factor_x = (1 / g[t]**(1 / self.alpha))
        factor_eps = factor_x * (Gamma * (1 - bg[t])**(1/self.alpha))
        factor_noise = torch.sqrt(self.sigma_tilde(t, a_t, a_t_1, (g, bg))) / torch.sqrt(a_t_1)
        # indeed we want the noise factor in front of the equivalent eps_t instead of z_t
        # hence the division by torch.sqrt(a_t_1)
        return factor_x, factor_eps, factor_noise
    
    def reverse_factors_lim(self, t, size):
        tmp = torch.ones(size)
        g, bg, t, a_t_1, a_t_prime, a_t = self._get_constants_to_incoming_batch_dims(tmp, t)
        Gamma = self.gamma_factor(t, a_t, a_t_1, (g, bg))
        factor_x = (1 / g[t]**(1 / self.alpha))
        factor_eps = self.alpha*(factor_x - 1) / (1 - bg[t])**(1 - 1 / self.alpha)
        factor_noise = (1 / g[t] - 1)**(1 / self.alpha)
        return factor_x, factor_eps, factor_noise 
    
    # lambda, and not lambda squared
    def compute_lambda(self, t, a_t, a_t_1, constants = None):
        g, bg = self.get_constants(constants, a_t.size())
        return self.gamma_factor(t, a_t, a_t_1, (g, bg)) \
                * (1 - bg[t])**(1/self.alpha) / (g[t]**(1 / self.alpha))
    
    # when Gamma is already computed
    def compute_Sigma_t_1_over_a_t(self, t, Gamma, constants = None):
        g, bg = self.get_constants(constants, Gamma.size())
        return (1 - Gamma) * ((1 - bg[t]) / g[t])**(2 / self.alpha)
    
    def _eps_from_previous_x(self, t, xt, previous_x, Gamma, constants = None):
        g, bg = self.get_constants(constants, xt.size())
        return (xt - g[t]**(1/self.alpha)*previous_x) / (Gamma * (1 - bg[t])**(1/self.alpha))
    
    def eps_from_model_output(self, 
                                t, 
                                xt, 
                                model_output, 
                                Gamma,
                                a_t,
                                constants = None):
        if self.model_mean_type == ModelMeanType.EPSILON:
            model_eps = model_output
        elif self.model_mean_type == ModelMeanType.START_X:
            model_eps = self._predict_eps_from_xstart(x_t=xt,
                                                    t=t,
                                                    xstart=model_output)
        elif self.model_mean_type == ModelMeanType.Z:
            model_eps = torch.sqrt(a_t) * model_output
        elif self.model_mean_type == ModelMeanType.SQRT_GAMMA_EPSILON:
            model_eps = model_output / (1 - torch.sqrt(1 - Gamma))
        elif self.model_mean_type == ModelMeanType.PREVIOUS_X:
            model_eps = self._eps_from_previous_x(t, xt, model_output, Gamma, constants)
        else:
            raise NotImplementedError(self.model_mean_type)
        return model_eps
    
    def anterior_mean_from_model_output(self, 
                                        t, 
                                        xt, 
                                        model_output, 
                                        Gamma,
                                        a_t,
                                        constants = None):
        g, bg = self.get_constants(constants, Gamma.size())

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            return model_output
        model_eps = self.eps_from_model_output(t, xt, model_output, Gamma, a_t, constants)
        return self.anterior_mean(t, xt, model_eps, Gamma, constants)



'''        if False:
            g, bg = self._get_schedule_to_match_last_dims(x_start)
            t = self._get_t_to_batch_size(x_start, t)
            losses = torch.tensor([]).to(self.device)
            z_t = torch.randn_like(x_start, device=self.device)
            for i in range(monte_carlo_steps):
                # generate a's 
                a_t_1, a_t_prime, a_t = self._get_noises_to_incoming_batch_dims(x_start, t)
                eps = torch.sqrt(a_t) * z_t 
                terms = self.training_losses_fixed_a(model, 
                                                x_start, 
                                                t, 
                                                g,
                                                bg,
                                                a_t_1, 
                                                a_t_prime, 
                                                a_t, 
                                                eps,
                                                lploss = lploss,
                                                model_kwargs=model_kwargs)
                losses = torch.concat((losses, terms['loss'].unsqueeze(0)))'''