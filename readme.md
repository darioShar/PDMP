# PDMP

## How to run

Open and modify ***./PDMP/config/2d_data.yml*** to configurate the run (on 2d datasets). You will find all relevant adjustable parameters. To launch an experiment, run the `./run_PDMP.py` script:

> python ./run_PDMP.py --config 2d_data --name {pdmp_experiment_name}

this will load the parameters from `./PDMP/config/2d_data.yml`, do a run, and save the results in ***./models/{pdmp_experiment_name}***. In particular, the **run** subparameters in `2d_data.yml` control the following:
* ***epochs***: number of epochs
* ***eval_freq***: number of epochs between each evaluation
* ***checkpoint_freq***: number of epochs between each model checkpoint

## Modify some parameters with command line arguments

One can pass arguments to the command line, to overwrite the configuration file's parameters for a specific run. For instance:

> python ./run_pdmp.py --config 2d_data --name tmp --noising_process pdmp --sampler BPS --reverse_steps 50 --refresh_rate 1. --epochs 100 --eval 20 --scheme splitting --loss square --logistic --kl

loads `2d_data` config file and its parameters, saves the run in `./models/tmp`, and overwrites parameters so that we: use pdmp with BPS sampler, 50 reverse steps, a refresh rate of 1., train for 100 epochs, checkpoint and evaluate every 25 epochs, use the splitting scheme as the backward scheme, and use the square loss, logistic loss and KL loss for training (we just add these losses). 

To rather use diffusion, one can run
> python ./run_PDMP.py --config 2d_data --name 2d_pdmp --epochs 100 --eval 25 --check 25 --reverse_steps 100 --noising_process diffusion


## How to evaluate 
The `eval_pdmp.py` script is used to evaluate models that are already trained and checkpointed thanks to the previous script. Let's look at the following command:

> python ./eval_PDMP.py --config 2d_data --name 2d_pdmp --epochs 100 --eval 25 --reverse_steps 100 --noising_process pdmp --sampler ZigZag --loss hyvarinen

This will look into the `./models/2d_pdmp` folder to find models trained with the parameters specified in `./PDMP/config/2d_data.yml` (parameters specified in the command line overwrite those specified in the config file). The script will then load corresponding models checkpointed at epochs 25, 50, ..., 100, and evaluate each of them . 

Thus pass `eval` == `epochs` in order to evaluate a single model, checkpointed at epoch `epochs`. 

## Display plots
The `display_pdmp.py` script can be used to display a plot and an animation of some generated data. It will load a model in the same fashion as the `eval_pdmp.py` script, but will always load the latest model saved with the specified parameters.
Example:

> python ./display_pdmp.py --config 2d_data --name 2d_pdmp --sampler BPS --epochs 20 --eval 20 --loss hyvarinen ml logistic --noising_process pdmp  --reverse_steps 50

Please modify lines 45 to 59 to obtain plots as desired:

```
# display plot and animation, for a specific model
anim = exp.manager.display_plots(ema_mu=None, # can specify ema rate, if such a model has been trained
                            plot_original_data=False, 
                            title=title,
                            nb_datapoints=25000, # number of points to display.
                            marker='.', # '.' marker displays pixel-wide points.
                            color='blue', # color of the points
                            xlim = (-.5, 1.5), # x-axis limits
                            ylim = (-.5, 1.5), # y-axis limits
                            alpha = 1.0,
                            )
# save animation
path = os.path.join(SAVE_ANIMATION_PATH, '_'.join([str(x) for x in run_info]))
anim.save(path + '.mp4')
print('Animation saved in {}'.format(path))
```

## Load and display plots, evaluation metrics, further results (advanced)

See `review_experiments` notebook.


## Structure of the project

We plan to draw a figure representing the whole program flow. It is a bit complicated for the moment so we will just point at the most important parts of the code

### Models
* The MLP model used by ZigZag and diffusion for 2D data is located in `./PDMP/models/Model.py` and corresponds to the `MLPModel` object
* The U-Net model used by ZigZag and diffusion for image data is located in `./PDMP/models/unet.py` and corresponds to the `UNetModel` object
* The Normalizing Flow models used by HMC and BPS is located in `./PDMP/models/NormalizingFlow.py` and corresponds to the `NormalizingFlowModel` object. They can be used in conjunction with a VAE, but this has not worked out very well (for the moment the vae implementation clutters the code a bit, this will be dealt with later on.)

### PDMP computations
* Look at the `./PDMP/compute/pdmp.py` file, where we define the forward, the training loss, and the backward sampling.
* Look at `get_densities_from_zigzag_output` to see how we retrieve the densities from ZigZag output. ZigZag's output is of format (B, 2*C, ...) where B is the batch size and C the number of channels of the data ($C=1$ for 2D data). The first C channels correspond to velocity=-1, the second to velocity=1. 


### Todo...
generation, evaluation, training manager and ema models, experiment design, overall codeflow