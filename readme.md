Here's the README adapted for your new project *PDMP*, following the same structure and style as the one you provided:

---

# PDMP

PDMP (Piecewise Deterministic Markov Process) is a generative model that replaces traditional Gaussian noise-based diffusion with piecewise deterministic processes. This modification aims to enhance sample generation through structured exploration, particularly useful for high-dimensional and complex distributions.

This repository contains the full implementation of PDMP, providing the tools for training, evaluation, and generation of data using this model. It includes a modular structure, allowing users to customize different components like the model, logging mechanisms, and experiment setups.

For further details on the theoretical background and methodology, please refer to our preprint [here](https://arxiv.org/abs/2407.18609).

## Repository Overview

We are using [BEM (Better Experimentation Manager)](https://github.com/darioShar/bem) to manage our experiments.

- **Generative Model Implementation**: Located in `PDMP/methods/pdmp.py` and `PDMP/methods/Diffusion.py`, these files contain the core logic of the PDMP generative model, and the improved DDPM generative model (incorporating its heavy-tailed extension [DLPM](https://github.com/darioShar/DLPM)). Users interested in understanding or modifying the underlying generative processes should start here.
  
- **Neural Network Architecture**: If you want to change the neural networks used in the PDMP, head to the `PDMP/models` directory. This is where all the neural network models are defined and can be customized according to your needs.
  
- **Logging Configuration**: To customize how logging is handled, you can inherit from `bem/Logger.py` for integrating your own logger. An example of a custom logging setup is available in `pdmp/NeptuneLogger.py`.

- **Experiment Workflow**: The `PDMP/pdmp_experiment` file orchestrates the initialization of the training, evaluation, and data generation processes. To integrate your modifications into the experiment flow, update the `init` functions here. These functions will be provided to the `Experiment` class from the `bem` library.

- **Configuration Parameters**: Parameters for specific runs are passed in a dictionary called `p`, which is loaded from configuration files located in `pdmp/configs/`. Adjust these files to fine-tune the behavior of the model and experiment settings.

- **Comparison between diffusion and PDMP methods**: This repository supports both traditional diffusion models and PDMP, which allows for structured sampling using piecewise deterministic processes. When using the PDMP method, users can specify different samplers such as 'ZigZag', 'HMC', and 'BPS', and choose from various loss types. Our paper provides a detailed discussion on the advantages of PDMP, though users can experiment with both approaches here.

## Supported Datasets

Here’s a brief overview of the supported datasets, as provided by `BEM`, and how to specify them:

- **2D Datasets**: 
  - The repository supports synthetic 2D datasets. See `Generator.available_distributions` in `bem/datasets/Data.py`.

- **Image Datasets**: 
  - You can use standard image datasets (MNIST, CIFAR-10, its long-tailed version CIFAR-10-LT, CELEBA etc.). See `bem/datasets/__init__.py`.

Configuration files for some of these datasets are provided in the `pdmp/configs/` directory: `mnist.yml` for MNIST, `cifar10.yml` for CIFAR-10, `cifar10_lt.yml` for CIFAR-10-LT.
    
You can modify the configuration files to adjust data loading settings, such as the batch size or data augmentation options, according to your experiment needs.

## Using the Provided Scripts

This repository includes scripts that simplify the process of training, evaluating, and visualizing the results of PDMP. Below is a description of each script and how to use them:

### 1. `run.py`

This script is used to train a model. It accepts various command-line arguments to control the training process, including configuration settings and experiment parameters.

**Example Command**:
```bash
python ./run.py --config mnist --name pdmp_test --method pdmp --sampler ZigZag --loss hyvarinen --epochs 100 --eval 50 --check 50 --train_reverse_steps 1000
```

**Explanation**:
- `--config`: Specifies the configuration file to use (e.g., `mnist`).
- `--name`: The name of the experiment run, used for logging and identification. Here, the checkpointed models will be stored in `/models/pdmp_test/`.
- `--method`: Specifies the generative method to use (either `diffusion` or `pdmp`), in this case, `pdmp`.
- `--sampler`: Specifies the sampler to use with PDMP (options: 'ZigZag', 'HMC', 'BPS'). Required when `--method pdmp` is selected.
- `--loss`: Specifies the loss type for training (options: 'square', 'kl', 'logistic', 'hyvarinen', 'ml', 'hyvarinen_simple', 'kl_simple'). We recommend 'hyvarinen' for ZigZag and 'ml' for HMC and BPS.
- `--epochs`: The total number of training epochs.
- `--eval`: Specifies the interval (in epochs) for running evaluations during training.
- `--check`: Interval for model checkpointing (in epochs).
- `--train_reverse_steps`: The number of reverse steps to use during training.

### 2. `eval.py`

This script evaluates a pre-trained model and can also be used for generating samples from the trained model.

**Example Command**:
```bash
python ./eval.py --config mnist --name pdmp_test --method pdmp --sampler HMC --loss ml --epochs 100 --eval 100 --generate 2000 --reverse_steps 1000
```

**Explanation**:
- `--config`, `--name`, `--method`, `--sampler`, `--loss`, and `--epochs`: Same as in `run.py`.
- `--eval`: Specifies the evaluation checkpoint to use.
- `--generate`: Number of samples to generate.
- `--reverse_steps`: Number of reverse steps to use during the generation process.

### 3. `display.py`

This script is used to visualize the generated samples or the results from an experiment.

**Example Command**:
```bash
python ./display.py --config mnist --name pdmp_test --method pdmp --sampler ZigZag --loss hyvarinen --epochs 100 --reverse_steps 1000 --generate 1
```

**Explanation**:
- `--config`, `--name`, `--method`, `--sampler`, `--loss`, `--epochs`, and `--reverse_steps`: Same as in the previous scripts.
- `--generate`: Specifies the number of samples to visualize (e.g., `1` for displaying a single sample).

## Citation

```bash
@misc{bertazzi2024piecewisedeterministicgenerativemodels,
      title={Piecewise deterministic generative models}, 
      author={Andrea Bertazzi and Alain Oliviero-Durmus and Dario Shariatian and Umut Simsekli and Eric Moulines},
      year={2024},
      eprint={2407.19448},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/2407.19448}, 
}
```

## Contribute

We welcome issues, pull requests, and contributions. We will try our best to improve readability and answer questions.



<!-- 

# PDMP

This repository is the implementation of our Piecewise Deterministic Generative Model, as can be found [here](https://arxiv.org/abs/2407.19448)

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


## Evaluation
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


## Structure of the project

We will point at the most important parts of the codebase.

### Models
* The MLP model used by ZigZag and diffusion for 2D data is located in `./PDMP/models/Model.py` and corresponds to the `MLPModel` object
* The U-Net model used by ZigZag and diffusion for image data is located in `./PDMP/models/unet.py` and corresponds to the `UNetModel` object
* The Normalizing Flow models used by HMC and BPS is located in `./PDMP/models/NormalizingFlow.py` and corresponds to the `NormalizingFlowModel` object. They can be used in conjunction with a VAE, but this has not worked out very well (for the moment the vae implementation clutters the code a bit, this will be dealt with later on.)

### PDMP computations
* Look at the `./PDMP/methods/pdmp.py` file, where we define the forward, the training loss, and the backward sampling.
* Look at `get_densities_from_zigzag_output` to see how we retrieve the densities from ZigZag output. ZigZag's output is of format (B, 2*C, ...) where B is the batch size and C the number of channels of the data ($C=1$ for 2D data). The first C channels correspond to velocity=-1, the second to velocity=1. 
 -->
