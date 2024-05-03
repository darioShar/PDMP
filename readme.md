# PDMP

## What can be done

* train and evaluate PDMP/diffusion models for data generation
* datasets: simple 2d data (gmm_grid, alpha_stable_grid etc.), image datasets (mnist, cifar10, celeba)

## Structure of the project

Todo

## How to run

Open and modify ***./PDMP/config/2d_data.yml*** to configurate the run (on 2d datasets). You will find all relevant adjustable parameters. To launch an experiment, run the `./run_PDMP.py` script:

> python ./run_PDMP.py --config 2d_data --name {pdmp_experiment_name}

this will save the results in ***./models/{pdmp_experiment_name}***. In particular, the **run** subparameters in `2d_data.yml` control the following:
* ***epochs***: number of epochs
* ***eval_freq***: number of epochs between each evaluation
* ***checkpoint_freq***: number of epochs between each model checkpoint

## Additional possibilities for automating experiments

One can pass arguments to the command line to change most of the configuration file's parameters. For instance:

> python ./run_pdmp.py --config 2d_data --name tmp --noising_process pdmp --sampler BPS --reverse_steps 50 --refresh_rate 1. --epochs 100 --eval 20 --scheme splitting --square_loss --logistic_loss --kl_loss

loads `2d_data` config file and its parameters, saves the runs in `./models/tmp`, uses pdmp with BPS sampler, 50 reverse steps, a refresh rate of 1. runs for 100 epochs, checkpoints and evaluates every 25 epochs, uses the splitting scheme as the backward scheme and adds the square loss, logistic loss and KL loss to the training. 

To use diffusion for instance:
> python ./run_PDMP.py --config 2d_data --name 2d_pdmp --epochs 100 --eval 25 --check 25 --reverse_steps 100 --noising_process diffusion

The `eval_pdmp.py` script is used to evaluate models that are already trained and checkpointed thanks to the previous script. For instance:

`python ./eval_PDMP.py --config 2d_data --name 2d_pdmp --epochs 100 --eval 25 --reverse_steps 100 --noising_process diffusion`

will load the parameters we used previously, evaluate and store each checkpointed models at epochs multiples of 25. Pass `eval` equal to `epochs` to only evaluate a single checkpointed model.


## Load and display results

See `review_experiments` notebook.