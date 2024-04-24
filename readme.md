# PDMP

## What can be done

* train and evaluate PDMP models, diffusion models for data generation
* datasets: simple 2d data (gmm_grid, alpha_stable_grid etc.), image datasets (mnist, cifar10, celeba)

## Structure of the project

Todo

## How to run

Open and modify ***./PDMP/config/2d_data.yml*** to configurate the run (on 2d datasets). You will find all relevant adjustable parameters. To effectively launch an experiment, launch the `./run_PDMP.py` script:

`python ./run_PDMP.py --config 2d_data --name {pdmp_experiment_name}`

this will save the results in ***./models/{pdmp_experiment_name}***. In particular, the **run** subparameters in `2d_data.yml` control the following:
* ***epochs***: number of epochs
* ***eval_freq***: number of epochs between each evaluation
* ***checkpoint_freq***: number of epochs between each model checkpoint

## Additional possibilities for automating experiments

One can pass arguments to the command line to change most of the configuration file parameters. For instance:

`python ./run_PDMP.py --config 2d_data --name 2d_pdmp --epochs 100 --eval 25 --check 25 --reverse_steps 100 --noising_process pdmp --sampler HMC`

loads `2d_data` config file and its parameters, runs for 100 epochs, checkpoints and evaluates every 25 epochs, uses pdmp as the noising process, uses 100 reverse steps, uses the HMC ssampler;

`python ./run_PDMP.py --config 2d_data --name 2d_pdmp --epochs 100 --eval 25 --check 25 --reverse_steps 100 --noising_process diffusion`

does the same but uses diffusion.


The `eval_pdmp.py` script is used to evaluate models that are already trained and stored thanks to the previous script. For instance:

`python ./eval_PDMP.py --config 2d_data --name 2d_pdmp --epochs 100 --eval 25 --reverse_steps 100 --noising_process diffusion`

will load the experiment that we ran just before, evaluate and store each model checkpointed at epochs multiples of 25. You could thus only evaluate the last model with `--eval 100`.



## Load and display results

Notebook to come. Needs small adaptations from my diffusion project.