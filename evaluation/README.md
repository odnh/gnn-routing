# Evaluation

In this folder resides the code for performing the evaluation. This includes
training, running experiments on models, and graphing results.

## Layout

- `configs`: contains all the hyperparameter configs
- `models`: location to store trained models
- `plots`: plot output location from graphing scrips
- `results`: results outputs from running experiments

## Script explanation

- `experiments.py`: Runs all the training and evaluating. Example configuration
                    is in code at the top of the file.
- `make_plots.py`: plots the results obtained (to pgf format)
- `overfit_experiments.py`: Further experiments to test overfitting
- `make_overfit_plots.py`: plots the results obtained (to pgf format)


## Results

Results are stored in a file format where each line is a json object. The json
object is simply the configuration of the run and 3 extra values: a list of the
optimal congestion, oblivious congestion, and agent congestion per dm in the
sequence.