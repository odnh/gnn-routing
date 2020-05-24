# Evauluation

In this folder resides the code for performing the evaluation. This includes
training, running experiments on models, and graphing results.

## Layout

- `configs`: contains all the config files for training and experiments
- `models`: location to store trained models
- `plots`: plot output location from graphing scrips
- `results`: results outputs from running experiments

## Config explanation

We use hierarchical yaml files (each can specify a list of parents which they
partially override). `train-a.yml` is a top-level training specification.
`train-a.b.yml` is that training tweaked for a speicifc policy. `test-a._.c.yml`
is settings for a test to be run on all models `b` trained from `train-a.b.yml`
configs. The json files in this folder are for hyperparameter configuration.

## Results

Results are stroed in a file format where each line is a json object. The json
object is simply the configuration of the run and 3 extra values: a list of the
optimal congestion, oblivous congestion, and agent congestion per dm in the
sequence.