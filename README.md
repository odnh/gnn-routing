# Part III Project: examining RL in data driven routing with GNNs

The aim is to extend work done in "Learning to route with deep RL" to use GNNs,
hopefully allowing generalisation to unseen graphs (taken from the same
distribution).

### Layout:

- gym-ddr: Implementation of gym env dor data driven routing
- stable-baselines-extensions: modifications to stable-baselines for custom
                               learning.
- experiments: scripts to run learning experiments
- data: contains data to use in experiments (e.g. graphs, demands)
- results: will hopefully contain results at some point

### Requirements:

Relies on stable-baselines. A vagrant box can be set up using the Vagrantfile.
To set up on a linux machine, look at the `bootstrap.sh` file.

To setup python env, `env.yml` can be used with `conda env -f env.yml`
