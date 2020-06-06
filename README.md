# Part III Project: "Generalisable data-driven routing using RL with GNNs"

This is the code accompanying my dissertation submitted for my masters degree
(which can be found at https://github.com/odnh/gnn-routing-diss).

The aim is to extend work done in "Learning to route with deep RL" to use GNNs,
hopefully allowing generalisation to unseen graphs (read the dissertation for
further details and a more in depth explanation).

### Layout:

- gym-ddr: Implementation of openai gym env for RL routing
- stable-baselines-extensions: modifications to stable-baselines for custom
                               learning policies using GNNs.
- dd-learning-helpers: library of helper functions that don't rely on gym or
                       stable-baselines
- experiments: scripts to learn and test models
- data: contains data to use in experiments (e.g. graphs, demands)
- evaluation: contains scripts that combined with a configuration will run the
              experiments for project evaluation and plot the results.
- raeke: OCaml code for generating an oblivious routing. Has issues so was not
         used.

### Requirements:

Based on stable-baselines and using Graph Nets for GNN bits and pieces. 
To set up on a Linux (Ubuntu) machine from scratch, run the `bootstrap.sh` file
(Of course, read what the script does before running it).

To setup python env, `env.yml` can be used with `conda env -f env.yml`
