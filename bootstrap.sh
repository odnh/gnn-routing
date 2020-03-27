#!/bin/bash
sudo apt-get update
sudo apt-get upgrade

# install conda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p $HOME/miniconda
source ~/miniconda/bin/activate

# set up conda env
conda create --name baselines --python=3.7
conda activate baselines
pip install -U pip
pip install tensorflow==1.14 stable-baselines dgl

# set up gym env
pip install -e gym-ddr
