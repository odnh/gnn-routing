#!/bin/bash
apt-get -y update
apt-get -y upgrade
apt-get install -y cmake libopenmpi-dev python3-dev zlib1g-dev libsm6 libxext6 libxrender-dev

# install conda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p /vagrant/miniconda
eval "$(/vagrant/miniconda/bin/conda shell.bash hook)"

# set up conda env
conda create --name baselines python=3.7
conda activate baselines
pip install -U pip
pip install tensorflow==1.14 stable-baselines dgl

# set up gym env
pip install -e gym-ddr
