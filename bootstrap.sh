#!/bin/bash
sudo apt-get -y update
sudo apt-get -y upgrade
sudo apt-get install -y cmake libopenmpi-dev python3-dev zlib1g-dev libsm6 libxext6 libxrender-dev unzip

# install conda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p ~/miniconda
eval "$(~/miniconda/bin/conda shell.bash hook)"

# set up conda env
conda env create -f env.yml
conda activate ddr

# set up gym env
pip install -e gym-ddr
pip install -e ddr-learning-helpers
pip install -e stable-baselines-ddr

# get data
cd data
bash populate.sh
