#!/bin/bash

script_path=$(dirname "$0")
cd "$script_path"/.. || exit
python experiments/run_models -c evaluation/configs/train-"$1".yml
