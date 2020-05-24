#!/bin/bash

script_path=$(dirname "$0")
cd "$script_path"/.. || exit
for test_config in evaluation/configs/"$1"-test-*.yml; do
  python experiments/run_models -c "$test_config"
done
