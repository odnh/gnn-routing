import argparse
import importlib

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run experiments')
    parser.add_argument('filename', help='name of experiment to run')
    args = parser.parse_args()
    i = importlib.import_module('experiments.'+args.filename)
    i.run_experiment()
