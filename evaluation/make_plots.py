"""Helpful functions for reading results and plotting them"""

import matplotlib
import numpy as np
import pandas as pd
from jsonlines import jsonlines

matplotlib.use('pgf')  # to play nice with custom lualatex stuff
import seaborn as sns
import matplotlib.pyplot as plt


def demo_plot():
    """Demo of how to do a plot. Not related to project at all."""
    t = np.linspace(0.0, 1.0, 100)
    s = np.cos(4 * np.pi * t) + 2

    fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)
    ax.plot(t, s)

    ax.set_xlabel(r'\textbf{time (s)}')
    ax.set_ylabel('\\textit{Velocity (\N{DEGREE SIGN}/sec)}', fontsize=16)
    ax.set_title(r'\TeX\ is Number $\displaystyle\sum_{n=1}^\infty'
                 r'\frac{-e^{i\pi}}{2^n}$!', fontsize=16, color='r')
    plt.savefig('figure.pgf')


def read_results(path: str) -> pd.DataFrame:
    """
    Reads in a results file, returns a dataframe including ratios
    """
    df = pd.DataFrame(columns=['policy', 'utilisation', 'opt_utilisation',
                               'oblivious_utilisation', 'sequence_type', 'graph'])
    with jsonlines.open(path) as f:
        for result in f:
            policy = result['policy']
            sequence_type = result['sequence_type']
            graph = result['graph']
            for i in range(len(result['utilisations'])):
                df.append(
                    {'policy': policy, 'utilisation': result['utilisations'][i],
                     'opt_utilisation': result['opt_utilisations'][i],
                     'oblivious_utilisation': result['oblivious_utilisations'][i],
                     'sequence_type': sequence_type,
                     'graph': graph
                })

    # calculate ratios
    df1 = df.copy()
    df1['ratio'] = df1['utilisation'] / df1['opt_utilisation']
    df1['strategy'] = 'agent'
    df2 = df.copy()
    df2['ratio'] = df1['oblivious_utilisation'] / df1['opt_utilisation']
    df2['strategy'] = 'oblivious'

    return pd.concat(df1, df2)


def plot_exp1():
    """Generate plots for exp1"""
    # labels = ['MLP', 'LSTM', 'GNN', 'Iterative']
    files = ['results/1.{}-{}'.format(exp, pol) for exp in [1, 2] for pol in ['mlp', 'lstm', 'iter']]
    results = [read_results(file) for file in files]
    df = pd.concat(results)

    plt.clf()
    sns.boxplot(y='ratio', x='policy', data=df, palette="colorblind", hue='strategy')
    plt.savefig('plots/exp1.pgf')


def plot_exp2():
    """Generate plots for exp2"""
    # labels = ['MLP', 'LSTM', 'GNN', 'Iterative']
    files = ['results/2.{}-{}'.format(exp, pol) for exp in [1, 2, 3, 4] for pol in ['mlp', 'lstm', 'iter']]
    results = [read_results(file) for file in files]
    df = pd.concat(results)

    # TODO: change the hue field to get more separation
    plt.clf()
    sns.boxplot(y='ratio', x='policy', data=df, palette="colorblind", hue='strategy')
    plt.savefig('plots/exp2.pgf')


def plot_exp3():
    """Generate plots for exp1"""
    # labels = ['MLP', 'LSTM', 'GNN', 'Iterative']
    files = ['results/2.{}-{}'.format(exp, pol) for exp in [1, 2, 3, 4] for pol
             in ['mlp', 'lstm', 'iter']]
    results = [read_results(file) for file in files]
    df = pd.concat(results)

    plt.clf()
    sns.boxplot(y='ratio', x='policy', data=df, palette="colorblind",
                       hue='strategy')
    plt.savefig('plots/exp1.pgf')


def plot_exp4():
    """Generate plots for exp1"""
    # labels = ['MLP', 'LSTM', 'GNN', 'Iterative']
    files = ['results/2.{}-{}'.format(exp, pol) for exp in [1, 2, 3, 4] for pol
             in ['mlp', 'lstm', 'iter']]
    results = [read_results(file) for file in files]
    df = pd.concat(results)


    plt.clf()
    sns.boxplot(y='ratio', x='policy', data=df, palette="colorblind",
                       hue='strategy')
    plt.savefig('plots/exp1.pgf')


if __name__ == '__main__':
    plot_exp1()
    plot_exp2()
    plot_exp3()
    plot_exp4()
