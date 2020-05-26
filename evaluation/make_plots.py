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


def read_results(test_number: str, policy_name: str) -> pd.DataFrame:
    """
    Reads in a results file, returns a dataframe including ratios
    """
    path = 'results/{}-{}'.format(test_number, policy_name)

    df = pd.DataFrame(
        columns=['utilisation', 'opt_utilisation', 'oblivious_utilisation',
                 'sequence_type', 'graph'])
    with jsonlines.open(path) as f:
        for result in f:
            sequence_type = result['sequence_type']
            graph = result['graph']
            for i in range(len(result['utilisations'])):
                df.append(
                    {'utilisation': result['utilisations'][i],
                     'opt_utilisation': result['opt_utilisations'][i],
                     'oblivious_utilisation': result['oblivious_utilisations'][
                         i],
                     'sequence_type': sequence_type,
                     'graph': graph
                     })

    # to easily separate experiments in plots
    df['test_number'] = test_number
    df['policy_name'] = policy_name
    # calculate ratios
    df['ratio'] = df['utilisation'] / df['opt_utilisation']
    df['oblivious_ratio'] = df['oblivious_utilisation'] / df['opt_utilisation']

    return df


def plot_exp1():
    """Generate plots for exp1"""
    tests = [(test_number, policy_name) for test_number in ['1.1', '1.2'] for
             policy_name in ['mlp', 'lstm', 'gnn', 'iter']]
    results = [read_results(*test) for test in tests]
    df = pd.concat(results)

    plt.clf()
    sns.boxplot(y='ratio', x='policy_name', data=df, palette='colorblind',
                hue='test_number')
    sns.lineplot(y='oblivious_ratio', x='policy_name', data=df,
                 palette='colorblind', hue='test_number', markers=False)
    plt.savefig('plots/exp1.pgf')


def plot_exp2():
    """Generate plots for exp2"""
    tests = [(test_number, policy_name) for test_number in
             ['2.1', '2.2', '2.3', '2.4'] for
             policy_name in ['mlp', 'lstm', 'gnn', 'iter']]
    results = [read_results(*test) for test in tests]
    df = pd.concat(results)

    plt.clf()
    sns.boxplot(y='ratio', x='policy_name', data=df, palette='colorblind',
                hue='test_number')
    sns.lineplot(y='oblivious_ratio', x='policy_name', data=df,
                 palette='colorblind', hue='test_number', markers=False)
    plt.savefig('plots/exp2.pgf')


def plot_exp3():
    """Generate plots for exp3"""
    tests = [(test_number, policy_name) for test_number in
             ['3.1', '3.2', '3.3', '3.4', '3.5', '3.6'] for
             policy_name in ['gnn', 'iter']]
    results = [read_results(*test) for test in tests]
    df = pd.concat(results)

    plt.clf()
    sns.boxplot(y='ratio', x='policy_name', data=df, palette='colorblind',
                hue='test_number')
    sns.lineplot(y='oblivious_ratio', x='policy_name', data=df,
                 palette='colorblind', hue='test_number', markers=False)
    plt.savefig('plots/exp3.pgf')


def plot_exp4():
    """Generate plots for exp4"""
    tests = [(test_number, policy_name) for test_number in ['4.1'] for
             policy_name in ['mlp', 'lstm', 'gnn', 'iter']]
    results = [read_results(*test) for test in tests]
    df = pd.concat(results)

    plt.clf()
    sns.boxplot(y='ratio', x='policy_name', data=df, palette='colorblind',
                hue='test_number')
    sns.lineplot(y='oblivious_ratio', x='policy_name', data=df,
                 palette='colorblind', hue='test_number', markers=False)
    plt.savefig('plots/exp4.pgf')


if __name__ == '__main__':
    plot_exp1()
    plot_exp2()
    plot_exp3()
    plot_exp4()
