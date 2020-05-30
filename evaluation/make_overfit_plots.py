"""Plot results from the overfit experiments"""
from typing import Iterable

import matplotlib
import pandas as pd
import numpy as np
from jsonlines import jsonlines

matplotlib.use('pgf')  # to play nice with custom lualatex stuff
import seaborn as sns
import matplotlib.pyplot as plt


def read_result(model_id: str, test_id: str, policy_id: str) -> pd.DataFrame:
    """
    Reads in a results file, returns a dataframe including ratios
    """
    path = "results/overfit-{}-{}-{}".format(model_id, test_id, policy_id)

    df = pd.DataFrame(
        columns=['utilisation', 'opt_utilisation', 'oblivious_utilisation'])
    with jsonlines.open(path) as f:
        for result in f:
            for i in range(len(result['utilisations'])):
                df = df.append(
                    {'utilisation': result['utilisations'][i],
                     'opt_utilisation': result['opt_utilisations'][i],
                     'oblivious_utilisation': result['oblivious_utilisations'][
                         i],
                     'action': result['actions'][i]
                     }, ignore_index=True)

    # to easily separate experiments in plots
    if len(model_id) > 1:
        x_value = int(model_id[2])
    else:
        x_value = int(model_id)
    df['model_id'] = model_id
    df['x_value'] = x_value
    df['test_id'] = test_id
    df['policy_id'] = policy_id
    # calculate ratios
    df['ratio'] = df['utilisation'] / df['opt_utilisation']
    df['oblivious_ratio'] = df['oblivious_utilisation'] / df['opt_utilisation']

    return df


def plot():
    """Make the plot"""
    specs = [
        ('1', ['1', 'out'], 'mlp'),
        ('1_2', ['1', '2', 'out'], 'mlp'),
        ('1_3', ['1', '2', '3', 'out'], 'mlp'),
        ('1_4', ['1', '2', '3', '4', 'out'], 'mlp'),
        ('1_5', ['1', '2', '3', '4', '5', 'out'], 'mlp'),
        ('2', ['2'], 'mlp'),
        ('3', ['3'], 'mlp'),
        ('4', ['4'], 'mlp'),
        ('5', ['5'], 'mlp')]
    results = []
    for model, tests, policy in specs:
        for test in tests:
            results.append(read_result(model, test, policy))
    df = pd.concat(results)

    plt.clf()
    palette = sns.color_palette('colorblind')

    # ratio perf:
    singles_df = df.loc[df['model_id'].isin(['1', '2', '3', '4', '5'])]
    singles_df = singles_df.loc[singles_df['test_id'] != 'out']
    singles_df['single_ratio'] = singles_df['ratio']
    singles_df = singles_df[['single_ratio', 'test_id']]
    ratio_df = df.loc[df['test_id'] != 'out']
    ratio_df = ratio_df.loc[ratio_df['model_id'].isin(['1', '1_2', '1_3', '1_4', '1_5'])]
    ratio_df = ratio_df.set_index('test_id').join(singles_df.set_index('test_id'))
    ratio_df['rescaled_ratio'] = ratio_df['ratio'] / ratio_df['single_ratio']
    ratio_df = ratio_df.loc[ratio_df['x_value'] > 1]
    ax = sns.lineplot(x='x_value', y='rescaled_ratio', data=ratio_df, color=palette[0], marker="o", label='Test matrices')

    # out:
    out_df = df.loc[df['test_id'] == 'out']
    out_df = out_df.loc[out_df['x_value'] > 1]
    sns.lineplot(x='x_value', y='ratio', data=out_df, ax=ax, color=palette[1], marker="o", label='Out of distribution')

    # action max difference:
    ax2 = plt.twinx()
    action_df = df.loc[df['model_id'].isin(['1', '1_2', '1_3', '1_4', '1_5'])]
    action_df = action_df.loc[action_df['test_id'] != 'out']
    action_df = action_df.groupby('model_id')
    action_df = action_df.agg({'action': maxdiff, 'x_value': max})
    action_df = action_df.loc[action_df['x_value'] > 1]
    sns.lineplot(x='x_value', y='action', data=action_df, ax=ax2, color=palette[2], marker="o", label='Action difference')

    # cosmetic fixes
    plt.xticks([2, 3, 4, 5])
    ax.set_xlabel("Number of DMs in training set")
    ax2.set_xlabel("Number of DMs in training set")
    ax.set_ylabel("Ratio between optimal and achieved max-link-utilisation")
    ax2.set_ylabel("Maximum difference between actions")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc=2)
    ax2.legend().remove()

    plt.savefig('plots/overfit.pgf')


def maxdiff(groups: Iterable) -> float:
    """
    Finds the maximum difference between values in a single location across a
    set of arrays.
    Args:
        groups: Iterable of a list of lists of lists of floats

    Returns:
        Single float value max difference
    """
    grouped = [actions[0] for actions in list(groups)]
    actions = np.array(grouped)
    min_vals = np.min(actions, axis=0)
    max_vals = np.max(actions, axis=0)
    diff = max_vals - min_vals
    max_diff = np.max(diff)
    return max_diff


if __name__ == '__main__':
    plot()
