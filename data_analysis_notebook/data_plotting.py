import numpy as np

import seaborn as sns; sns.set()
import matplotlib as mpl
import matplotlib.pyplot as plt

# plt.rcParams['figure.figsize'] = [15, 10]
# plt.rcParams['figure.figsize'] = [15, 15]
plt.rcParams['figure.figsize'] = [35, 20]




font = {'size'   : 30, }
axes = {'labelsize': 'large', 'titlesize': 'large'}

mpl.rc('font', **font)
mpl.rc('axes', **axes)
sns.set_context("poster", font_scale=1.75)
sns.set_style("white")

import data_utils
# import matplotlib.style as style

def plot_acceptance_rate_distribution(model_space_report_df, output_path, hide_x_ticks=True, show_mean=True):

    # Make plot!
    fig, ax = plt.subplots()

    ax.errorbar(model_space_report_df.index, 
                model_space_report_df['acceptance_ratio'], 
                yerr=model_space_report_df['stdev'], fmt=',', color='black', alpha=0.5,
                label=None, elinewidth=0.5)

    sns.barplot(model_space_report_df.index, model_space_report_df.acceptance_ratio, 
                     data=model_space_report_df, alpha=1, ax=ax, color=None)

    mean = np.mean(model_space_report_df['acceptance_ratio'].values)


    if hide_x_ticks:
        ax.set(xticklabels=[])
        ax.set(xlabel='')
        ax.legend().remove()    
    
    else:
        ax.set(xticklabels=model_space_report_df['model_idx'])
        ax.set(xlabel='Model')
        ax.legend()

    if show_mean:
        ax.axhline(mean, ls='--', label='Mean', linewidth=1.0)
        ax.legend()



    ax.set(ylabel='Model posterior probability')
    ax.set(xlim=(-0.5,None))
    ax.set(ylim=(-0))
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_alpha(0.5)
    ax.spines["bottom"].set_alpha(0.5)

    plt.savefig(output_path, dpi=500)

    print("\n")

def plot_acceptance_probability_distribution(model_space_report_df, output_path, hide_x_ticks=True):
    # Make plot!
    fig, ax = plt.subplots()

    ax.errorbar(model_space_report_df.index,
                model_space_report_df['acceptance_probability'],
                yerr=model_space_report_df['stdev'], fmt=',', color='black', alpha=0.5,
                label=None, elinewidth=0.5)

    sns.barplot(model_space_report_df.index, model_space_report_df['acceptance_probability'],
                     data=model_space_report_df, alpha=1, ax=ax)

    mean = np.mean(model_space_report_df['acceptance_probability'].values)
    ax.axhline(mean, ls='--', label='Mean', linewidth=1.0)

    if hide_x_ticks:
        ax.set(xticklabels=[])
        ax.set(xlabel='')

    else:
        ax.set(xticklabels=model_space_report_df['model_idx'])
        ax.set(xlabel='Model')

    ax.set(ylabel='Acceptance probability')
    ax.set(xlabel='')
    ax.set(xlim=(-1,None))
    ax.set(ylim=(-0))
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.legend()
    ax.spines["left"].set_alpha(0.5)
    ax.spines["bottom"].set_alpha(0.5)

    plt.savefig(output_path, dpi=500)