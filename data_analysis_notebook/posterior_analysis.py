import numpy as np
import seaborn as sns; sns.set()
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import data_utils
import glob
import os
import matplotlib.style as style
import math
import sklearn
from sklearn.preprocessing import MinMaxScaler

plt.rcParams['figure.figsize'] = [20, 15]

font = {'size'   : 11, }
axes = {'labelsize': 'small', 'titlesize': 'small'}

mpl.rc('font', **font)
mpl.rc('axes', **axes)



def generate_file_paths(posterior_dir, priors_dir):
    prior_param_paths = [file_path for file_path in glob.iglob(priors_dir + "*.csv")]
    posterior_param_paths = [file_path for file_path in glob.iglob(posterior_dir + "*_all_params")]
    num_models = len(posterior_param_paths)

    ordered_prior_species_paths = [0 for f in range(num_models)]
    ordered_prior_param_paths = [0 for f in range(num_models)]
    ordered_posterior_paths = [0 for f in range(num_models)]

    # Split prior paths into species and parameter priors in model number order
    for f in prior_param_paths:
        file_name = os.path.basename(f)
        split_name = file_name.split('_')
        model_num = int(''.join(list(filter(str.isdigit, split_name[1]))))

        if split_name[0] == "species":
            ordered_prior_species_paths[model_num] = f

        if split_name[0] == "params":
            ordered_prior_param_paths[model_num] = f

    for f in posterior_param_paths:
        file_name = os.path.basename(f)
        split_name = file_name.split('_')
        model_num = int(split_name[1])
        ordered_posterior_paths[model_num] = f

    return ordered_posterior_paths, ordered_prior_param_paths, ordered_prior_species_paths


def visualise_posterior_distributions(posterior_dir, priors_dir, output_dir):
    # Plot settings
    style.use('seaborn-muted')
    sns.set_style("white")
    sns.set_context("talk")
    sns.despine(top=True)
    show_hist = False
    show_kde = True
    nbins = 30

    # Get ordered paths
    ordered_posterior_paths, \
    ordered_prior_param_paths, \
    ordered_prior_species_paths = generate_file_paths(posterior_dir, priors_dir)

    for model_ref, f in enumerate(ordered_posterior_paths):
        print(model_ref)
        model_posterior_df = pd.read_csv(f, sep=',')
        model_posterior_df, free_params = data_utils.normalise_parameters(model_posterior_df)

        accepted_sims = model_posterior_df.loc[model_posterior_df['Accepted'] == True]

        if len(accepted_sims) <= 1:
            continue

        # Generate subplotgrid
        ncols = 5
        nrows = math.ceil(len(free_params)/ncols)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
        axes = axes.reshape(-1)


        # Calculate kolmogorov_smirnov 95% critical value
        D_crit = 1.36 * math.sqrt(1 / len(accepted_sims) + 1 / len(model_posterior_df))

        # Iterate free parameters
        i = 0
        for param in free_params:
            D_n = data_utils.kolmogorov_smirnov_test(accepted_sims[param].values, model_posterior_df[param].values)

            if D_n > D_crit:
                text_color = 'red'

            else:
                text_color = 'black'

            sns.distplot(model_posterior_df[param], bins=nbins, hist=show_hist, kde=show_kde,
                         label="Sample distribution", ax=axes[i], norm_hist=True)
            sns.distplot(accepted_sims[param], bins=nbins, hist=show_hist, kde=show_kde,
                         label="Posterior distribution", ax=axes[i], norm_hist=True)

            D_n = round(D_n, 4)
            axes[i].text(1, 1, r"$D_n = " + str(D_n) + "$", fontsize='small',
                 horizontalalignment='center',
                 verticalalignment='center',
                 transform=axes[i].transAxes, color=text_color)

            # axes[i].set_title(r"$D_n = " + str(D_n) + "$")

            i += 1

        # plotting aesthetics
        for ax in axes:
            # ax.set(yscale='log')
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            # ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
            legend = ax.legend()
            legend.remove()

        # Add legend in place of last subplot
        handles, labels = axes[0].get_legend_handles_labels()

        D_crit = round(D_crit, 4)
        axes[-1].text(0.6, 0.5, r"$D_{crit} = " + str(D_crit) + "$",
                     horizontalalignment='center',
                     verticalalignment='center',
                     transform=axes[i].transAxes)

        axes[-1].set_axis_off()
        axes[-1].legend(handles, labels, loc='lower right')

        figure_name = output_dir + "distribution_model_REF_.pdf"
        figure_name = figure_name.replace('REF', str(model_ref))

        plt.tight_layout()
        plt.savefig(figure_name, dpi=100)
        plt.close(fig)

def generate_posterior_KS_csv(posterior_dir, priors_dir, output_dir):
    ordered_posterior_paths, \
    ordered_prior_param_paths, \
    ordered_prior_species_paths = generate_file_paths(posterior_dir, priors_dir)

    out_path = output_dir + "model_NUM_KS.csv"

    for model_idx, f in enumerate(ordered_posterior_paths):
        print(model_idx)
        model_posterior_df = pd.read_csv(f, sep=',')
        KS_df = data_utils.make_KS_df(model_idx, model_posterior_df)
        if KS_df is None:
            continue

        KS_df.to_csv(out_path.replace('NUM', str(model_idx)))



def main():
    wd = "/home/behzad/Documents/barnes_lab/cplusplus_software/speed_test/repressilator/cpp/"
    data_dir = wd + "output/two_species_stable_6/Population_0/"

    posterior_params_dir = data_dir + "model_sim_params/"
    priors_dir = wd + "input_files_two_species/"
    output_dir = wd + "data_analysis_notebook/posterior_analysis/"
    distributions_dir = output_dir + "distributions/"
    KS_out_dir = output_dir + "KS_data/"

    # generate_posterior_KS_csv(posterior_params_dir, priors_dir, KS_out_dir)
    visualise_posterior_distributions(posterior_params_dir, priors_dir, distributions_dir)


if __name__ == "__main__":
    print("hello world")
    main()
