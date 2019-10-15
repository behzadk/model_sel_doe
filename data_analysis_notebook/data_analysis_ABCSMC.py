import numpy as np
import pandas as pd
import data_utils

import seaborn as sns; sns.set()
import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib import rcParams

plt.rcParams['figure.figsize'] = [15, 10]

font = {'size'   : 50, }
axes = {'labelsize': 'large', 'titlesize': 'large'}

sns.set_context("poster")
sns.set_style("white")
mpl.rc('font', **font)
mpl.rc('axes', **axes)
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
# plt.rcParams['text.usetex'] = True
plt.rcParams['axes.unicode_minus'] = False

import pandas as pd
import posterior_analysis
import matplotlib.style as style
import math

from tqdm import tqdm
import subprocess
import data_plotting
import networkx as nx

import ml_analysis
import os
from glob import glob


def merge_model_space_report_df_list(df_list):
    merge_func = lambda x, y, suff: pd.merge(x, y, on='model_idx', suffixes=suff)
    for i in range(1, len(df_list)):
        df_list[0] = merge_func(df_list[0], df_list[i], [str("_" + str(i-1)), str("_" + str(i))])
        print(list(df_list[0]))
    model_space_report_df = df_list[0]

    return model_space_report_df


def generate_model_space_statistics(df, target_column_name):
    # Get appropriate columns to generate stdev
    column_names = list(df)
    target_cols = [col for col in column_names if target_column_name in col]

    model_stds = []
    model_means = []
    for idx, row in df.iterrows():
        model_stds.append(np.std(row[target_cols].values))
        model_means.append(np.median(row[target_cols].values))

    df[target_column_name + "_std"] = model_stds
    df[target_column_name + "_mean"] = model_means

def generate_marginal_probability_distribution(pop_dir_list, output_dir, hide_x_ticks=True, drop_unnacepted=True, show_median=True):
    print("Generating marginal probability distribution... ")

    model_space_report_list = []
    num_pops = len(pop_dir_list)

    for data_dir in pop_dir_list:
        # Load model space report
        model_space_report_path = data_dir + "model_space_report.csv"
        model_space_report_df = pd.read_csv(model_space_report_path, index_col=0)
        model_space_report_list.append(model_space_report_df)

    model_space_report_df = merge_model_space_report_df_list(model_space_report_list)
    generate_model_space_statistics(model_space_report_df, "model_marginal")
    
    if drop_unnacepted:
        model_space_report_df.drop(model_space_report_df[model_space_report_df['model_marginal_mean'] == 0].index, inplace=True)

    model_space_report_df = model_space_report_df.sort_values(by='model_marginal_mean', ascending=False).reset_index(drop=True)

    output_path = output_dir + "model_marginal_probability.pdf"
    
    fig, ax = plt.subplots()

    if num_pops > 1:
        ax.errorbar(model_space_report_df.index, 
                    model_space_report_df['model_marginal_mean'], 
                    yerr=model_space_report_df['model_marginal_std'], fmt=',', color='black', alpha=1,
                    label=None, elinewidth=0.5)

    sns.barplot(model_space_report_df.index, model_space_report_df.model_marginal_mean, 
                     data=model_space_report_df, alpha=0.9, ax=ax)
    ax.unicode_minus = True

    if hide_x_ticks:
        ax.set(xticklabels=[])
        ax.set(xlabel='')
        ax.legend().remove()    
    
    else:
        ax.set(xticklabels=model_space_report_df['model_idx'])
        ax.set(xlabel='Model')
        ax.legend()

    if show_median:
        median = np.median(model_space_report_df['model_marginal_mean'].values)
        ax.axhline(median, ls='--', label='Median', linewidth=1.0)
        ax.legend()


    ax.set(ylabel='Model marginal probability')
    ax.set(xlim=(-0.5,None))
    ax.set(ylim=(-0))
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_alpha(0.5)
    ax.spines["bottom"].set_alpha(0.5)
    fig.tight_layout()
    plt.savefig(output_path, dpi=500)

    output_path = output_dir + "model_marginal_probability_log_scale.eps"
    ax.set(yscale="log")
    ax.set(ylim=(None, None))
    plt.savefig(output_path, dpi=500)

    print(output_path)
    print("\n")

def get_total_simulations(rep_dirs):
    total_sims = 0

    for rep in rep_dirs:
        sub_dirs = glob(rep + "*/")
        pop_dirs = [f for f in sub_dirs if "Population" in f.split('/')[-2]]
        pop_dirs = sorted(pop_dirs, key=lambda a: a[-2])

        for pop in pop_dirs:
            model_space_report_path = pop + "model_space_report.csv"
            df = pd.read_csv(model_space_report_path)
            total_sims += sum(df['simulated_count'].values)

    return total_sims

def plot_all_model_param_distributions(pop_dir, inputs_dir, figure_output_dir):
    model_space_report_path = pop_dir + "model_space_report.csv"
    model_space_report_df = pd.read_csv(model_space_report_path, index_col=0)
    # generate_model_space_statistics(model_space_report_df, "model_marginal")
    
    model_space_report_df.drop(model_space_report_df[model_space_report_df['accepted_count'] == 0].index, inplace=True)

    for model_idx in model_space_report_df['model_idx']:
        top_ten_models = [95, 145, 43, 101, 149, 49, 93, 99, 164, 92]
        if model_idx not in top_ten_models:
            continue

        model_output_parameter_path = pop_dir + "model_sim_params/" + "model_" + str(model_idx) + "_all_params"
        model_input_parameter_path = inputs_dir + "input_files/params_" + str(model_idx) + ".csv"
        model_input_species_path = inputs_dir + "input_files/species_" + str(model_idx) + ".csv"

        R_script = "dens_plot_2D_spock.R"
        subprocess.call(['Rscript', R_script, model_output_parameter_path, model_input_parameter_path, model_input_species_path, str(model_idx), figure_output_dir])

    exit()

def population_analysis(pop_dir, inputs_dir):
    analysis_dir = pop_dir + "analysis/"
    data_utils.make_folder(analysis_dir)
    print(pop_dir)
    generate_marginal_probability_distribution([pop_dir], analysis_dir, hide_x_ticks=True, drop_unnacepted=True)
    write_model_order([pop_dir], analysis_dir)
    
    param_dists_dir = analysis_dir + "param_dists/"
    data_utils.make_folder(param_dists_dir)
    plot_all_model_param_distributions(pop_dir, inputs_dir, param_dists_dir)


def compare_top_models_by_parts(data_dir, adj_mat_dir, output_dir, drop_unnacepted=False):
    hide_x_ticks = False
    show_median = True

    model_space_report_path = data_dir + "combined_model_space_report.csv"
    model_space_report_df = pd.read_csv(model_space_report_path)

    adj_matrix_path_template = adj_mat_dir + "model_#REF#_adj_mat.csv"
    all_num_parts, all_AHL_num_parts, all_microcin_num_parts = data_utils.make_num_parts(model_space_report_df, adj_matrix_path_template)

    model_space_report_df['num_parts'] = all_num_parts

    unique_num_parts = model_space_report_df['num_parts'].unique()
    unique_num_parts.sort()

    best_model_idxs = []

    # Get best model for each part class
    for num_parts in unique_num_parts:
        sub_model_space_df = model_space_report_df.loc[model_space_report_df['num_parts'] == num_parts]

        # Sort data frame in order of highest acceptance ratio to lowest
        sub_model_space_df = sub_model_space_df.sort_values(by='model_marginal_mean', ascending=False).reset_index(drop=True)
        best_model_idxs.append(sub_model_space_df.iloc[0]['model_idx'])

        # Make subset of only the best models
        sub_model_space_df = model_space_report_df.loc[model_space_report_df['model_idx'].isin(best_model_idxs)]

        # Sort data frame in order of highest acceptance ratio to lowest
        sub_model_space_df = sub_model_space_df.sort_values(by='num_parts', ascending=True).reset_index(drop=True)

    # Plot 
    output_path = output_dir + "best_models_marginal_by_parts.pdf"
    
    fig, ax = plt.subplots()

    ax.errorbar(sub_model_space_df.index, 
                sub_model_space_df['model_marginal_mean'], 
                yerr=sub_model_space_df['model_marginal_std'], fmt=',', color='black', alpha=1,
                label=None, elinewidth=0.5)

    sns.barplot(sub_model_space_df.index, sub_model_space_df.model_marginal_mean, 
                     data=sub_model_space_df, alpha=0.9, ax=ax)
    ax.unicode_minus = True

    if hide_x_ticks:
        ax.set(xticklabels=[])
        ax.set(xlabel='')
        ax.legend().remove()    
    
    else:
        ax.set(xticklabels=sub_model_space_df['model_idx'])
        ax.set(xlabel='Model')
        ax.legend()

    if show_median:
        median = np.median(sub_model_space_df['model_marginal_mean'].values)
        ax.axhline(median, ls='--', label='Median', linewidth=1.0)
        ax.legend()


    ax.set(ylabel='Model marginal probability')
    ax.set(xlim=(-0.5,None))
    ax.set(ylim=(-0))
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_alpha(0.5)
    ax.spines["bottom"].set_alpha(0.5)
    fig.tight_layout()
    plt.savefig(output_path, dpi=500)

    # output_path = output_dir + "model_marginal_probability_log_scale.eps"
    # ax.set(yscale="log")
    # ax.set(ylim=(None, None))
    # plt.savefig(output_path, dpi=500)

    print(output_path)
    print("\n")







    # Generate bayes factors
    model_posterior_probs = [sub_model_space_df.loc[sub_model_space_df['model_idx']==idx]['model_marginal_mean'].values[0] for idx in best_model_idxs]
    bayes_factor_mat = np.zeros([len(model_posterior_probs), len(model_posterior_probs)])

    B_ij = lambda p_m1, p_m2: p_m1/p_m2
    sig_func = lambda x: 1/ np.exp(-x)

    for i in range(len(best_model_idxs)):
        for j in range(len(best_model_idxs)):
            p_m1 = model_posterior_probs[i]
            p_m2 = model_posterior_probs[j]

            bayes_factor_mat[i, j] = (p_m1/p_m2) 

    print(bayes_factor_mat)



def split_by_num_parts(data_dir, adj_mat_dir, output_dir, drop_unnacepted=False):
    hide_x_ticks = False
    show_median = True

    model_space_report_path = data_dir + "combined_model_space_report.csv"
    model_space_report_df = pd.read_csv(model_space_report_path)

    adj_matrix_path_template = adj_mat_dir + "model_#REF#_adj_mat.csv"
    # all_num_parts, all_AHL_num_parts, all_microcin_num_parts = data_utils.make_num_parts(model_space_report_df, adj_matrix_path_template)
    sum_adj_mat = data_utils.make_num_parts_alt(model_space_report_df, adj_matrix_path_template)

    model_space_report_df['num_parts'] = sum_adj_mat

    unique_num_parts = model_space_report_df['num_parts'].unique()

    # Acceptance rate
    for num_parts in unique_num_parts:
        sub_model_space_df = model_space_report_df.loc[model_space_report_df['num_parts'] == num_parts]
        valid_model_refs = sub_model_space_df['model_idx'].unique()

        if drop_unnacepted:
            sub_model_space_df.drop(sub_model_space_df[sub_model_space_df['model_marginal_mean'] == 0].index, inplace=True)
        
        if len(sub_model_space_df) == 0:
            continue

        # Sort data frame in order of highest acceptance ratio to lowest
        sub_model_space_df = sub_model_space_df.sort_values(by='model_marginal_mean', ascending=False).reset_index(drop=True)


        # Generate standard deviation
        output_path = output_dir + "model_marginal_probability_" + str(int(num_parts)) + "_parts.pdf"

        fig, ax = plt.subplots()

        ax.errorbar(sub_model_space_df.index, 
                    sub_model_space_df['model_marginal_mean'], 
                    yerr=sub_model_space_df['model_marginal_std'], fmt=',', color='black', alpha=1,
                    label=None, elinewidth=0.5)
        print(sub_model_space_df)

        sns.barplot(sub_model_space_df.index, sub_model_space_df.model_marginal_mean, 
                         data=sub_model_space_df, alpha=0.9, ax=ax)
        ax.unicode_minus = True

        if hide_x_ticks:
            ax.set(xticklabels=[])
            ax.set(xlabel='')
            ax.legend().remove()    
        
        else:
            ax.set(xticklabels=sub_model_space_df['model_idx'])
            ax.set(xlabel='Model')
            ax.legend()

        if show_median:
            median = np.median(sub_model_space_df['model_marginal_mean'].values)
            ax.axhline(median, ls='--', label='Median', linewidth=1.0)
            ax.legend()


        ax.set(ylabel='Model marginal probability')
        ax.set(xlim=(-0.5,None))
        ax.set(ylim=(-0))
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_alpha(0.5)
        ax.spines["bottom"].set_alpha(0.5)
        fig.tight_layout()
        plt.savefig(output_path, dpi=500)

        # output_path = output_dir + "model_marginal_probability_log_scale.eps"
        # ax.set(yscale="log")
        # ax.set(ylim=(None, None))
        # plt.savefig(output_path, dpi=500)

        print(output_path)
        print("\n")


        print("\n")


def generate_critical_parameter_bar_plot(data_dir, KS_data_dir, output_dir, num_params):
    print("Generating critical parameter bar plot... ")

    # Load model space report 
    model_space_report_path = data_dir + "model_space_report.csv"
    model_space_report_df = pd.read_csv(model_space_report_path)
    

    model_indexes = model_space_report_df['model_idx'].values
    KS_data_path_template = KS_data_dir + "model_NUM_KS.csv"
    
    model_space_report_df['crit_param_1'] = np.nan

    for model_idx in tqdm(model_indexes):
        ks_data_path = KS_data_path_template.replace('NUM', str(int(model_idx)))

        try:
            df = pd.read_csv(ks_data_path, sep=',')
    
        except(FileNotFoundError, ValueError):
            continue

        param_names = df.columns[3:]
        row = df.iloc[0]
        param_KS_values = row[3:].values

        
        # Reorder parameters according to their KS value
        params_order = [param for _,param in sorted(zip(param_KS_values, param_names), reverse=True)]
        critical_parameters = [param for param in params_order]
        model_space_report_df.loc[model_idx, 'crit_param_1'] = critical_parameters[0]

    # Make plot!
    style.use('seaborn-muted')
    current_palette = sns.color_palette()

    fig, axes = plt.subplots(ncols=1)
    crit_params = [1]

    for idx, col_num in enumerate(crit_params):
        print(col_num)
        crit_param_col = 'crit_param_NUM'.replace('NUM', str(col_num))
        names = data_utils.translate_param_names(model_space_report_df[crit_param_col].value_counts().index)
        value_counts = model_space_report_df[crit_param_col].value_counts()

        value_counts = value_counts[:num_params]
        names = names[:num_params]
        
        sns.barplot(value_counts, names, ax=axes)
        
        axes.spines["right"].set_visible(False)
        axes.spines["top"].set_visible(False)
        axes.set_title('Rank ' + str(col_num) + ' critical parameter')
        axes.tick_params(labelsize=40)
        axes.set_xlabel('Count')

    plt.tight_layout()
    plt.savefig(output_dir+'rank_one_params.pdf', dpi=500)

    print("\n")


def write_model_order(pop_dir_list, output_dir):
    model_space_report_list = []
    for data_dir in pop_dir_list:
        # Load model space report
        model_space_report_path = data_dir + "model_space_report.csv"
        model_space_report_df = pd.read_csv(model_space_report_path, index_col=0)
        model_space_report_list.append(model_space_report_df)

    model_space_report_df = merge_model_space_report_df_list(model_space_report_list)
    generate_model_space_statistics(model_space_report_df, "model_marginal")

    model_space_report_df = model_space_report_df.sort_values(by='model_marginal_mean', ascending=False).reset_index(drop=True)
    
    file = open(output_dir + "model_order.txt", "w") 
    for item in list(model_space_report_df['model_idx'].values):
        file.write("%s\n" % item)
    file.close()

    print("\n")


def write_combined_model_space_report(pop_dir_list, output_dir):
    output_path = output_dir + "combined_model_space_report.csv"

    model_space_report_list = []
    for data_dir in pop_dir_list:
        # Load model space report
        model_space_report_path = data_dir + "model_space_report.csv"
        model_space_report_df = pd.read_csv(model_space_report_path, index_col=0)
        model_space_report_list.append(model_space_report_df)

    model_space_report_df = merge_model_space_report_df_list(model_space_report_list)
    generate_model_space_statistics(model_space_report_df, "model_marginal")

    model_space_report_df = model_space_report_df.sort_values(by='model_marginal_mean', ascending=False).reset_index(drop=True)
    model_space_report_df.to_csv(output_path)

def write_experiment_summary(population_size, n_repeats, exp_repeat_dirs, inputs_dir, output_dir):
    # Number of repeats
    # Number of models
    # Population size
    # Total number of simulations

    total_sims = get_total_simulations(exp_repeat_dirs)

    model_space_report_df = pd.read_csv(output_dir + "combined_model_space_report.csv")
    n_models = len(model_space_report_df['model_idx'].values)
    n_dead_models = len(model_space_report_df.loc[model_space_report_df['model_marginal_mean'] == 0])

    file = open(output_dir + "experiment_summary.txt", "w") 
    file.write("Number of repeats: " + str(n_repeats) + "\n")
    file.write("Number of models: " + str(n_models) + "\n")
    file.write("Number of dead models: " + str(n_dead_models) + "\n")
    file.write("Population size: " + str(population_size) + "\n")
    file.write("Total simulations: " + str(total_sims) + "\n")
    file.close()





def ABC_SMC_analysis():
    wd = "/home/behzad/Documents/barnes_lab/cplusplus_software/speed_test/repressilator/cpp/"
    
    ## Two species
    if 0:
        experiment_name = "two_species_stable_0"
        inputs_dir = wd + "/input_files/input_files_two_species_0/"
        R_script = "plot-motifs-two.R"

    ## Two species SMC
    if 0:
        experiment_name = "two_species_stable_0_SMC"
        inputs_dir = wd + "/input_files/input_files_two_species_0/"
        R_script = "plot-motifs-two.R"

    ## Two species SMC
    if 0:
        experiment_name = "three_species_stable_0_SMC"
        inputs_dir = wd + "/input_files/input_files_two_species_0/"
        R_script = "plot-motifs-two.R"

    ## Three species
    if 0:
        experiment_name = "three_species_stable_0_comb"
        inputs_dir = wd + "/input_files/input_files_three_species_0/"
        R_script = "plot-motifs-three.R"

    ## Spock Manuscript
    if 1:
        experiment_name = "spock_manu_stable_0_SMC"
        inputs_dir = wd + "input_files/input_files_two_species_spock_manu_1/"
        R_script = "plot-motifs-spock.R"

    ## Spock Manuscript
    if 1:
        experiment_name = "spock_manu_stable_1_SMC"
        inputs_dir = wd + "input_files/input_files_two_species_spock_manu_1/"
        R_script = "plot-motifs-spock.R"


    combined_analysis_output_dir = wd + "output/" + experiment_name + "/experiment_analysis/"
    data_utils.make_folder(combined_analysis_output_dir)
    adj_mat_dir = inputs_dir + "adj_matricies/"

    exp_repeat_dirs = glob(wd + "output/" + experiment_name + "/" + "*/")
    clean_exp_repeat_dirs = []
    final_pop_dirs = []
    first_pop_dirs = []

    n_repeats = 0
    for rep in exp_repeat_dirs:
        sub_dirs = glob(rep + "*/")
        pop_dirs = [f for f in sub_dirs if "Population" in f.split('/')[-2]]
        pop_dirs = sorted(pop_dirs, key=lambda a: a[-2])

        for pop in pop_dirs:
            # population_analysis(pop, inputs_dir)
            pass

        try:
            first_pop_dirs.append(pop_dirs[0])
            final_pop_dirs.append(pop_dirs[-1])
            clean_exp_repeat_dirs.append(rep)
            n_repeats += 1

        except IndexError:
            continue

    for idx, pop_dir in enumerate(final_pop_dirs):
        analysis_dir = pop_dir + "analysis/"

        data_utils.make_folder(analysis_dir)

        param_dists_dir = analysis_dir + "param_dists/"
        data_utils.make_folder(param_dists_dir)

        KS_data_dir = analysis_dir + "KS_data/"
        data_utils.make_folder(KS_data_dir)
        # posterior_analysis.generate_posterior_KS_csv_ABCSMC(final_pop_dir=pop_dir, first_pop_dir=first_pop_dirs[idx], 
        #     priors_dir=inputs_dir, output_dir=KS_data_dir)
        
        plot_all_model_param_distributions(pop_dir, inputs_dir, param_dists_dir)


    # Use final population data for analysis
    write_combined_model_space_report(final_pop_dirs, combined_analysis_output_dir)
    generate_marginal_probability_distribution(final_pop_dirs, combined_analysis_output_dir, hide_x_ticks=True, drop_unnacepted=True)
    # compare_top_models_by_sparts(combined_analysis_output_dir, adj_mat_dir, combined_analysis_output_dir, drop_unnacepted=False)
    split_by_num_parts(combined_analysis_output_dir, adj_mat_dir, combined_analysis_output_dir, drop_unnacepted=True)
    write_model_order(final_pop_dirs, combined_analysis_output_dir)
    write_experiment_summary(2500, n_repeats, exp_repeat_dirs, inputs_dir, combined_analysis_output_dir)
    # subprocess.call(['Rscript', R_script, adj_mat_dir, combined_analysis_output_dir, combined_analysis_output_dir])


if __name__ == "__main__":
    ABC_SMC_analysis()
