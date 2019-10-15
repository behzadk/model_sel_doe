import numpy as np
import pandas as pd
import data_utils

import seaborn as sns; sns.set()
import matplotlib as mpl
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [15, 10]

font = {'size'   : 50, }
axes = {'labelsize': 'large', 'titlesize': 'large'}

sns.set_context("poster")
sns.set_style("white")
mpl.rc('font', **font)
mpl.rc('axes', **axes)

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

def generate_marginal_probability_distribution(data_dir, output_dir, hide_x_ticks=False, drop_unnacepted=False):
    print("Generating marginal probability distribution... ")

    # Load model space report
    model_space_report_path = data_dir + "model_space_report.csv"
    model_space_report_df = pd.read_csv(model_space_report_path)

    # Load distances.csv . Contains data on individal simulations
    distances_path = data_dir + "/distances.csv"
    distances_df = pd.read_csv(distances_path)

    if drop_unnacepted:
        model_space_report_df.drop(model_space_report_df[model_space_report_df['accepted_count'] == 0].index, inplace=True)
    
    # Calculate acceptance ratio for each model

    # Sort data frame in order of highest acceptance ratio to lowest
    model_space_report_df = model_space_report_df.sort_values(by='model_marginal', ascending=False).reset_index(drop=True)

    # Generate standard deviation
    # data_utils.generate_replicates_and_std(distances_df, model_space_report_df, 3)
    output_path = output_dir + "model_marginal_probability.pdf"

    data_plotting.plot_model_marginal_distribution(model_space_report_df, output_path, hide_x_ticks)

    print("\n")



def generate_acceptance_probability_distribution(data_dir, output_dir, hide_x_ticks=False, drop_unnacepted=False):
    print("Generating acceptance probability distribution... ")

    # Load model space report
    model_space_report_path = data_dir + "model_space_report.csv"
    model_space_report_df = pd.read_csv(model_space_report_path)

    # Load distances.csv . Contains data on individal simulations
    distances_path = data_dir + "/distances.csv"
    distances_df = pd.read_csv(distances_path)

    if drop_unnacepted:
        model_space_report_df.drop(model_space_report_df[model_space_report_df['accepted_count'] == 0].index, inplace=True)

    # Calculate acceptance ratio for each model
    model_space_report_df['acceptance_probability'] = model_space_report_df.apply(lambda row: row['accepted_count'] / row['simulated_count'], axis=1)

    # Sort data frame in order of highest acceptance ratio to lowest
    model_space_report_df = model_space_report_df.sort_values(by='acceptance_probability', ascending=False).reset_index(drop=True)

    # Generate standard deviation
    data_utils.generate_replicates_and_std(distances_df, model_space_report_df, 3)
    output_path = output_dir + "model_acceptance_probability.pdf"

    data_plotting.plot_acceptance_probability_distribution(model_space_report_df, output_path, hide_x_ticks)

    print("\n")


def generate_acceptance_rate_distribution(data_dir, output_dir, drop_unnacepted=False, hide_x_ticks=False, show_mean=True, show_bf_over_3=True):
    print("Generating acceptance rate distribution... ")

    # Load model space report 
    model_space_report_path = data_dir + "model_space_report.csv"
    model_space_report_df = pd.read_csv(model_space_report_path)
    # print(model_space_report_df['acceptance_probability'])
    print("Number of dead models: ", len(model_space_report_df[model_space_report_df['accepted_count'] == 0]))
    # Load distances.csv . Contains data on individal simulations
    distances_path = data_dir + "/distances.csv"
    distances_df = pd.read_csv(distances_path)
    total_accepted = sum(model_space_report_df['accepted_count'].values)

    if drop_unnacepted:
        model_space_report_df.drop(model_space_report_df[model_space_report_df['accepted_count'] == 0].index, inplace=True)

    # Calculate acceptance ratio for each model
    model_space_report_df['acceptance_ratio'] = model_space_report_df.apply(lambda row: row['accepted_count'] / total_accepted, axis=1)

    # Sort data frame in order of highest acceptance ratio to lowest
    model_space_report_df = model_space_report_df.sort_values(by='acceptance_ratio', ascending=False).reset_index(drop=True)


    # Generate standard deviation
    data_utils.generate_replicates_and_std(distances_df, model_space_report_df, 3)

    output_path = output_dir + "model_posterior_probability.pdf"
    data_plotting.plot_acceptance_rate_distribution(model_space_report_df, output_path, hide_x_ticks, show_mean, show_bf_over_3)

    print("\n")


def split_by_num_parts(data_dir, adj_mat_dir, output_dir, drop_unnacepted=False):
    model_space_report_path = data_dir + "model_space_report.csv"
    model_space_report_df = pd.read_csv(model_space_report_path)

    distances_path = data_dir + "/distances.csv"
    distances_df = pd.read_csv(distances_path)

    adj_matrix_path_template = adj_mat_dir + "model_#REF#_adj_mat.csv"
    all_num_parts, all_AHL_num_parts, all_microcin_num_parts = data_utils.make_num_parts(model_space_report_df, adj_matrix_path_template)

    model_space_report_df['num_parts'] = all_num_parts

    unique_num_parts = model_space_report_df['num_parts'].unique()

    # Acceptance probability
    for num_parts in unique_num_parts:
        sub_model_space_df = model_space_report_df.loc[model_space_report_df['num_parts'] == num_parts]

        valid_model_refs = sub_model_space_df['model_idx'].unique()
        sub_distance_df = distances_df.loc[distances_df['model_ref'].isin(valid_model_refs)]


        if drop_unnacepted:
            sub_model_space_df.drop(sub_model_space_df[sub_model_space_df['accepted_count'] == 0].index, inplace=True)

        sub_model_space_df['acceptance_probability'] = sub_model_space_df.apply(lambda row: row['accepted_count'] / row['simulated_count'], axis=1)

        # Sort data frame in order of highest acceptance ratio to lowest
        sub_model_space_df = sub_model_space_df.sort_values(by='acceptance_probability', ascending=False).reset_index(drop=True)

        # Generate standard deviation
        data_utils.generate_replicates_and_std(sub_distance_df, sub_model_space_df, 3)
        output_path = output_dir + "model_acceptance_probability_" + str(int(num_parts)) + "_parts.pdf"

        data_plotting.plot_acceptance_probability_distribution(sub_model_space_df, output_path, hide_x_ticks=False)


    # Acceptance rate
    for num_parts in unique_num_parts:
        sub_model_space_df = model_space_report_df.loc[model_space_report_df['num_parts'] == num_parts]
        valid_model_refs = sub_model_space_df['model_idx'].unique()
        sub_distance_df = distances_df.loc[distances_df['model_ref'].isin(valid_model_refs)]

        if drop_unnacepted:
            sub_model_space_df.drop(sub_model_space_df[sub_model_space_df['accepted_count'] == 0].index, inplace=True)
        
        # Calculate acceptance ratio for each model
        total_sims = sum(sub_model_space_df['simulated_count'].values)
        sub_model_space_df['acceptance_ratio'] = sub_model_space_df.apply(lambda row: row['accepted_count'] / total_sims, axis=1)

        # Sort data frame in order of highest acceptance ratio to lowest
        sub_model_space_df = sub_model_space_df.sort_values(by='acceptance_ratio', ascending=False).reset_index(drop=True)


        # Generate standard deviation
        data_utils.generate_replicates_and_std(sub_distance_df, sub_model_space_df, 3)
        output_path = output_dir + "model_posterior_probability_" + str(int(num_parts)) + "_parts.pdf"
        data_plotting.plot_acceptance_rate_distribution(sub_model_space_df, output_path, hide_x_ticks=False)

        print("\n")


def compare_top_models_by_parts(data_dir, adj_mat_dir, output_dir, drop_unnacepted=False):
    model_space_report_path = data_dir + "model_space_report.csv"
    model_space_report_df = pd.read_csv(model_space_report_path)

    distances_path = data_dir + "/distances.csv"
    distances_df = pd.read_csv(distances_path)

    adj_matrix_path_template = adj_mat_dir + "model_#REF#_adj_mat.csv"
    all_num_parts, all_AHL_num_parts, all_microcin_num_parts = data_utils.make_num_parts(model_space_report_df, adj_matrix_path_template)

    model_space_report_df['num_parts'] = all_num_parts

    unique_num_parts = model_space_report_df['num_parts'].unique()
    unique_num_parts.sort()

    best_model_idxs = []

    # Get best model for each part class
    for num_parts in unique_num_parts:
        print(num_parts)
        sub_model_space_df = model_space_report_df.loc[model_space_report_df['num_parts'] == num_parts]

        total_accepted = sum(sub_model_space_df['accepted_count'].values)
        sub_model_space_df['acceptance_ratio'] = sub_model_space_df.apply(lambda row: row['accepted_count'] / total_accepted, axis=1)

        # Sort data frame in order of highest acceptance ratio to lowest
        sub_model_space_df = sub_model_space_df.sort_values(by='acceptance_ratio', ascending=False).reset_index(drop=True)

        best_model_idxs.append(sub_model_space_df.iloc[0]['model_idx'])


    # Make subset of only the best models
    sub_model_space_df = model_space_report_df.loc[model_space_report_df['model_idx'].isin(best_model_idxs)]
    total_accepted = sum(sub_model_space_df['accepted_count'].values)
    sub_model_space_df['acceptance_ratio'] = sub_model_space_df.apply(lambda row: row['accepted_count'] / total_accepted, axis=1)

    # Sort data frame in order of highest acceptance ratio to lowest
    sub_model_space_df = sub_model_space_df.sort_values(by='num_parts', ascending=True).reset_index(drop=True)
    sub_distance_df = distances_df.loc[distances_df['model_ref'].isin(best_model_idxs)]

    # Generate standard deviation
    data_utils.generate_replicates_and_std(sub_distance_df, sub_model_space_df, 3)

    # Plot 
    output_path = output_dir + "best_models_posterior_probability" + "_parts.pdf"
    data_plotting.plot_acceptance_rate_distribution(sub_model_space_df, output_path, hide_x_ticks=False, show_mean=False)

    # Generate bayes factors
    model_posterior_probs = [sub_model_space_df.loc[sub_model_space_df['model_idx']==idx]['acceptance_ratio'].values[0] for idx in best_model_idxs]
    bayes_factor_mat = np.zeros([len(model_posterior_probs), len(model_posterior_probs)])

    B_ij = lambda p_m1, p_m2: p_m1/p_m2
    sig_func = lambda x: 1/ np.exp(-x)

    for i in range(len(best_model_idxs)):
        for j in range(len(best_model_idxs)):
            p_m1 = model_posterior_probs[i]
            p_m2 = model_posterior_probs[j]

            bayes_factor_mat[i, j] = (p_m1/p_m2) 

    print(bayes_factor_mat)





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


def generate_posterior_distributions(data_dir, priors_dir, output_dir):
    print("Generating posterior distributions... ")

    output_dir = output_dir + "KS_dist_plots/"
    data_utils.make_folder(output_dir)

    sns.set_style("white")
    sns.set_context("talk", font_scale=1.5)

    sns.despine(top=True)
    show_hist = False
    show_kde = True
    norm_hist = True
    nbins = 30
    kde_bandwidth = 0.05

    posterior_dir = data_dir + "model_sim_params/"

    # Get ordered paths
    ordered_posterior_paths, \
        ordered_prior_param_paths, \
        ordered_prior_species_paths = posterior_analysis.generate_file_paths(posterior_dir, priors_dir)

    # Iterate models
    for model_ref, (f_posterior, f_prior_params, f_prior_species) in enumerate(tqdm(zip(ordered_posterior_paths, ordered_prior_param_paths, ordered_prior_species_paths), total=len(ordered_posterior_paths))):
        model_posterior_df = pd.read_csv(f_posterior, sep=',')

        model_prior_dict = posterior_analysis.import_input_file(f_prior_params)
        model_prior_dict.update(posterior_analysis.import_input_file(f_prior_species))

        # Extract parameter names
        param_names = model_posterior_df.columns[3:]

        # Translate names to math style
        # translated_names = data_utils.translate_param_names(param_names)
        # model_posterior_df.rename(columns=dict(zip(model_posterior_df.columns[3:], translated_names)), inplace=True)
        param_names = model_posterior_df.columns[3:]
        free_params = []

        # Extract parameter columns that are not constants
        for param in param_names:
            if model_posterior_df[param].nunique() == 1:
                continue

            else:
                free_params.append(param)

        # Select for only accepted simulations
        accepted_sims = model_posterior_df.loc[model_posterior_df['Accepted'] == True]
        if len(accepted_sims) <= 1:
            continue

        # Generate subplotgrid
        ncols = 4
        nrows = math.ceil(len(free_params)/ncols)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
        axes = axes.reshape(-1)

        # Calculate kolmogorov_smirnov 95% critical value
        D_crit = 1.224 * math.sqrt(1 / len(accepted_sims) + 1 / len(model_posterior_df))

        # Iterate free parameters and plot each
        i = 0
        for param in free_params:
            D_n = data_utils.kolmogorov_smirnov_test(accepted_sims[param].values, model_posterior_df[param].values)
            # D_n = data_utils.entropy(accepted_sims[param].values, model_posterior_df[param].values)

            clip_max = max(model_posterior_df[param])
            clip_min = min(model_posterior_df[param])
            # kde_params = {"bw": kde_bandwidth, 'cumulative': False}
            kde_params = {'cumulative': False}

            sns.distplot(model_posterior_df[param], bins=nbins, hist=show_hist, kde=show_kde, kde_kws=kde_params,
                         label="Sample distribution", ax=axes[i], norm_hist=norm_hist)

            sns.distplot(accepted_sims[param], bins=nbins, hist=show_hist, kde=show_kde, kde_kws=kde_params,
                         label="Posterior distribution", ax=axes[i], norm_hist=norm_hist)

            D_n = round(D_n, 4)
            axes[i].text(1, 1, r"$D_n = " + str(D_n) + "$", fontsize='small',
                         horizontalalignment='center',
                         verticalalignment='center',
                         transform=axes[i].transAxes)
            i += 1

        # plotting aesthetics
        for ax in axes:
            # ax.set(xscale='log', yscale='log')
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0), useMathText=True)
            legend = ax.legend()
            ax.legend().remove()

        # Add legend in place of last subplot
        handles, labels = axes[0].get_legend_handles_labels()

        D_crit = round(D_crit, 4)
        axes[-1].text(0.6, 0.5, r"$D_{crit} = " + str(D_crit) + "$",
                      horizontalalignment='right',
                      verticalalignment='center')

        axes[-1].set_axis_off()
        axes[-1].legend(handles, labels, loc='lower right')

        figure_name = output_dir + "distribution_model_REF_.pdf"
        figure_name = figure_name.replace('REF', str(model_ref))

        plt.tight_layout()
        plt.savefig(figure_name, dpi=500, bbox_inches='tight')
        plt.close(fig)
    print("\n")


def write_model_order(data_dir, output_dir):
    print("Writing model_order.txt ... ")

    # Load model space report 
    model_space_report_path = data_dir + "model_space_report.csv"
    model_space_report_df = pd.read_csv(model_space_report_path)

    # Load distances.csv . Contains data on individual simulations
    distances_path = data_dir + "/distances.csv"
    distances_df = pd.read_csv(distances_path)

    # Calculate acceptance ratio for each model
    total_sims = sum(model_space_report_df['simulated_count'].values)
    model_space_report_df['acceptance_ratio'] = model_space_report_df.apply(lambda row: row['accepted_count'] / total_sims, axis=1)

    # Sort data frame in order of highest acceptance ratio to lowest
    model_space_report_df = model_space_report_df.sort_values(by='acceptance_ratio', ascending=False).reset_index(drop=True)
    top_model = list(model_space_report_df['acceptance_ratio'].values)[0]

    # for i, j in zip(list(model_space_report_df['model_idx'].values), list(model_space_report_df['acceptance_ratio'].values)):
    #   print(i, top_model/j)

    # exit()


    file = open(output_dir + "model_order.txt", "w") 
    
    for item in list(model_space_report_df['model_idx'].values):
        file.write("%s\n" % item)
    file.close()

    print("\n")


def loop_scatter_test(data_dir, inputs_dir):
    model_space_report_df = pd.read_csv(data_dir + "model_space_report.csv")
    x = data_utils.make_feedback_loop_counts(data_dir, inputs_dir)

    model_space_report_df['pos_loops'] = x[0]
    model_space_report_df['neg_loops'] = x[1]
    model_space_report_df['acceptance_probability'] = model_space_report_df.apply(lambda row: row['accepted_count'] / row['simulated_count'], axis=1)

    plt.scatter(model_space_report_df['pos_loops']/model_space_report_df['neg_loops'], model_space_report_df['acceptance_probability'], s=2)
    plt.show()

    plt.scatter(model_space_report_df['neg_loops'], model_space_report_df['acceptance_probability'], s=2)
    plt.show()


def generate_solver_performace_report():
    eig_init_df = pd.read_csv(eigenvalues_init_path)
    print(eig_init_df.columns)
    eig_init_df_full_term = eig_init_df.loc[eig_init_df['integ_error'].isnull()]
    eig_init_df_early_term = eig_init_df[eig_init_df['integ_error'].str.contains('no_progess_error')]

    eig_init_df_early_term = eig_init_df.loc[eig_init_df['integ_error'] == ('no_progress_error')]
    


def select_k(spectrum, minimum_energy = 0.9):
    running_total = 0.0
    total = sum(spectrum)
    if total == 0.0:
        return len(spectrum)
    for i in range(len(spectrum)):
        running_total += spectrum[i]
        if running_total / total >= minimum_energy:
            return i + 1
    return len(spectrum)


def adjacency_matrix_ranking(data_dir, inputs_dir):
    model_space_report_df = pd.read_csv(data_dir + "model_space_report.csv")
    adj_mat_name_template = "model_#REF#_adj_mat.csv"
    adj_matrix_path_template = inputs_dir + "adj_matricies/" + adj_mat_name_template
    model_space_report_df.drop(model_space_report_df[model_space_report_df['accepted_count'] == 0].index, inplace=True)

    models = model_space_report_df.model_idx.values

    positive_loops = []
    negative_loops = []


    total_accepted = sum(model_space_report_df['accepted_count'].values)

    # Calculate acceptance ratio for each model
    model_space_report_df['acceptance_ratio'] = model_space_report_df.apply(lambda row: row['accepted_count'] / total_accepted, axis=1)

    adj_mat_194 = adj_matrix_path_template.replace("#REF#", str(39))
    adj_mat_194_df = pd.read_csv(adj_mat_194)
    adj_mat_194_df.drop([adj_mat_194_df.columns[0]], axis=1, inplace=True)
    
    G_1 = nx.from_numpy_matrix(adj_mat_194_df.values)
    laplacian1 = nx.spectrum.laplacian_spectrum(G_1)

    posterior_probs = []
    similarity = []

    for m in models:
        x = model_space_report_df.loc[model_space_report_df['model_idx'] == m ]['acceptance_ratio'].values[0]
        posterior_probs.append(x)

        adj_mat_path = adj_matrix_path_template.replace("#REF#", str(m))
        adj_mat_df = pd.read_csv(adj_mat_path)
        adj_mat_df.drop([adj_mat_df.columns[0]], axis=1, inplace=True)

        G_2 = nx.from_numpy_matrix(adj_mat_df.values)

        laplacian2 = nx.spectrum.laplacian_spectrum(G_2)

        k1 = select_k(laplacian1)
        k2 = select_k(laplacian2)
        k = min(k1, k2)

        similarity.append(sum((laplacian1[:k] - laplacian2[:k])**2))


    sns.scatterplot(x=posterior_probs, y=similarity)
    plt.show()
    exit()
    for m in models:
        adj_mat_path = adj_matrix_path_template.replace("#REF#", str(m))
        adj_mat_df = pd.read_csv(adj_mat_path)
    

        adj_mat_df.drop([adj_mat_df.columns[0]], axis=1, inplace=True)

        col_names = adj_mat_df.columns
        strain_indexes = [idx for idx, i in enumerate(col_names) if 'N_' in i]
        adj_mat = adj_mat_df.values
        G=nx.from_numpy_matrix(adj_mat)
        pos = nx.circular_layout(G)
        nx.draw_circular(G)
        labels = {i : col_names[i] for i in G.nodes()}
        
    

    # Remove column 0, which contains row names
    # adj_mat = adj_mat[:, 1:]
    n_species = np.shape(adj_mat)[0]

    output = []


def ABC_SMC_analysis():
    wd = "/home/behzad/Documents/barnes_lab/cplusplus_software/speed_test/repressilator/cpp/"
    
    ## Two species
    if 0:
        experiment_name = "two_species_stable_0"
        inputs_dir = wd + "/input_files/input_files_two_species_0/"
        R_script = "plot-motifs-two.R"

    ## Two species SMC
    if 0:
        experiment_name = "two_species_stable_0_SMC_"
        inputs_dir = wd + "/input_files/input_files_two_species_0/"
        R_script = "plot-motifs-two.R"

    ## Two species SMC
    if 1:
        experiment_name = "three_species_stable_0_SMC"
        inputs_dir = wd + "/input_files/input_files_two_species_0/"
        R_script = "plot-motifs-two.R"

    ## Three species
    if 0:
        experiment_name = "three_species_stable_0_comb"
        inputs_dir = wd + "/input_files/input_files_three_species_0/"
        R_script = "plot-motifs-three.R"

    ## Spock Manuscript
    if 0:
        experiment_name = "spock_manu_stable_0"
        inputs_dir = wd + "input_files_two_species_spock_manu_0/"
        # R_script = "plot-motifs-three.R"

    adj_mat_dir = inputs_dir + "adj_matricies/"

    exp_repeat_dirs = glob(wd + "output/" + experiment_name + "*/")
    final_pop_dirs = []

    for rep in exp_repeat_dirs:
        sub_dirs = glob(rep + "*/")
        pop_dirs = [f for f in sub_dirs if "Population" in f.split('/')[-2]]
        pop_dirs = sorted(pop_dirs, key=lambda a: a[-2])
        final_pop_dirs.append(pop_dirs[-1])
        print(pop_dirs[-1])

    exit()
    final_pop_dir = pop_dirs[-1]

    for pop_dir in pop_dirs:
        output_dir = pop_dir + "analysis/"
        priors_dir = inputs_dir + "input_files/"
        KS_data_dir = pop_dir + "KS_data/"

        data_utils.make_folder(output_dir)
        data_utils.make_folder(KS_data_dir)

        # compare_top_models_by_parts(pop_dir, adj_mat_dir, output_dir)
        write_model_order(pop_dir, output_dir)
        subprocess.call(['Rscript', R_script, adj_mat_dir, pop_dir+"analysis/", output_dir])
        generate_marginal_probability_distribution(pop_dir, output_dir, hide_x_ticks=True, drop_unnacepted=True)

        # split_by_num_parts(pop_dir, adj_mat_dir, output_dir)

    priors_dir = inputs_dir + "input_files/"
    KS_data_dir = final_pop_dir + "KS_data/"
    posterior_analysis.generate_posterior_KS_csv(final_pop_dir, priors_dir, KS_data_dir)


    generate_posterior_distributions(final_pop_dir, priors_dir, output_dir)


def main():
    wd = "/home/behzad/Documents/barnes_lab/cplusplus_software/speed_test/repressilator/cpp/"
    
    ## Two species
    if 1:
        experiment_name = "two_species_stable_0"
        inputs_dir = wd + "/input_files/input_files_two_species_0/"
        R_script = "plot-motifs-two.R"


    ## Three species
    if 0:
        experiment_name = "three_species_stable_0_comb"
        inputs_dir = wd + "/input_files/input_files_three_species_0/"
        R_script = "plot-motifs-three.R"

    ## Spock Manuscript
    if 0:
        experiment_name = "spock_manu_stable_0"
        inputs_dir = wd + "input_files_two_species_spock_manu_0/"
        # R_script = "plot-motifs-three.R"


    adj_mat_dir = inputs_dir + "adj_matricies/"

    data_dir = wd + "output/" + experiment_name + "/Population_0/"
    output_dir = data_dir + "analysis/"
    priors_dir = inputs_dir + "input_files/"
    KS_data_dir = data_dir + "KS_data/"

    data_utils.make_folder(output_dir)
    data_utils.make_folder(KS_data_dir)

    # ml_analysis.adj_mat_spectral_cluster(inputs_dir, data_dir, output_dir)
    # ml_analysis.rdn_forest_test(inputs_dir, data_dir, output_dir)
    # ml_analysis.hierarchical_cluster(inputs_dir, data_dir, output_dir)


    # exit()
    # adjacency_matrix_ranking(data_dir, inputs_dir)
    # exit()
    compare_top_models_by_parts(data_dir, adj_mat_dir, output_dir)
    write_model_order(data_dir, output_dir)

    # generate_critical_parameter_bar_plot(data_dir, KS_data_dir, output_dir, 5)
    subprocess.call(['Rscript', R_script, adj_mat_dir, data_dir+"analysis/", output_dir])
    generate_acceptance_rate_distribution(data_dir, output_dir, drop_unnacepted=True, hide_x_ticks=True, show_mean=False, show_bf_over_3=True)
    generate_acceptance_probability_distribution(data_dir, output_dir, hide_x_ticks=True, drop_unnacepted=True)
    exit()
    # # # Generate KS value files
    # generate_posterior_distributions(data_dir, priors_dir, output_dir)
    split_by_num_parts(data_dir, adj_mat_dir, output_dir)

    posterior_analysis.generate_posterior_KS_csv(data_dir, priors_dir, KS_data_dir)


    generate_posterior_distributions(data_dir, priors_dir, output_dir)
    eigenvalues_init_path = data_dir + "/eigenvalues_do_fsolve_state.csv"


if __name__ == "__main__":
    ABC_SMC_analysis()