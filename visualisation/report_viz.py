import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def report_summary(model_space_report_path):
    df = pd.read_csv(model_space_report_path)

    models = df.model_idx.values
    accepted_count = df.accepted_count.values
    simulated_count = df.simulated_count.values

    total_simulations = sum(simulated_count)
    total_accepted_simulations = sum(accepted_count)

    mean_simulations = np.mean(simulated_count)
    mean_accepted = np.mean(accepted_count)

    acceptance_ratio = []

    for m in models:
        acceptance_ratio.append(accepted_count[m] / simulated_count[m])

    df['acceptance_ratio'] = acceptance_ratio

    mean_acceptance_ratio = np.mean(acceptance_ratio)

    print("Total_simulations: ", total_simulations)
    print("Total accepted simulations: ", total_accepted_simulations)
    print("")
    print("Mean simulations per model: ", mean_simulations)
    print("Mean accepted per model: ", mean_accepted)
    print("Mean acceptance ratio: ", mean_acceptance_ratio)

    df = df.loc[df['acceptance_ratio'] == 0]
    print("Dead models: ", len(df))


def find_spock(model_ref_path):
    df = pd.read_csv(model_ref_path)
    col_names = list(df)
    col_names[0] = 'model_ref'
    df.columns = col_names

    # cell_0 producing one microcin
    df.replace('', np.NaN, inplace=True)

    # Cell 0 producing a microcin
    df = df.loc[(df['cell_0_M_0'] != None)]

    # Cell 1 not producing any microcin
    df = df.loc[np.isnan(df["cell_1_M_0"]) == True]
    print(len(df))

    # Cell 1 not producing any AHL
    df = df.loc[np.isnan(df["cell_1_AHL_0"]) == True]

    # Cell 1 is sensitive
    cell_1_sens_idx = []
    sens_0_vals = df.cell_1_Sens_0.values
    sens_1_vals = df.cell_1_Sens_1.values

    for idx_1, i in enumerate(sens_0_vals):
        if isinstance(sens_0_vals[idx_1], str) or isinstance(sens_1_vals[idx_1], str):
            cell_1_sens_idx.append(idx_1)
    df = df.iloc[cell_1_sens_idx]

    # Cell 0 is not sensitive
    cell_0_not_sens_idx = []
    sens_0_vals = df.cell_0_Sens_0.values
    sens_1_vals = df.cell_0_Sens_1.values
    for idx_1, i in enumerate(sens_0_vals):
        if not isinstance(sens_0_vals[idx_1], str) and not isinstance(sens_1_vals[idx_1], str):
            cell_0_not_sens_idx.append(idx_1)
    df = df.iloc[cell_0_not_sens_idx]

    spock_model_refs = df.model_ref.values

    return(spock_model_refs)


def plot_spock_acceptance_ratio(model_space_report_path, model_ref_path, output_img_name):
    df = pd.read_csv(model_space_report_path)

    models = df.model_idx.values
    accepted_count = df.accepted_count.values
    simulated_count = df.simulated_count.values

    acceptance_ratio = []

    for m in models:
        acceptance_ratio.append(accepted_count[m] / simulated_count[m])

    df['acceptance_ratio'] = acceptance_ratio

    spock_idx_list = find_spock(model_ref_path)

    df = df.loc[df['model_idx'].isin(spock_idx_list)]
    models = df.model_idx.values
    print(df)
    print(models)

    plt.bar(models, df.acceptance_ratio, width=1)
    plt.xlabel('Model reference')
    plt.ylabel(' #accepted / #simulated')
    # plt.savefig(output_img_name, dpi=300)
    plt.show()


def plot_all_acceptance_ratios(model_space_report_path, output_img_name):
    df = pd.read_csv(model_space_report_path)

    models = df.model_idx.values
    accepted_count = df.accepted_count.values
    simulated_count = df.simulated_count.values

    acceptance_ratio = []

    for m in models:
        acceptance_ratio.append(accepted_count[m] / simulated_count[m])

    df['acceptance_ratio'] = acceptance_ratio
    plt.bar(models, df.acceptance_ratio, width=1)
    plt.xlabel('Model reference')
    plt.ylabel(' #accepted / #sampled')
    plt.savefig(output_img_name, dpi=300)

    plt.show()


def plot_above_threshold(model_space_report_path, acceptance_threshold, output_img_name):
    df = pd.read_csv(model_space_report_path)

    models = df.model_idx.values
    accepted_count = df.accepted_count.values
    simulated_count = df.simulated_count.values

    acceptance_ratio = []

    for m in models:
        acceptance_ratio.append(accepted_count[m] / simulated_count[m])

    df['acceptance_ratio'] = acceptance_ratio
    df = df.loc[df['acceptance_ratio'] > acceptance_threshold]
    print(len(df))

    print(df)
    print(sum(df.simulated_count.values))

    models = df.model_idx.values
    plt.bar(models, df.acceptance_ratio, width=1)
    plt.xlabel('Model reference')
    plt.ylabel(' #accepted / #sampled')
    plt.savefig(output_img_name, dpi=300)

    plt.show()


def acceptance_ratio_hist(model_space_report_path, bins, output_img_name):
    df = pd.read_csv(model_space_report_path)

    models = df.model_idx.values
    accepted_count = df.accepted_count.values
    simulated_count = df.simulated_count.values

    acceptance_ratio = []

    for m in models:
        acceptance_ratio.append(accepted_count[m] / simulated_count[m])

    df['acceptance_ratio'] = acceptance_ratio

    print(sum(df.simulated_count.values))
    print(np.histogram(df.acceptance_ratio, bins))
    plt.hist(df.acceptance_ratio, bins=bins)
    plt.title('Histogram of acceptance_ratios. Bins = ' + str(bins))
    plt.ylabel('Frequency')
    plt.xlabel('acceptance_ratio')
    plt.savefig(output_img_name, dpi=300)

    plt.show()


def two_D_parameter_plots(model_ref, input_dir, accepted_params_dir):
    output_name = "model_IDX_param_posterior.png".replace('IDX', str(model_ref))

    input_params_file = input_dir + \
        "params_IDX.csv".replace("IDX", str(model_ref))
    accepted_params_file = accepted_params_dir + \
        "model_IDX_accepted_params".replace('IDX', str(model_ref))

    # Load input file
    input_file_columns = list(['param_name', 'lwr_bound', 'upr_bound'])
    input_df = pd.read_csv(
        input_params_file, names=input_file_columns, header=None)

    # Load parameters list
    param_names_list = input_df.param_name.values.tolist()
    print(param_names_list)

    plot_params = ['D', 'mu_max_x', 'mu_max_c']
    # plot_params = ['D', 'kBmax_mccB', 'kBmax_mccV']

    accepted_params_df = pd.read_csv(accepted_params_file, names=param_names_list, header=None, index_col=None)
    accepted_params_df = accepted_params_df.reset_index()

    plt.rcParams.update({'font.size': 8})
    fig, axes = plt.subplots(len(plot_params), len(
        plot_params), figsize=(30, 15))

    fig.suptitle('Model_' + str(model_ref), fontsize=20)
    for idx_1, p_name_1 in enumerate(plot_params):
        for idx_2, p_name_2 in enumerate(plot_params):
            if idx_1 == idx_2:
                continue

            input_row_param_1 = input_df.loc[input_df['param_name'] == p_name_1]
            input_row_param_2 = input_df.loc[input_df['param_name'] == p_name_2]

            p_1_upr = input_row_param_1['upr_bound'].values[0]
            p_1_lwr = input_row_param_1['lwr_bound'].values[0]

            p_2_upr = input_row_param_2['upr_bound'].values[0]
            p_2_lwr = input_row_param_2['lwr_bound'].values[0]

            if p_1_upr == p_1_lwr:
                continue

            if p_2_upr == p_2_lwr:
                continue
            p_1_values = accepted_params_df[p_name_1].values[:100]
            p_2_values = accepted_params_df[p_name_2].values[:100]

            axes[idx_1, idx_2].set_xlim(p_1_lwr, p_1_upr)
            axes[idx_1, idx_2].set_ylim(p_2_lwr, p_2_upr)
            axes[idx_1, idx_2].set_xlabel(p_name_1, fontdict={'fontsize': 15})
            axes[idx_1, idx_2].set_ylabel(p_name_2, fontdict={'fontsize': 15})

            axes[idx_1, idx_2].scatter(p_1_values, p_2_values, s=1.5)


    plt.savefig(output_name, dpi=300)
    # plt.show()

def growth_distribution(model_ref, input_dir, accepted_params_dir):
    output_name = "model_IDX_growth_ratio.png".replace('IDX', str(model_ref))
    input_params_file = input_dir + \
    "params_IDX.csv".replace("IDX", str(model_ref))
    accepted_params_file = accepted_params_dir + \
        "model_IDX_accepted_params".replace('IDX', str(model_ref))

    # Load input file
    input_file_columns = list(['param_name', 'lwr_bound', 'upr_bound'])
    input_df = pd.read_csv(
        input_params_file, names=input_file_columns, header=None)

    param_names_list = input_df.param_name.values.tolist()
    print(param_names_list)

    accepted_params_df = pd.read_csv(accepted_params_file, names=param_names_list, header=None, index_col=None)
    accepted_params_df = accepted_params_df.reset_index()

    # Load parameters list

    plot_params = ['D', 'mu_max_x', 'mu_max_c']

    s_1_max_growth = accepted_params_df['mu_max_x'].values
    s_2_max_growth = accepted_params_df['mu_max_c'].values

    s1_s2_max_growth_ratio = []
    for idx, i in enumerate(s_1_max_growth):
        s1_s2_max_growth_ratio.append(s_1_max_growth[idx]/s_2_max_growth[idx])

    plt.hist(s1_s2_max_growth_ratio, bins=50)
    plt.xlabel('s1 max_growth / s2 max_growth')
    plt.ylabel('Frequency')
    plt.title('Model_' + str(model_ref), fontsize=20)
    plt.savefig(output_name, dpi=300)
    plt.cla()

def plot_accepted_parameters(model_ref, input_dir, accepted_params_dir, bins):
    input_params_file = input_dir + \
        "params_IDX.csv".replace("IDX", str(model_ref))
    accepted_params_file = accepted_params_dir + \
        "model_IDX_accepted_params".replace('IDX', str(model_ref))

    # Load input file
    input_file_columns = list(['param_name', 'lwr_bound', 'upr_bound'])
    input_df = pd.read_csv(
        input_params_file, names=input_file_columns, header=None)

    # Load parameters list
    param_names_list = input_df.param_name.values.tolist()

    accepted_params_df = pd.read_csv(
        accepted_params_file, names=param_names_list, header=None, index_col=None)
    accepted_params_df = accepted_params_df.reset_index()

    output_img_name = "model_IDX_PARAM_posterior.png".replace(
        'IDX', str(model_ref))

    for p_name in param_names_list:
        input_param_row = input_df.loc[input_df['param_name'] == p_name]

        param_upr_bound = input_param_row['upr_bound'].values[0]
        param_lwr_bound = input_param_row['lwr_bound'].values[0]
        temp_name = output_img_name.replace('PARAM', p_name)

        # Skip non-varied parameters
        if param_upr_bound == param_lwr_bound:
            continue

        else:
            weights = np.ones_like(
                accepted_params_df[p_name].values) / float(len(accepted_params_df))
            plt.title("Accepted parameter hist")
            plt.hist(accepted_params_df[p_name].values,
                     bins=bins, weights=weights)
            plt.xlim(param_lwr_bound, param_upr_bound)
            plt.xlabel(p_name)
            plt.ylabel('Relative density')
            plt.savefig(temp_name, dpi=300)
            plt.show()


if __name__ == "__main__":
    wd = "/home/behzad/myriad_home/Scratch/cpp_consortium_sim/model_sel_doe/"
    wd = "/home/behzad/Documents/barnes_lab/cplusplus_software/speed_test/repressilator/cpp/"
    model_space_report_path = wd + \
        "/output/two_species_big_0/Population_0/model_space_report.csv"
    model_ref_path = '/home/behzad/Documents/barnes_lab/sympy_consortium_framework/output/two_species_no_symm/model_ref.csv'

    accepted_params_dir = wd + \
        "/output/two_species_big_0/Population_0/model_accepted_params/"
    inputs_dir = wd + "/input_files_two_species/"

    report_summary(model_space_report_path)

    plot_spock_acceptance_ratio(model_space_report_path, model_ref_path, "test")

    plot_all_acceptance_ratios(model_space_report_path, "all_models.png")
    plot_above_threshold(model_space_report_path, 0.4, "top_models.png")
    acceptance_ratio_hist(model_space_report_path, 20, "acceptance_ratio_hist.png")
    # plot_accepted_parameters(104, inputs_dir, accepted_params_dir, 25)

    # growth_distribution(41, inputs_dir, accepted_params_dir)
    # growth_distribution(5, inputs_dir, accepted_params_dir)
    # growth_distribution(88, inputs_dir, accepted_params_dir)
    # growth_distribution(104, inputs_dir, accepted_params_dir)
    # growth_distribution(106, inputs_dir, accepted_params_dir)

    # two_D_parameter_plots(5, inputs_dir, accepted_params_dir)
    # two_D_parameter_plots(41, inputs_dir, accepted_params_dir)
    # two_D_parameter_plots(88, inputs_dir, accepted_params_dir)
    # two_D_parameter_plots(104, inputs_dir, accepted_params_dir)
    # two_D_parameter_plots(106, inputs_dir, accepted_params_dir)

