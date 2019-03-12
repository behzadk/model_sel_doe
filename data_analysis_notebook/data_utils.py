import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import pandas as pd

def recur_get_connected_nodes(adj_mat, path_length, path_sign, visited_nodes, from_idx, target_idx, loop_found, path_log, output):
    n_species = np.shape(adj_mat)[0]
    neighbours = [i for i in range(n_species) if adj_mat[i, from_idx] != 0]
    
    if loop_found:
        return 0

    if from_idx is target_idx:
            loop_found = True
            output.append([path_sign, path_length])
            return 0

    for n in neighbours:
        if n not in visited_nodes:
            path_log.append(n)
            visited_nodes.append(n)
            path_length += 1
            path_sign = path_sign * adj_mat[n, from_idx]
            recur_get_connected_nodes(adj_mat, path_length, path_sign, visited_nodes, n, target_idx, loop_found, path_log, output)

    return output


# Counts feedback existing between all.
# 
def get_feedback_loops(adj_mat_df):
    # Drop row names column
    adj_mat_df.drop([adj_mat_df.columns[0]], axis=1, inplace=True)

    col_names = adj_mat_df.columns
    strain_indexes = [idx for idx, i in enumerate(col_names) if 'N_' in i]
    adj_mat = adj_mat_df.values

    # Remove column 0, which contains row names
    # adj_mat = adj_mat[:, 1:]
    n_species = np.shape(adj_mat)[0]

    output = []
    for strain_idx in strain_indexes:

        for row_idx in range(n_species):

            # Find strain to species interaction
            if adj_mat[row_idx, strain_idx] != 0:

                loop_closed = False
                path_length = 1

                from_idx = row_idx
                path_log = [strain_idx, row_idx]
                path_sign = adj_mat[row_idx, strain_idx]
                strain_feedback_loops = recur_get_connected_nodes(adj_mat, path_length, path_sign, [row_idx], from_idx, strain_idx, False, path_log, [])

                for l in strain_feedback_loops:
                    output.append(l)

    return output


def get_num_parts(adj_mat_df):
    # Drop row names column
    adj_mat_df.drop([adj_mat_df.columns[0]], axis=1, inplace=True)

    col_names = adj_mat_df.columns
    strain_indexes = [idx for idx, i in enumerate(col_names) if 'N_' in i]
    adj_mat = adj_mat_df.values

    n_species = np.shape(adj_mat)[0]
    num_parts = 0
    for strain_idx in strain_indexes:
        num_parts += sum(adj_mat[:, strain_idx])

    return num_parts

def make_num_parts(model_space_report_df, adj_matrix_path_template):
    models = model_space_report_df.model_idx.values
    all_num_parts = []
    for m in models:
        adj_mat_path = adj_matrix_path_template.replace("#REF#", str(m))
        adj_mat_df = pd.read_csv(adj_mat_path)
        num_parts = get_num_parts(adj_mat_df)
        all_num_parts.append(num_parts)

    return all_num_parts


def make_feedback_loop_counts(model_space_report_df, adj_matrix_path_template):
    models = model_space_report_df.model_idx.values
    positive_loops = []
    negative_loops = []

    for m in models:
        adj_mat_path = adj_matrix_path_template.replace("#REF#", str(m))
        adj_mat_df = pd.read_csv(adj_mat_path)
        feedback_loops = get_feedback_loops(adj_mat_df)
        positive_count = 0
        negative_count = 0

        for loop in feedback_loops:
            loop_sign = loop[0]
            loop_length = loop[1]

            if  loop_sign == 1:
                positive_count += 1/loop_length

            if loop_sign == -1:
                negative_count += 1/loop_length

            if loop_sign != 1 and loop_sign != -1:
                print("Loop sign is wrong: ", loop_sign)

        positive_loops.append(positive_count)
        negative_loops.append(negative_count)


    return positive_loops, negative_loops






def generate_replicates_and_std(all_sims_df, model_space_report_df, num_replicates):
    models = model_space_report_df.model_idx.values
    batches = all_sims_df.batch_num.unique()
    batch_sim_indexes = all_sims_df.sim_idx.unique()

    total_simulations = len(all_sims_df)

    all_sims_df['replicate'] = pd.qcut(
        range(total_simulations), num_replicates, labels=np.arange(num_replicates))

    # Create replicate subset
    for rep_num in range(num_replicates):
        rep_subset_df = all_sims_df.loc[all_sims_df['replicate'] == rep_num]
        print("Replicate: ", rep_num, " ",
              " Simulations: ", len(rep_subset_df.index))
        replicate_acceptance_ratios = []

        # Calculate acceptance ratio for each model within a subset
        for m in models:
            models_subset_df = rep_subset_df.loc[rep_subset_df['model_ref'] == m]
            num_accepted = len(
                models_subset_df.loc[models_subset_df['Accepted'] == True].index)
            num_rejected = len(
                models_subset_df.loc[models_subset_df['Accepted'] == False].index)

            try:
                acceptance_ratio = num_accepted / (num_rejected + num_accepted)
            except(ZeroDivisionError):
                acceptance_ratio = np.NaN

            replicate_acceptance_ratios.append(acceptance_ratio)

        replicate_name = 'replicate_NUM'.replace('NUM', str(rep_num))
        model_space_report_df[replicate_name] = replicate_acceptance_ratios

    # Calculate standard deviation between replicates
    model_standard_deviations = []
    for m in models:
        model_subset_df = model_space_report_df.loc[model_space_report_df['model_idx'] == m]
        model_replicate_acceptance_ratios = []

        for rep_num in range(num_replicates):
            replicate_name = 'replicate_NUM'.replace('NUM', str(rep_num))
            replicate_ratio = model_subset_df[replicate_name].values[0]
            model_replicate_acceptance_ratios.append(replicate_ratio)

        model_standard_deviations.append(
            np.std(model_replicate_acceptance_ratios))

    model_space_report_df['stdev'] = model_standard_deviations

    sims_per_replicate = len(all_sims_df.loc[all_sims_df['replicate'] == 0])

    return model_space_report_df, sims_per_replicate




##
# Adds new bool column for species 1 and 2 indicating whether the species was susitaned or not
# by the end of the simulation. Uses a threshold where above that number, the species is sustained.
#
##
def species_sustained(df, threshold=1e3):

    N_1_final = df.d3
    N_1_sustained = []

    for i in N_1_final:
        if i >= threshold and i < 1e100:
            N_1_sustained.append(True)

        else:
            N_1_sustained.append(False)

    N_2_final = df.d6
    N_2_sustained = []

    for i in N_2_final:
        if float(i) >= threshold and i < 1e100:
            N_2_sustained.append(True)

        else:
            N_2_sustained.append(False)

    df['N_1_sustained'] = N_1_sustained
    df['N_2_sustained'] = N_2_sustained

    return df


def make_max_eig(df):
    # Make subset containing only eigenvalues
    col_names = df.columns
    eign_cols = [x for x in col_names if 'eig' in x]
    eign_cols = [x for x in eign_cols if 'real' in x]

    eig_df = df[eign_cols]
    max_real_eigs = []

    for row in eig_df.values:
        abs_vals = [abs(x) for x in row]
        try:
            max_val_idx = np.nanargmax(abs_vals)
            max_real_eigs.append(row[max_val_idx])

        except(ValueError):
            max_real_eigs.append(np.nan)

    # print(max_real_eigs)
    max_real_eigs

    return max_real_eigs

def make_sum_eig(df):
    # Make subset containing only eigenvalues
    col_names = df.columns
    eign_cols = [x for x in col_names if 'eig' in x]
    eign_cols = [x for x in eign_cols if 'real' in x]

    eig_df = df[eign_cols]
    all_sum_eigs = []

    for row in eig_df.values:
        sum_eig = 0
        for val in row:
            if pd.isnull(val):
                continue

            else:
                sum_eig = sum_eig + val

        all_sum_eigs.append(sum_eig)

    # print(max_real_eigs)
    all_sum_eigs

    return all_sum_eigs


##
# Adds a column noting whether all the real part eigenvalues are negative
#
##
def all_negative_eigs(df):
    # Make subset containing only eigenvalues
    col_names = df.columns
    eign_cols = [x for x in col_names if 'eig' in x]
    eign_cols = [x for x in eign_cols if 'real' in x]
    eig_df = df[eign_cols]
    all_negative = []

    for row in eig_df.values:
        row_negative = True
        for val in row:
            if pd.isnull(val):
                continue

            elif val < 0:
                continue

            else:
                row_negative = False

        all_negative.append(row_negative)

    # print(max_real_eigs)
    all_negative

    return all_negative

##
# Adds a column noting whether all the real part eigenvalues are positive
#
##
def all_positive_eigs(df):
    # Make subset containing only eigenvalues
    col_names = df.columns
    eign_cols = [x for x in col_names if 'eig' in x]
    eign_cols = [x for x in eign_cols if 'real' in x]
    eig_df = df[eign_cols]
    all_positive = []

    for row in eig_df.values:
        row_positive = True
        for val in row:
            if pd.isnull(val):
                continue

            elif val > 0:
                continue

            else:
                row_positive = False

        all_positive.append(row_positive)

    # print(max_real_eigs)
    all_positive

    return all_positive


def all_real_eigs(df):
    # Make subset containing only eigenvalues
    col_names = df.columns
    eign_cols = [x for x in col_names if 'eig' in x]
    eign_cols = [x for x in eign_cols if 'imag' in x]

    eig_df = df[eign_cols]
    all_real = []

    for row in eig_df.values:
        is_real = True

        for val in row:
            if val == 0:
                continue

            elif pd.isnull(val):
                continue

            else:
                is_real = False

        # print(is_real, ": ", row)
        all_real.append(is_real)

    # print(max_real_eigs)
    all_real

    return all_real

def all_zero_eigs(df):
    # Make subset containing only eigenvalues
    col_names = df.columns
    eign_cols = [x for x in col_names if 'eig' in x]
    real_part_cols = [x for x in eign_cols if 'real' in x]

    eig_df = df[real_part_cols]
    print(eig_df.columns)
    all_zero = []

    for row in eig_df.values:
        is_zero = True

        for val in row:
            if val == 0:
                continue

            elif pd.isnull(val):
                continue

            else:
                is_zero = False

        # print(is_real, ": ", row)
        all_zero.append(is_zero)

    # print(max_real_eigs)
    all_zero

    return all_zero


##
#   Finds conjugate pairs of eigenvalues from a list format of [ [real, imaginary], [real, imaginary] ]
#
#   Returns indexes of eigenvalues which are pairs
##
def get_conjugate_pairs(df):
    eign_cols = [x for x in df.columns if 'eig' in x]
    real_part_cols = [x for x in eign_cols if 'real' in x]
    imag_part_cols = [x for x in eign_cols if 'imag' in x]

    df_num_conj_pairs = []
    for idx, row in df.iterrows():
        real_parts = row[real_part_cols]
        imag_parts = row[imag_part_cols].values

        # Find matching real parts
        real_part_pairs = []
        for idx, i in enumerate(real_parts):
            if np.isnan(i):
                continue

            matches = np.equal(real_parts, i)
            match_indexes = [i for i, x in enumerate(matches) if x]

            # Add pairs to array
            if len(match_indexes) == 2 and match_indexes not in real_part_pairs:
                real_part_pairs.append(match_indexes)

        conjugate_pairs = []
        for p in real_part_pairs:
            i_0 = imag_parts[p[0]]
            i_1 = imag_parts[p[1]]

            if i_0 == 0 and i_1 == 0:
                continue

            if i_0 == i_1 * -1:
                conjugate_pairs.append(p)

        df_num_conj_pairs.append(len(conjugate_pairs))

    return df_num_conj_pairs

def set_number_imaginary(df):
    eign_cols = [x for x in df.columns if 'eig' in x]
    real_part_cols = [x for x in eign_cols if 'real' in x]
    imag_part_cols = [x for x in eign_cols if 'imag' in x]

    df_num_imag = []
    for idx, row in df.iterrows():
        imag_parts = row[imag_part_cols].values
        imag_count = 0
        for idx, i in enumerate(imag_parts):
            if np.isnan(i):
                continue

            if i == 0.0:
                continue

            else:
                imag_count += 1

        df_num_imag.append(imag_count)

    return df_num_imag








def distances_pre_processing(df):
    col_names = df.columns
    dist_cols = ['d1', 'd2', 'd3', 'd4', 'd5', 'd6']

    for c in dist_cols:
        df[c] = [np.float128(i) for i in df[c].values]

    return df


def set_accepted_column(df):
    col_names = df.columns
    dist_cols = ['d1', 'd2', 'd3', 'd4', 'd5', 'd6']

    df.loc[df['d1']]

def main():
    output_dir= "./output/two_species_big_3/Population_0"
    distances_path= output_dir + "/distances.csv"
    eigenvalues_path= output_dir + "/eigenvalues.csv"
    model_space_report_path= output_dir + "/model_space_report.csv"

    distaces_df= pd.read_csv(distances_path)
    distaces_df= distances_pre_processing(distaces_df)

    eigenvalues_df= pd.read_csv(eigenvalues_path)

    # Ignore failed simulations
    eigenvalues_df= eigenvalues_df.loc[eigenvalues_df['integ_error'].isnull()]

    joint_df= pd.merge(left=eigenvalues_df, right=distaces_df, how='inner', on=['sim_idx', 'batch_num', 'model_ref'])
    joint_df.reset_index()

    joint_df= species_sustained(joint_df)
    joint_df= make_max_eig(joint_df)
    joint_df= all_negative_eigs(joint_df)
    joint_df= all_real_eigs(joint_df)

    print(len(joint_df))
    print(joint_df.columns)

    loc_all_negative
    # loc_all_negative = joint_df.loc[joint_df['all_negative_eigs'] == True]
    print(len(loc_all_negative))
    loc_all_negative['sum_std']= loc_all_negative['d2'] + loc_all_negative['d3']

    # loc_all_negative = loc_all_negative.loc[loc_all_negative['N_1_sustained'] == True]
    # loc_all_negative = loc_all_negative.loc[loc_all_negative['N_2_sustained'] == True]

    all_real= loc_all_negative.loc[loc_all_negative['all_real_eigs'] == True]
    not_real= loc_all_negative.loc[loc_all_negative['all_real_eigs'] == False]
    print(len(all_real))
    exit()
    ax= sns.scatterplot(x='sum_std', y='max_eig', data=all_real)
    ax= sns.scatterplot(x='sum_std', y='max_eig', data=not_real)

    ax.set(xscale="symlog", yscale='symlog')

    plt.plot()
    plt.show()


if __name__ == "__main__":
    adj_mat_dir = "/home/behzad/Documents/barnes_lab/sympy_consortium_framework/output/two_species_no_symm/adj_matricies/model_0_adj_mat.csv"
    adj_mat_df = pd.read_csv(adj_mat_dir)
    o = get_feedback_loops(adj_mat_df)
    print(o)
