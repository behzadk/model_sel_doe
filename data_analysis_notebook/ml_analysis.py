import numpy as np
import pandas as pd
import posterior_analysis
import matplotlib.style as style
import math

# Load scikit's random forest classifier library
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import random
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy import spatial
import scipy
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.cluster import KMeans
from sklearn.cluster import OPTICS
from sklearn.cluster import SpectralClustering

from scipy.cluster.hierarchy import dendrogram, linkage
import data_utils
import data_analysis_ABCSMC
# import sklearn
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from sklearn.neural_network import MLPRegressor
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score

import motif_counting

if 1:
    wd = "/home/behzad/Documents/barnes_lab/cplusplus_software/speed_test/repressilator/cpp/"
    experiment_name = "two_species_stable_0_SMC"
    inputs_dir = wd + "/input_files/input_files_two_species_0/"
    R_script = "plot-motifs-two.R"


def k_means_test():
    combined_analysis_output_dir = wd + "output/" + experiment_name + "/experiment_analysis/"
    data_utils.make_folder(combined_analysis_output_dir)
    adj_mat_dir = inputs_dir + "adj_matricies/"

    exp_dir = wd + "output/" + experiment_name + "/"
    clean_exp_repeat_dirs = []
    final_pop_dirs = []
    first_pop_dirs = []
    finished_exp_final_population_dirs = data_analysis_ABCSMC.find_finished_experiments(exp_dir)
    model_space_report_list = []

    for data_dir in finished_exp_final_population_dirs:
        # Load model space report
        model_space_report_path = data_dir + "model_space_report.csv"
        model_space_report_df = pd.read_csv(model_space_report_path, index_col=0)
        model_space_report_list.append(model_space_report_df)
    
    model_space_report_df = data_analysis_ABCSMC.merge_model_space_report_df_list(model_space_report_list)
    data_analysis_ABCSMC.generate_model_space_statistics(model_space_report_df, "model_marginal")
    model_space_report_df = model_space_report_df.sort_values(by='model_marginal_mean', ascending=False).reset_index(drop=True)

    model_idxs = model_space_report_df['model_idx'].values

    strain_loop_bal, sum_pos_loops, sum_neg_loops, paths_log_out = data_utils.make_strain_feedback_loop_balance(model_idxs, adj_mat_dir)

    model_space_report_df['symmetrical'] = [4 if x == True else 0 for x in  data_utils.get_two_strain_symmetric_adj_mats(model_idxs, adj_mat_dir)]
    symm_models = model_space_report_df.loc[model_space_report_df['symmetrical'] == 4]['model_idx'].values
    symm_models = [symm_models[0]]

    sym_col_names = []
    for m in symm_models:
        col_name = str(m) + "_dist"
        sym_col_names.append(col_name)
        model_space_report_df[col_name] = data_utils.get_adj_mat_distances(m, model_idxs, adj_mat_dir)
    
    min_symm_neighbour = []
    min_dist_val = []
    for m_idx in model_idxs:
        min_symm_vals = model_space_report_df.loc[model_space_report_df['model_idx'] == m_idx][sym_col_names].values[0]
        min_symm_neighbour.append(int(sym_col_names[np.argmin(min_symm_vals)].split('_')[0]))
        min_dist_val.append(min(model_space_report_df.loc[model_space_report_df['model_idx'] == m_idx][sym_col_names].values[0]))

    model_space_report_df['min_symm_neighbour'] = min_symm_neighbour
    model_space_report_df['min_symm_dist'] = min_dist_val

    unique_interaction_paths = []

    for p in paths_log_out:
        for sub_p in p:
            if sub_p not in unique_interaction_paths:
                print(sub_p)
                if len(sub_p) > 0:
                    unique_interaction_paths.append(sub_p)

    model_endcoded_path_possessions = []
    for idx, m_idx in enumerate(model_idxs):
        model_paths = paths_log_out[idx]
        encoded_paths = [0 for x in range(len(unique_interaction_paths))]
        for p_idx, unqe_p in enumerate(unique_interaction_paths):
            if unqe_p in model_paths:
                encoded_paths[p_idx] = 1
        model_endcoded_path_possessions.append(encoded_paths)

    stacked_encoded_paths = np.column_stack(model_endcoded_path_possessions)
    print(np.shape(stacked_encoded_paths))

    unqe_p_cols = ["path_" + str(x) for x in range(len(unique_interaction_paths))]

    # linked = linkage(X, 'single', optimal_ordering=True)
    # labels = y['model_idx'].values

    # print(np.shape(linkage))
    # print(np.shape(labels))

    # plt.figure(figsize=(10, 7))
    # d = dendrogram(linked,
    #             orientation='top',
    #             labels=labels,
    #             distance_sort='descending',
    #             show_leaf_counts=True)
    X = stacked_encoded_paths.T
    X_label = model_space_report_df['model_marginal_mean'].values
    cluster = AgglomerativeClustering(n_clusters=7, affinity='euclidean', linkage='ward')
    cluster.fit_predict(X)

    sns.scatterplot(min_dist_val, X_label, hue=min_symm_neighbour, 
                palette=sns.color_palette("Set1", n_colors=len(symm_models)))
    plt.show()


def find_nearest_neighbours_alt(current_model_idx, all_states, tree, visited_indexes):
    neighbour_distances, neighbour_indexes = tree.query(all_states[current_model_idx], k=10)
    clean_distances = []
    clean_neighbour_indexes = []
    for idx, model_idx in enumerate(neighbour_indexes):
        if model_idx not in visited_indexes:
            clean_neighbour_indexes.append(model_idx)
            clean_distances.append(neighbour_distances[idx])

    clean_distances = clean_distances[1:]
    if len(clean_distances) == 0:
        return []
    min_distance_indexes = np.where(clean_distances == np.min(clean_distances))[0]

    min_neigbour_indexes = [clean_neighbour_indexes[int(idx)] for idx in min_distance_indexes]

    return min_neigbour_indexes

def get_degree_matrix(adj_mat):
    degree_mat = np.zeros(np.shape(adj_mat))

    for x in range(np.shape(adj_mat)[0]):
        degree_mat[x, x] = sum(map(abs, adj_mat[:, x])) + sum(map(abs, adj_mat[x,]))

    return degree_mat


def adj_mat_random_walk(inputs_dir, data_dir, output_dir):
    model_space_report_path = data_dir + "model_space_report.csv"
    adj_mat_name_template = "model_#REF#_adj_mat.csv"
    adj_matrix_path_template = inputs_dir + "adj_matricies/" + adj_mat_name_template


    model_space_report_df = pd.read_csv(model_space_report_path)
    total_accepted = sum(model_space_report_df['accepted_count'].values)


    # Calculate acceptance ratio for each model
    model_space_report_df['acceptance_ratio'] = model_space_report_df.apply(lambda row: row['accepted_count'] / total_accepted, axis=1)

    species_names = [ 'N_1', 'N_2', 'S_glu', 'B_1', 'B_2', 'A_1', 'A_2']
    from_too_features_list = ['model_idx', 'posterior_probability']

    for s_1 in species_names:
        for s_2 in species_names:
            from_too_features_list.append(s_2 + '_' + s_1)
    
    # Generate features data frame
    flat_features_df = pd.DataFrame(columns=from_too_features_list)
    encoded_models = []

    for model_idx in model_space_report_df['model_idx'].values:
        adj_mat_path = adj_matrix_path_template.replace("#REF#", str(model_idx))
        adj_mat_df = pd.read_csv(adj_mat_path)
        adj_mat_df.drop([adj_mat_df.columns[0]], axis=1, inplace=True)

        model_posterior_prob = model_space_report_df.loc[model_space_report_df['model_idx'] == model_idx]['acceptance_ratio'].values[0]

        model_features = [model_idx, model_posterior_prob]
        model_features = model_features + list(adj_mat_df.values.ravel())
        flat_features_df.loc[model_idx] = model_features


        encoded_models.append(list(adj_mat_df.values.ravel()))

    all_states = flat_features_df.loc[:, ~flat_features_df.columns.isin(['model_idx', 'posterior_probability'])].values
    all_model_idxs = flat_features_df['model_idx'].values

    simple_state = [sum(map(abs, x)) for x in all_states]

    posterior_prob_list = []
    visited_indexes = []
    tree = spatial.KDTree(all_states)
    z = 0
    while z < 4:
        current_model_idx = 194

        z +=1
        current_posterior_prob = flat_features_df.loc[current_model_idx]['posterior_probability']
        posterior_prob_list = []
        visited_indexes = []

        for i in range(100):
            current_posterior_prob = flat_features_df.loc[current_model_idx]['posterior_probability']
            posterior_prob_list.append(current_posterior_prob)
            visited_indexes.append(current_model_idx)

            nearest_neighbours = find_nearest_neighbours_alt(current_model_idx=current_model_idx, all_states=all_states, tree=tree, visited_indexes=visited_indexes)        

            if len(nearest_neighbours) == 0:
                break

            current_model_idx = random.choice(nearest_neighbours)
            plt.plot(range(len(posterior_prob_list)), posterior_prob_list)

    plt.show()


def convert_adj_mat(adj_df):

    adj_df = adj_d
    # for i, row in adj_df.iterrows():
    #     if row['A_1'] == -1:
    #         adj_df.at[i, 'A_1'] = 2
    #     if row['A_2'] == -1:
    #         adj_df.at[i, 'A_2'] = 2

        # if row['A_3'] == -1:
        #     adj_df.at[i, 'A_3'] = 2

    return adj_df

def norm_signed_laplacian(adj_mat, deg_mat):
    norm_lap = np.zeros(np.shape(adj_mat))

    for u, _ in enumerate(range(np.shape(norm_lap)[0])):
        for v, _ in enumerate(range(np.shape(norm_lap)[0])):
            if u == v and sum(deg_mat[:, v]) != 0:
                norm_lap[u][v] = 1

            else:
                norm_lap[u][v] = -adj_mat[u][v] * np.sqrt(sum(deg_mat[:, u]) * sum(deg_mat[:, v]))


    return norm_lap



def adj_mat_spectral_cluster(inputs_dir, data_dir, output_dir):
    # https://www.cs.cmu.edu/~aarti/Class/10701/slides/Lecture21_2.pdf
    model_space_report_path = data_dir + "model_space_report.csv"
    adj_mat_name_template = "model_#REF#_adj_mat.csv"
    adj_matrix_path_template = inputs_dir + "adj_matricies/" + adj_mat_name_template

    model_space_report_df = pd.read_csv(model_space_report_path)
    total_accepted = sum(model_space_report_df['accepted_count'].values)

    # Calculate acceptance ratio for each model
    model_space_report_df['acceptance_ratio'] = model_space_report_df.apply(lambda row: row['accepted_count'] / total_accepted, axis=1)

    models_list = []
    for model_idx in model_space_report_df['model_idx'].values:
        adj_mat_path = adj_matrix_path_template.replace("#REF#", str(model_idx))
        adj_mat_df = pd.read_csv(adj_mat_path)
        # adj_mat_df = convert_adj_mat(adj_mat_df)
        adj_mat_df.drop([adj_mat_df.columns[0]], axis=1, inplace=True)

        model_posterior_prob = model_space_report_df.loc[model_space_report_df['model_idx'] == model_idx]['acceptance_ratio'].values[0]
        # has_negative = False
        # for v in adj_mat_df['A_1'].values:
        #     if v < 0:
        #         has_negative = True
        # for v in adj_mat_df['A_2'].values:
        #     if v < 0:
        #         has_negative = True

        # if has_negative:
        #     continue

        model_adj_mat = (adj_mat_df.values)
        # model_adj_mat = model_adj_mat + 1
        model_adj_mat = abs(model_adj_mat)
        degree_mat = get_degree_matrix(model_adj_mat)

        identity_mat = np.identity(np.shape(degree_mat)[0])

        laplacian_mat = degree_mat - model_adj_mat

        first_term = identity_mat
        second_term = np.power(degree_mat, -0.5)
        second_term[second_term == np.inf] = 1
        third_term = model_adj_mat * second_term
        
        norm_laplacian = first_term - second_term * third_term
        # print(norm_laplacian)

        # norm_laplacian = norm_signed_laplacian(model_adj_mat, degree_mat)
        # print(norm_laplacian)

        eig_vals, eig_vecs = np.linalg.eig(model_adj_mat)

        eig_vecs = [x for _, x in sorted(zip(eig_vals, eig_vecs), key=lambda pair: pair[0])]

        lowest_non_zero_eig = 0
        
        eig_vec_order = np.argsort(eig_vals)

        # print(eig_vals)
        second_lowest_eig_val_arg = np.argsort(eig_vals)

        for e_idx in second_lowest_eig_val_arg:
            if eig_vals[e_idx] != 0:
                second_lowest_eig_val_arg = e_idx
                break

        second_lowest_eig_vec = eig_vecs[0]
        second_lowest_eig_val = eig_vals[0]

        eig_vals = [e.real for e in eig_vals]

        clean_eig_vecs = []
        for f in eig_vecs:
            new_f = []
            for val in f:
                new_f.append(val.real)

            clean_eig_vecs.append(new_f)


        # if median_val == 0:
        #     print(eig_vals[1])
        #     print(second_lowest_eig_vec)
        #     print(model_adj_mat)
        #     exit()
        model_dict = {'model_idx': model_idx, 'posterior_prob': model_posterior_prob, 'adj_mat': model_adj_mat, 
        'laplacian_mat': norm_laplacian, 'eig_vec': second_lowest_eig_vec, 'eig_vals': np.sort(eig_vals), 'eig_vecs':clean_eig_vecs}

        models_list.append(model_dict)

    all_eig_vals = []
    all_posterior_probs = []


    # # Make new pdf
    # pdf = PdfPages(out_path)

    # with PdfPages(out_path) as pdf:

    #     for model in models_list:
    #         plt.scatter(range(len(model['eig_vals'])), np.sort(model['eig_vals']))
    #         pdf.savefig()
    #         plt.close()
    # pdf.close()

    colours = []

    models_list = sorted(models_list, key=lambda m: m['posterior_prob'], reverse=True)

    eigen_gaps = []
    for m in models_list:
        model_eig_gaps = []
        for idx, e in enumerate(m['eig_vals'][1:]):
            model_eig_gaps.append(e - m['eig_vals'][idx-1])
        eigen_gaps.append(model_eig_gaps)


    eigen_vecs = []
    for m in models_list:
        model_eig_vecs = []
        for idx, e_vec in enumerate(m['eig_vecs'][1:]):
            for val in e_vec:
                model_eig_vecs.append(val)

        eigen_vecs.append(model_eig_vecs)
    X = [m['eig_vals'][0:4] for m in models_list]


    kmeans = KMeans(n_clusters=3, n_init=1000, max_iter=500, tol=1e-25).fit(X)

    # clust = OPTICS(min_samples=10, xi=.09, min_cluster_size=0.2).fit(X)

    data_labels = kmeans.labels_
    cluster_0_sores = []
    cluster_1_scores = []
    cluster_2_scores = []

    clust_0 = []
    clust_1 = []
    clust_2 = []

    for idx, model in enumerate(models_list):
        eig_vals = np.sort(model['eig_vals'])

        if data_labels[idx] == 0:
            cluster_0_scores.append(model['posterior_prob'])
            model['colour'] = 'blue'
            clust_0.append(model)
            colour = 'pink'

        elif data_labels[idx] == 1:
            cluster_1_scores.append(model['posterior_prob'])
            model['colour'] = 'green'

            clust_1.append(model)

            colour = 'green'

        elif data_labels[idx] == 2:
            cluster_2_scores.append(model['posterior_prob'])
            model['colour'] = 'red'

            clust_2.append(model)

            colour = 'red'



        # if eig_vals[5] > 0.1:
        #     colour = 'pink'

        # if eig_vals[4] > 0.1:
        #     colour = 'green'

        # if eig_vals[3] > 0.1:
        #     colour = 'yellow'

        colours.append(colour)


    models_list = sorted(models_list, key=lambda m: colours[m['model_idx']], reverse=True)

    print(np.shape(clust_0))
    print(np.shape(clust_1))
    print(np.shape(clust_2))


    out_path = output_dir + "clustered_posterior_prob_split.pdf"

    count = 0
    with PdfPages(out_path) as pdf:
        for idx, model in enumerate(clust_0):
            plt.bar(count, model['posterior_prob'], color=model['colour'])
            count += 1

        for idx, model in enumerate(clust_1):
            plt.bar(count, model['posterior_prob'], color=model['colour'])
            count += 1

        for idx, model in enumerate(clust_2):
            plt.bar(count, model['posterior_prob'], color=model['colour'])
            count += 1

        plt.ylabel('Model likelihood')
        plt.xlabel('Model')
        plt.ylim(-0.001)
        plt.tight_layout()

        pdf.savefig()
    plt.close()

    models_list = sorted(models_list, key=lambda m: m['posterior_prob'], reverse=True)
    out_path = output_dir + "clustered_posterior_prob_sorted.pdf"
    count = 0
    with PdfPages(out_path) as pdf:
        for idx, model in enumerate(models_list):
            plt.bar(count, model['posterior_prob'], color=model['colour'])
            count +=1
        plt.ylabel('Model likelihood')
        plt.xlabel('Model')
        plt.tight_layout()
        pdf.savefig()


    plt.close()



def hierarchical_cluster(inputs_dir, data_dir, output_dir):
    model_space_report_path = data_dir + "model_space_report.csv"
    adj_mat_name_template = "model_#REF#_adj_mat.csv"
    adj_matrix_path_template = inputs_dir + "adj_matricies/" + adj_mat_name_template


    model_space_report_df = pd.read_csv(model_space_report_path)
    total_accepted = sum(model_space_report_df['accepted_count'].values)

    # model_space_report_df.drop(model_space_report_df[model_space_report_df['accepted_count'] == 0].index, inplace=True)

    # Calculate acceptance ratio for each model
    model_space_report_df['acceptance_ratio'] = model_space_report_df.apply(lambda row: row['accepted_count'] / total_accepted, axis=1)

    species_names = [ 'N_1', 'N_2', 'S_glu', 'B_1', 'B_2', 'A_1', 'A_2']
    from_too_features_list = ['model_idx', 'posterior_probability']
    
    for s_1 in species_names:
        for s_2 in species_names:
            from_too_features_list.append(s_2 + '_' + s_1)
    
    # Generate features data frame
    flat_features_df = pd.DataFrame(columns=from_too_features_list)
    for model_idx in model_space_report_df['model_idx'].values:
        adj_mat_path = adj_matrix_path_template.replace("#REF#", str(model_idx))
        adj_mat_df = pd.read_csv(adj_mat_path)
        adj_mat_df.drop([adj_mat_df.columns[0]], axis=1, inplace=True)

        model_posterior_prob = model_space_report_df.loc[model_space_report_df['model_idx'] == model_idx]['acceptance_ratio'].values[0]

        model_features = [model_idx, model_posterior_prob]
        model_features = model_features + list(adj_mat_df.values.ravel())
        flat_features_df.loc[model_idx] = model_features

    X = flat_features_df.loc[:, ~flat_features_df.columns.isin(['model_idx', 'posterior_probability'])].values
    y = flat_features_df[['model_idx', 'posterior_probability']]

    linked = linkage(X, 'single', optimal_ordering=True)
    labels = y['model_idx'].values

    print(np.shape(linkage))
    print(np.shape(labels))

    plt.figure(figsize=(10, 7))
    d = dendrogram(linked,
                orientation='top',
                labels=labels,
                distance_sort='descending',
                show_leaf_counts=True)
    plt.show()
    print(d)
    print(np.shape(linked))
    exit()
    y['cluster'] = linked[:, 3]
    print(np.shape(linked))
    print(y)

def rdn_forest_test(inputs_dir, data_dir, output_dir):
    model_space_report_path = data_dir + "model_space_report.csv"
    adj_mat_name_template = "model_#REF#_adj_mat.csv"
    adj_matrix_path_template = inputs_dir + "adj_matricies/" + adj_mat_name_template


    model_space_report_df = pd.read_csv(model_space_report_path)
    total_accepted = sum(model_space_report_df['accepted_count'].values)

    model_space_report_df.drop(model_space_report_df[model_space_report_df['accepted_count'] == 0].index, inplace=True)

    # Calculate acceptance ratio for each model
    model_space_report_df['acceptance_ratio'] = model_space_report_df.apply(lambda row: row['accepted_count'] / total_accepted, axis=1)

    species_names = [ 'N_1', 'N_2', 'S_glu', 'B_1', 'B_2', 'A_1', 'A_2']
    from_too_features_list = ['model_idx', 'posterior_probability']
    
    for s_1 in species_names:
        for s_2 in species_names:
            from_too_features_list.append(s_2 + '_' + s_1)
    
    # Generate features data frame
    flat_features_df = pd.DataFrame(columns=from_too_features_list)
    for model_idx in model_space_report_df['model_idx'].values:
        adj_mat_path = adj_matrix_path_template.replace("#REF#", str(model_idx))
        adj_mat_df = pd.read_csv(adj_mat_path)
        adj_mat_df.drop([adj_mat_df.columns[0]], axis=1, inplace=True)

        model_posterior_prob = model_space_report_df.loc[model_space_report_df['model_idx'] == model_idx]['acceptance_ratio'].values[0]

        model_features = [model_idx, model_posterior_prob]
        model_features = model_features + list(adj_mat_df.values.ravel())
        flat_features_df.loc[model_idx] = model_features

    # Split train and test data
    X = flat_features_df.loc[:, ~flat_features_df.columns.isin(['model_idx', 'posterior_probability'])].values
    y = flat_features_df[['model_idx', 'posterior_probability']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    clf = RandomForestRegressor(n_jobs=2, random_state=0, n_estimators=500, max_features=None)
    clf.fit(X_train, y_train['posterior_probability'].values)

    test_predict = clf.predict(X_test)

    flat_features_predict_df = flat_features_df.loc[flat_features_df['model_idx'].isin(y_test['model_idx'].values)]
    flat_features_predict_df['prediction'] = test_predict
    flat_features_predict_df = flat_features_predict_df[['model_idx', 'posterior_probability', 'prediction']]
    
    flat_features_predict_df.sort_values(by='prediction', ascending=False, inplace=True)
    print(flat_features_predict_df)



def sklearn_nn_train():

    train_data_df= pd.read_csv('/media/behzad/DATA/experiments_data/BK_manu_data/output/two_species_3_rej_0/experiment_analysis/combined_model_space_report_with_motifs.csv')

    test_data_df = pd.read_csv('/media/behzad/DATA/experiments_data/BK_manu_data/output/three_species_stable_6_rej_0/experiment_analysis/combined_model_space_report_with_motifs.csv')
    

    train_data_df.drop('direct_self_limiting_counts', inplace=True, axis=1)
    train_data_df.drop('direct_competitive_counts', inplace=True, axis=1)
    test_data_df.drop('direct_self_limiting_counts', inplace=True, axis=1)
    test_data_df.drop('direct_competitive_counts', inplace=True, axis=1)

    # train_data_df.drop('dependent_counts', inplace=True, axis=1)
    # test_data_df.drop('dependent_counts', inplace=True, axis=1)

    X_data_columns = [x for x in train_data_df.columns if 'counts' in x]
    print(X_data_columns)

    # test_data_df = train_data_df


    Y_data = train_data_df['norm_marginal_means']


    X = train_data_df[X_data_columns].values
    y = Y_data.values
    X_test = test_data_df[X_data_columns].values

    y_test = test_data_df['norm_marginal_means'].values


    n=3 # how many times to shuffle the training data
    nhn_range=[8] # number of hidden neurons

    new_array = []
    X_noisy = np.copy(X)
    y_noisy = np.copy(y)

    for repeat_data in range(25):
        X_new = np.copy(X)
        X_new = X_new + np.random.normal(0, 0.8, 1)
        X_noisy = np.concatenate( (X_noisy, X_new), axis=0 )

        y_new = np.copy(y)
        y_noisy = np.concatenate((y_noisy, y_new), axis=0)


    print(np.shape(X))
    print(np.shape(y))
    print(np.shape(y_noisy))
    print(np.shape(X_noisy))

    score_dict = {}
    for nhn in nhn_range:
        print(nhn)
        # mlp = MLPRegressor(hidden_layer_sizes=(10, 5, 2), activation='tanh', learning_rate_init=0.001,
        #                    solver='sgd', shuffle=True, verbose=True, tol=1e-10, batch_size=500,
        #                    max_iter=200000, momentum=0.5, early_stopping=False, n_iter_no_change = 100,
        #                    validation_fraction=0.1)

        mlp = MLPRegressor(hidden_layer_sizes=(10), activation='relu', learning_rate_init=0.01,
                           solver='adam', shuffle=True, verbose=True, tol=1e-8, 
                           max_iter=200000, momentum=0.9, early_stopping=False, n_iter_no_change = 1000,
                           validation_fraction=0.1)


        mlp.fit(X_noisy, y_noisy)
        nhn_scores = []


        pred_y = mlp.predict(X)
        # pred_y = motif_counting.normalise_list(pred_y)

        pred_y = pred_y
        output = {
        'model_idx': test_data_df['model_idx'].values,
        'predictions': pred_y,
        'norm_marginal_mean': y_test
        }

        plt.scatter(x=range(len(pred_y)), y=pred_y)
        plt.scatter(x=range(len(pred_y)), y=y)

        plt.show()

        score_dict[nhn] = nhn_scores


        pred_y = mlp.predict(X_test)
        # pred_y = motif_counting.normalise_list(pred_y)

        pred_y = motif_counting.normalise_list(pred_y)
        output = {
        'model_idx': test_data_df['model_idx'].values,
        'predictions': pred_y,
        'norm_marginal_mean': y_test
        }

        print(len(pred_y))
        print(len(y_test))

        plt.scatter(x=range(len(pred_y)), y=y_test)
        plt.scatter(x=range(len(pred_y)), y=pred_y)

        plt.show()

        score_dict[nhn] = nhn_scores


    scores_df = pd.DataFrame(score_dict)
    scores_df.to_csv('scores_df_1.csv')


    pred_y = mlp.predict(X_test)
    pred_y = motif_counting.normalise_list(pred_y)

    output = {
    'model_idx': test_data_df['model_idx'].values,
    'predictions': pred_y,
    'norm_marginal_mean': y_test
    }


    df = pd.DataFrame(output)
    df.to_csv('output.csv')
    print(df)
if __name__ == "__main__":
    sklearn_nn_train()
    # k_means_test()
    print("hello world")