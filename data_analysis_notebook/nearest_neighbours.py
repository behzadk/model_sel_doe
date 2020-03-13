import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import matplotlib as mpl
from matplotlib import rcParams


font = {'size'   : 15, }
axes = {'labelsize': 'medium', 'titlesize': 'medium'}

sns.set_context("talk")
sns.set_style("white")

mpl.rc('font', **font)
mpl.rc('axes', **axes)
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
# plt.rcParams['text.usetex'] = True
plt.rcParams['axes.unicode_minus'] = False


def find_nearest_neighbours(current_combo, all_combos):
    neighbours = []

    min_nighbour_distance = 1e10

    for idx, potential_neighbour in enumerate(all_combos):
        if np.array_equal(potential_neighbour, current_combo):
            continue

        diff_combos = np.sum(np.absolute(np.subtract(potential_neighbour, current_combo)))


        if abs(diff_combos) == min_nighbour_distance:
            neighbours.append(idx)


        elif abs(diff_combos) < min_nighbour_distance:
            min_nighbour_distance = abs(diff_combos)
            print(min_nighbour_distance)
            print(current_combo)
            print(potential_neighbour)
            print("")

            neighbours = [idx]

        else:
            continue


    return neighbours


def find_additive_nearest_neighbours(current_combo, all_combos, feature_idx, feature_name):
    neighbour_idxs = []
    min_nighbour_distance = 1e10

    current_feature_value = current_combo[feature_idx]

    # Subset for only models that have feature value + 1
    subset_all_combos = all_combos.loc[all_combos[feature_name] == current_feature_value + 1]

    for idx, row in subset_all_combos.iterrows():

        candidate = row.values
        if np.array_equal(candidate, current_combo):
            continue

        diff_combos = np.sum(np.absolute(np.subtract(candidate, current_combo)))


        if abs(diff_combos) == min_nighbour_distance:
            neighbour_idxs.append(idx)


        elif abs(diff_combos) < min_nighbour_distance:
            min_nighbour_distance = abs(diff_combos)

            neighbour_idxs = [idx]

        else:
            continue

    return neighbour_idxs

def get_motif_neighbours(output_dir):
    model_space_report_df = pd.read_csv(output_dir + "combined_model_space_report_with_motifs.csv")
    output_path = output_dir + "motif_comparison.pdf"

    motif_columns = ['permissive_counts', 'dependent_counts',
       'submissive_counts', 'hedonistic_counts',
       'defensive_counts', 'logistic_counts', 'opportunistic_counts',
       'exponential_counts']

    motif_columns = ['permissive_counts', 'dependent_counts',
   'submissive_counts', 'hedonistic_counts',
   'defensive_counts', 'logistic_counts', 'opportunistic_counts',
   'exponential_counts']

    feature_effects_dict = {}

    for feature_idx, feature_name in enumerate(motif_columns):
        feature_effects = []
        n_additive_neighbours = 0

        # Iterate all systems
        for idx, row in model_space_report_df.iterrows():

            # Current model marginal
            current_marginal = row['norm_marginal_means']

            # Current encoded 
            enc_system = row[motif_columns].values

            neighbour_idxs = find_additive_nearest_neighbours(enc_system, model_space_report_df[motif_columns], feature_idx, feature_name)

            n_additive_neighbours += len(neighbour_idxs) 
            neighbours_df = model_space_report_df[model_space_report_df.index.isin(neighbour_idxs)]
            print("mdel_ref: ", row['model_idx'])
            print("marginal mean: ", row['norm_marginal_means'])
            print(neighbours_df)
            print("")

            feature_effects += [x - current_marginal for x in neighbours_df['norm_marginal_means'].values] 
        exit()
        feature_effects_dict[motif_columns[feature_idx]] = feature_effects

    motif_stdev = []
    motif_median = []
    motif_data = []

    all_data_names = []
    all_data_points = []
    all_data_medians = []

    for f in motif_columns:
        median_effect = np.median(feature_effects_dict[f])
        effect_std = np.std(feature_effects_dict[f])
        motif_stdev.append(effect_std)
        motif_median.append(median_effect)
        motif_data.append(feature_effects_dict[f])

        for x in feature_effects_dict[f]:
            all_data_names.append(f)
            all_data_points.append(x)
            all_data_medians.append(median_effect)


    df_data = {'name': motif_columns, 'all_data': motif_data, 'median': motif_median, 'stdev': motif_stdev}
    all_data_df = {'name': all_data_names, 'all_data_points': all_data_points, 'all_data_median': all_data_medians}
    
    all_data_df = pd.DataFrame(all_data_df)
    # all_data_df.sort_values('all_data_median', inplace=True, ascending=False)
    sorter = motif_columns
    sorter_index = dict(zip(sorter,range(len(sorter))))
    all_data_df['name_order']  = all_data_df['name'].map(sorter_index)
    all_data_df.sort_values('name_order', inplace=True, ascending=True)


    all_data_df.to_csv(output_dir + 'motif_datapoints.csv')
    print(all_data_df.columns)

    print(list(set(all_data_df['name'].values)))
    diverging_colours = sns.color_palette("RdBu_r", len(motif_columns) + 2)
    diverging_colours.pop(5)
    diverging_colours.pop(4)

    output_path = output_dir + "motif_comparison_boxplot_vert.pdf"

    width_inches = 174 / 25.4
    height_inches = 100 / 25.4
    fig, ax = plt.subplots(figsize=(width_inches, height_inches))

    # sns.barplot(x='name', y='mean', 
    #                  data=analysis_df, alpha=0.9, ax=ax, palette=diverging_colours, vert=False)

    # analysis_df.plot("name", "mean", kind="barh", color=diverging_colours, ax=ax, title='', width=1)
    
    sns.stripplot(y="all_data_points", x="name", data=all_data_df, ax=ax, size=2,
        orient="v", palette=diverging_colours, zorder=10)
    sns.boxplot(y="all_data_points", x="name", data=all_data_df, ax=ax, orient="v", palette=diverging_colours,
        boxprops={'facecolor':'None'}, showfliers=False, linewidth=2)
    
    # whiskerprops={'linewidth':0, "zorder":0}, showcaps=False,
    # plt.scatter()
    ax.set_ylabel('')
    ax.set_xticklabels('')
    ax.set_xlabel('')

    ax.set(xlim=(None, None))
    ax.set(ylim=(None, None))

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(True)

    ax.spines["bottom"].set_alpha(0.5)
    ax.spines["left"].set_alpha(0.5)
    ax.margins(x=0)
    ax.margins(y=0)

    ax.legend().remove()

    fig.tight_layout()
    plt.savefig(output_path, dpi=500, bbox_inches='tight')

    output_path = output_dir + "motif_comparison_violin.pdf"

    fig, ax = plt.subplots(figsize=(8.5, 5.11))

    # sns.barplot(x='name', y='mean', 
    #                  data=analysis_df, alpha=0.9, ax=ax, palette=diverging_colours, vert=False)

    # analysis_df.plot("name", "mean", kind="barh", color=diverging_colours, ax=ax, title='', width=1)
    # print(analysis_df)
    sns.stripplot(x="all_data_points", y="name", data=all_data_df, ax=ax, size=3,
    orient="h", palette=diverging_colours, zorder=10)
    sns.violinplot(x="all_data_points", y="name", data=all_data_df, ax=ax, orient="h", color="white", 
        showfliers=False, linewidth=2, width=0.9, scale_hue=False, saturation=1.0
        )
    ax.collections[0].set_edgecolor(diverging_colours)
    # for ax in grid.axes.flatten():
    #     ax.collections[1].set_edgecolor('red')

    # whiskerprops={'linewidth':0, "zorder":0}, showcaps=False,
    # plt.scatter()
    ax.set_xlabel('normalised marginal change')
    # ax.set_xlabel('mean marginal probability change')
    # ax.set_xticklabels('')
    ax.set_yticklabels([])
    ax.set_ylabel([])

    # ax.errorbar(motif_columns, 
    #             analysis_df['mean'], 
    #             yerr=analysis_df['stdev'], fmt=',', color='black', alpha=1,
    #             label=None, elinewidth=0.5)

    # ax.set(xlim=(-0,None))
    # ax.set(ylim=(-0))

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(True)

    ax.spines["bottom"].set_alpha(0.5)
    ax.spines["left"].set_alpha(0.5)

    ax.legend().remove()

    fig.tight_layout()
    plt.savefig(output_path, dpi=500)



def test():
    a = [0, 0, 0]
    b = [1, 0, 0]
    c = [2, 2, 0]
    d = [0, 0, 1]

    marginal_a = 0.05
    marginal_b = 0.5
    marginal_c = 0.3
    marginal_d = 0.15

    feature_idxs = list(range(3))


    models = [a, b, c, d]

    for x in feature_idxs:

        # Check if adding the feature to the model results in a change in model marginal probability
        for m in models:

            min_nighbour_distance = 100
            neighbours = []

            # Find nearest neighbours to m
            for neigh_idx, potential_neighbour in enumerate(models):

                if m == potential_neighbour:
                    continue

                if abs(sum(m) - sum(a)) == min_nighbour_distance:
                    neighbours.append(potential_neighbour)


                elif abs(sum(m) - sum(a)) < min_nighbour_distance:
                    min_nighbour_distance = abs(sum(m) - sum(potential_neighbour))
                    neighbours = [potential_neighbour]

                else:
                    continue

            # From neighbours, find those that have 1 extra of feature
            for n in neighbours:
                if n[x] == m[x] + 1:
                    print(x, n)

if __name__ == "__main__":
    test()
