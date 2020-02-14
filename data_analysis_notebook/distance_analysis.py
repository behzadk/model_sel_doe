import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib as mpl
from matplotlib import rcParams
import data_utils

import ternary
from scipy import stats
import math
from sklearn.neighbors import KernelDensity
from sklearn import mixture
from itertools import combinations

from sklearn import svm

def shannon_entropy(p):
    """Computes the Shannon Entropy at a distribution in the simplex."""
    s = 0.
    for i in range(len(p)):
        try:
            s += p[i] * math.log(p[i])
        except ValueError:
            continue
    return -1. * s

# def my_heatmapf(func, step=0.001, scale=10, boundary=True, cmap=None, ax=None,
#              style='triangular', permutation=None, vmin=None, vmax=None):
    

#     for i in np.arange(start=0, stop=scale, step=step)


# def simplex_iterator(scale=1.0, boundary=True, num_steps=50):
#     start = 0

#     for i in np.arange(start=start, stop=scale, step=step)
#         for j in np.arange(start=start, stop=scale + (1 - start) - i, step=step):
#             k = scale - i - j
#             yield (i, j, k)

class kde_func:
    def __init__(self, data, scale, step):
        # self.kde = kde = stats.gaussian_kde(data.T)
        
        self.sklearn_model = KernelDensity(bandwidth=0.1, kernel='gaussian', leaf_size=100)
        # self.sklearn_model = mixture.GaussianMixture(n_components=3, covariance_type='diag')
        self.sklearn_model.fit(data)

        # self.sklearn_model = svm.LinearSVR(tol=1e-5)
        # print(data)
        # exit()
        self.sklearn_model.fit(data)

        self.scale = scale
        self.step = step
        self.v_max = 0

    def run_kde(self, x):
        x = np.array(x)
        x = x.reshape(1, -1)
        # ret = self.kde(x) * self.scale

        ret = self.sklearn_model.score_samples(x)
        # ret = self.sklearn_model.predict(x)

        # print(np.exp(ret))
        if ret[0] > self.v_max:
            self.v_max = ret[0]
            print(x)
            print(self.v_max)
            print("")

        return ret[0]



def plot_steady_state_ratio(output_dir, distance_columns, figure_output_dir):
    model_space_report_df = pd.read_csv(output_dir + "combined_model_space_report.csv")
    data_utils.make_folder(figure_output_dir)

    model_idxs = model_space_report_df['model_idx'].values
    model_idxs = sorted(model_idxs)

    model_distances_path_template = output_dir +  "combined_model_distances/model_#IDX#_population_distances.csv"
    scatter_output_path_template = figure_output_dir + "distance_scatter_tern_#IDX#.pdf"
    heatmap_output_path_template = figure_output_dir + "distance_heatmap_tern_#IDX#.pdf"

    for model_idx in model_idxs:
        if model_idx not in [226, 872, 1294, 1416]:
            continue

        model_distances_path =  model_distances_path_template.replace("#IDX#", str(model_idx))
        model_dist_df = pd.read_csv(model_distances_path)
        
        # Convert distances back to OD values

        if len(distance_columns) == 3:
            # print(model_dist_df[distance_columns])
            t_data = model_dist_df[distance_columns]
            t_data = t_data
            if len(t_data) < 100:
                continue

            print(model_idx)
            t_data = t_data.applymap(lambda x: (1/x))
            # Convert to fractional populations
            t_data = t_data.div(t_data.sum(axis=1), axis=0)
            model_dist_df = t_data


            t_data = t_data.values
            t_data = t_data.astype(np.float64)
            # plot_data = np.multiply(t_data, 100)


            # kde = stats.gaussian_kde(t_data.T)


            # test = np.array([0.01, 0.01, 0.01])
            # print(np.shape(test))
            # kde(test.T)
            # exit()
            # v_max = np.max(t_data.values)
            # v_min = np.min(t_data.values)
            # min_max_scalar = lambda x: (x - v_min) / (v_max - v_min)

            # t_data = t_data.applymap(min_max_scalar)
            # t_data = t_data.values
            scale = int(100)
            func = kde_func(t_data, scale, step=0.1)

            # i = func.run_kde([0.1, 0.1, 0.8])
            # print(i)
            # i = func.run_kde([0.05, 0.05, 0.9])
            # print(i)

            # i = func.run_kde([0.99, 0.01, 0.0])
            # print(i)
            # i = func.run_kde([0.998, 0.001, 0.001])
            # print(i)

            # i = func.run_kde([1.0, 0.0, 0.0])
            # print(i)


            figure, t_ax = ternary.figure(scale=scale)
            t_ax.boundary(linewidth=2.0)
            t_ax.gridlines(color="black", multiple=scale/10)
            # t_ax.gridlines(color="red", multiple=scale/4, linewidth=0.5)
            t_ax.ticks(axis='lbr', linewidth=1, multiple=scale/10, tick_formats="%.1f")
            t_ax.clear_matplotlib_ticks()
            t_ax.get_axes().axis('off')
            # Set Axis labels and Title
            fontsize = 20
            t_ax.set_title("Model: " + str(model_idx), fontsize=fontsize)

            mpl_args = {'size': 2}

            t_ax.heatmapf(func.run_kde, scale=scale, boundary=True, style='hexagonal')

            figure_output_path = heatmap_output_path_template.replace("#IDX#", str(model_idx))
            # plt.show()
            plt.savefig(figure_output_path, dpi=500)
            plt.close()

            ### Do scatter ###
            scale = int(100)
            t_data = model_dist_df.div(model_dist_df.sum(axis=1), axis=0)
            plot_data = np.multiply(t_data.values, 100)
            figure, t_ax = ternary.figure(scale=scale)
            t_ax.boundary(linewidth=2.0)
            t_ax.gridlines(color="black", multiple=scale/10)
            t_ax.gridlines(color="red", multiple=scale/4, linewidth=0.5)
            t_ax.ticks(axis='lbr', linewidth=1, multiple=scale/10, tick_formats="%.1f")
            t_ax.clear_matplotlib_ticks()
            t_ax.get_axes().axis('off')
            # Set Axis labels and Title
            fontsize = 20
            t_ax.set_title("Model: " + str(model_idx), fontsize=fontsize)

            mpl_args = {'size': 2}

            t_ax.scatter(points=plot_data, s=2)

            figure_output_path = scatter_output_path_template.replace("#IDX#", str(model_idx))
            # plt.show()
            plt.savefig(figure_output_path, dpi=500)
            plt.close()


        else:
            fig, ax = plt.subplots()

            pop_distances_df = model_dist_df[distance_columns]

            pop_distances_df = pop_distances_df.applymap(lambda x: 1/x)
            pop_distances_df = pop_distances_df.assign(pop_ratio=lambda x: np.log(x[distance_columns[0]] /  x[distance_columns[1]] ))
            print(pop_distances_df)
            # sns.scatterplot(x=distance_columns[0], y=distance_columns[1], ax=ax, data=pop_distances_df)

            sns.distplot(pop_distances_df['pop_ratio'], hist=False, rug=False)

            plt.show()
        # exit()



def get_abs_diff_populations(mat_row):
    row_combos = combinations(mat_row, r=2)
    
    diff = []
    for i in row_combos:
        diff.append((i[0] - i[1])**2)


    return sum(diff)


def ratio_dimensionality_reduction(output_dir, distance_columns, figure_output_dir):
    model_space_report_df = pd.read_csv(output_dir + "combined_model_space_report.csv")
    data_utils.make_folder(figure_output_dir)

    model_idxs = model_space_report_df['model_idx'].values
    model_idxs = sorted(model_idxs)

    model_distances_path_template = output_dir +  "combined_model_distances/model_#IDX#_population_distances.csv"
    scatter_output_path_template = figure_output_dir + "distance_scatter_tern_#IDX#.pdf"
    heatmap_output_path_template = figure_output_dir + "distance_heatmap_tern_#IDX#.pdf"

    df_list = []
    
    min_pop_bal = 2.0

    for model_idx in model_idxs:
        if model_idx not in [226, 872, 1294, 1416]:
            continue

        model_distances_path =  model_distances_path_template.replace("#IDX#", str(model_idx))
        model_dist_df = pd.read_csv(model_distances_path)
        
        if len(distance_columns) == 3:
            # print(model_dist_df[distance_columns])
            t_data = model_dist_df[distance_columns]
            if len(t_data) < 100:
                continue

            t_data = t_data.applymap(lambda x: (1/x))
            # Convert to fractional populations
            t_data = t_data.div(t_data.sum(axis=1), axis=0)
            t_data['pop_balance'] = np.apply_along_axis(get_abs_diff_populations, axis=1, arr=t_data.values)
            # print(t_data.iloc[0])
            # exit()
            t_data['model_idx'] = [model_idx for i in range(len(t_data))]

            q75, q25 = np.percentile(t_data['pop_balance'].values, [75 ,25])

            # print(q75, q25)

            if q25 < min_pop_bal:
                min_pop_bal = q25
                print(model_idx)
                print(q25)
                print("")

            df_list.append(t_data)


    master_df = pd.concat(df_list)
    output_path = output_dir + "motif_comparison_boxplot.pdf"
    fig, ax = plt.subplots()

    # sns.stripplot(x="pop_balance", y="model_idx", data=master_df, ax=ax, size=4,
    #     orient="h", zorder=10)
    sns.boxplot(x="pop_balance",  y="model_idx", data=master_df, ax=ax, orient="h",
        boxprops={'facecolor':'None'}, showfliers=False, linewidth=2, width=0.9)
    ax.set_xlabel('normalised marginal change')
    # ax.set_xscale('log')
    # ax.set_yticklabels('')
    ax.set_ylabel('')

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(True)

    ax.spines["bottom"].set_alpha(0.5)
    ax.spines["left"].set_alpha(0.5)

    ax.legend().remove()

    fig.tight_layout()
    plt.show()
            # plt.savefig(output_path, dpi=500)






def test():
    mu=np.array([1,10,20])
    # Let's change this so that the points won't all lie in a plane...
    sigma=np.matrix([[20,10,10],
                     [10,25,1],
                     [10,1,50]])

    data=np.random.multivariate_normal(mu,sigma,1000)
    values = data.T

    kde = stats.gaussian_kde(values)


if __name__ == "__main__":
    test_dict = {"A": [0, 5, 7],
                "B": [9, 9, 9], 
                "C": [1000, 1000, 1000]}

    df = pd.DataFrame(test_dict)
    print(df)
    column_names = ["A", "B", "C"]
    x = df.div(df.sum(axis=1), axis=0)
    # x = 

    print(x)
