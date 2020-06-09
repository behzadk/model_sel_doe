import numpy as np
import pandas as pd
import glob
import pickle
import algorithms
import seaborn as sns
import matplotlib.pyplot as plt

def find_latest_population_pickle(exp_folder):
    sub_dirs = glob.glob(exp_folder + "**/")
    population_dirs = [f for f in sub_dirs if "Population" in f]
    
    if len(population_dirs) == 0:
        return None

    # Get folder name
    population_names = [f.split('/')[-2] for f in population_dirs]

    population_dirs = [x for y, x in sorted(zip(population_names, population_dirs), key=lambda y: int(y[0][-1]), reverse=True)]

    # Return top population dir
    for f in population_dirs:
        for idx in range(len(population_names)):
            pickles = glob.glob(f + "checkpoint.pickle")

            if len(pickles) == 0:
                continue

            elif len(pickles) > 1:
                print("Only one pickle should exist, exiting... ", population_names[-idx])

            else:
                return pickles[0]

        idx += 1

    return None


def main():
    data_dir = '/Volumes/Samsung_T5/BK_manu_data_backup/raw_output/three_species_7_SMC_3/'

    exp_dirs = glob.glob(data_dir + "**/")
    population_pickles_list = []

    pickle_path_list = []

    experiment_name = "chunk_"

    model_2665_marginals = []

    idx = 0
    for d in exp_dirs:
        pickle_path = find_latest_population_pickle(d)

        try:
            with open(pickle_path, 'rb') as p_handle:
                p_dir = "/".join(pickle_path.split('/')[:-1]) + "/"
                p = pickle.load(p_handle)
        except TypeError:
            continue

        model_2665_marginals.append(p.model_space._model_list[2265].curr_margin)
        print(idx)
        np.savetxt('./test.csv', model_2665_marginals, delimiter=',')
        idx += 1



    print("")
    print(np.mean(model_2665_marginals))
    print(np.median(model_2665_marginals))
    print(np.std(model_2665_marginals))

def split_data():
    x = np.loadtxt('./test.csv', delimiter=',')
    idx_max = np.argmax(x)
    x = np.delete(x, idx_max)

    print(x)
    print(np.mean(x))
    print(np.std(x))

    sns.stripplot(y=x)
    plt.show()

if __name__ == "__main__":
    split_data()