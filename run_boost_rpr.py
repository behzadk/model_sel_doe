import algorithms
from model_space import Model
import xml.etree.ElementTree as ET
import csv
import os
import population_modules
from model_space import ModelSpace
import algorithm_utils as alg_utils
import numpy as np
import matplotlib as mpl

import matplotlib.pyplot as plt
from operator import mul
import classification
import seaborn as sns;

sns.set()
from scipy.optimize import fsolve
import classification

import sys
import glob

import pandas as pd
import pickle

# Set time points
t_0 = 0
# t_end = 1000
t_end = 500
dt = 0.5

restart = True

if int(sys.argv[2]) == 1:
    input_folder = './input_files/input_files_two_species_0/input_files/'
    output_folder = './output/'
    experiment_name = 'two_species_stable_NUM/'
    experiment_number = str(sys.argv[1])

    fit_species = [0, 1]


elif int(sys.argv[2]) == 2:
    input_folder = './input_files/input_files_three_species_0/input_files/'
    output_folder = './output/'
    experiment_name = 'three_species_stable_NUM/'
    experiment_number = str(sys.argv[1])

    fit_species = [0, 1, 2]

elif int(sys.argv[2]) == 3:
    input_folder = './input_files/input_files_two_species_spock_manu_1/input_files/'
    output_folder = './output/'
    experiment_name = 'spock_manu_stable_NUM/'
    experiment_number = str(sys.argv[1])

    fit_species = [0, 1]
    # fit_species = [0, 1, 5, 6, 7]

elif int(sys.argv[2]) == 4:
    input_folder = './input_files/input_files_one_species_0/input_files/'
    output_folder = './output/'
    experiment_name = 'one_species_stable_NUM/'
    experiment_number = str(sys.argv[1])

    fit_species = [0]

else:
    experiment_name = None
    experiment_number = None
    output_folder=None
    fit_species = None
    input_folder = None
    print("Please specify routine... exiting ")
    exit()


def extract_parameters_from_xml(input_file):
    tree = ET.parse(input_file)
    root = tree.getroot()

    params = root.iter('parameters')
    init_species = root.iter('initial')
    input_params_dict_list = []
    input_init_species_dict_list = []

    for idx, child in enumerate(params):
        new_dict = {}
        for c in child:
            text = c.text.split()
            if text[0] == "uniform":
                new_dict[c.tag] = (float(text[1]), float(text[2]))

            elif text[0] == "constant":
                new_dict[c.tag] = (float(text[1]), float(text[1]))

            else:
                print("unknown input type")

        input_params_dict_list.append(new_dict)

    for idx, child in enumerate(init_species):
        new_dict = {}
        for c in child:
            text = c.text.split()
            if text[0] == "uniform":
                new_dict[c.tag] = (float(text[1]), float(text[2]))

            elif text[0] == "constant":
                new_dict[c.tag] = (float(text[1]), float(text[1]))

            else:
                print("unknown input type")

        input_init_species_dict_list.append(new_dict)

    return input_params_dict_list, input_init_species_dict_list


def import_input_file(input_path):
    data_dict = {}
    with open(input_path) as fin:
        reader = csv.reader(fin, skipinitialspace=True)
        for row in reader:
            data_dict[row[0]] = [float(i) for i in row[1:]]

    return data_dict


def find_latest_population_pickle(exp_folder):
    sub_dirs = glob.glob(exp_folder + "*/")
    population_dirs = [f for f in sub_dirs if "Population_" in f]
    print(population_dirs)
    if len(population_dirs) == 0:
        return None

    # Get folder name
    population_dirs = [f.split('/')[-2] for f in population_dirs]
    population_dirs = sorted(population_dirs, key=lambda x: int(x[-1]))

    # Return top population dir
    for f in sub_dirs:
        for idx in range(len(population_dirs)):
            pickles = glob.glob(f + "checkpoint.pickle")
            print(pickles)

            if len(pickles) == 0:
                continue

            elif len(pickles) > 1:
                print("Only one pickle should exist, exiting... ", population_dirs[-idx])

            else:
                return pickles[0]

        idx += 1

    return None

def ABC_rejection():
    # Set time points
    t_0 = 0
    t_end = 1000
    dt = 0.5


    experiment_folder = experiment_name.replace('NUM', experiment_number)
    exp_output_folder = output_folder + experiment_folder
    try:
        os.mkdir(exp_output_folder)

    except FileExistsError:
        pass

    # Load models from input files
    model_list = []
    for i in range(int((len(os.listdir(input_folder)) / 2))):
        input_params = input_folder + "params_" + str(i) + ".csv"
        input_init_species = input_folder + "species_" + str(i) + ".csv"
        init_params = import_input_file(input_params)
        init_species = import_input_file(input_init_species)
        model_new = Model(i, init_params, init_species)
        model_list.append(model_new)


    # Run ABC_rejecction algorithm
    ABC_algs = algorithms.ABC(t_0, t_end, dt, model_list, population_size=1e10, n_sims_batch=150, fit_species=fit_species, distance_function_mode=0, n_distances=3, out_dir=exp_output_folder)
    ABC_algs.run_rejection()
    # rejection_alg.run_rejection()

    print("")
    print("")


def ABCSMC():
    suffixes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    suffixes = [str(x) for x in suffixes]
    restart_idx = 0

    experiment_folder = experiment_name.replace('NUM', experiment_number)
    exp_output_folder = output_folder + experiment_folder
    latest_pickle_path = find_latest_population_pickle(exp_output_folder)

    ABC_algs = None

    ## reload previous population
    if latest_pickle_path:
        print("loading latest checkpoint... ")
        with open(latest_pickle_path, 'rb') as handle:
            ABC_algs = pickle.load(handle)
        ABC_algs.run_model_selection_ABC_SMC(alpha=0.3)

    # Otherwise start new
    else:
        try:
            os.mkdir(exp_output_folder)

        except FileExistsError:
            pass

        # Load models from input files
        model_list = []
        for i in range(int((len(os.listdir(input_folder)) / 2))):
            input_params = input_folder + "params_" + str(i) + ".csv"
            input_init_species = input_folder + "species_" + str(i) + ".csv"
            init_params = import_input_file(input_params)
            init_species = import_input_file(input_init_species)

            model_new = Model(i, init_params, init_species)

            model_list.append(model_new)

        print(fit_species)
        # Run ABC_rejecction algorithm
        ABC_algs = algorithms.ABC(t_0, t_end, dt, model_list=model_list, population_size=20000, n_sims_batch=200,
            fit_species=fit_species, distance_function_mode=0, n_distances=3, out_dir=exp_output_folder)

    ABC_algs.run_model_selection_ABC_SMC(alpha=0.1)


def resample_and_plot_posterior():
    # Set time points
    t_0 = 0
    t_end = 1000
    dt = 1

    print(sys.argv[2])

    if int(sys.argv[2]) == 1:
        input_folder = './input_files_two_species_0/input_files/'
        output_folder = './output/resample_posterior/'

        experiment_number = str(sys.argv[1])
        experiment_name = 'two_species_stable_NUM/'.replace('NUM', experiment_number)
        experiment_folder = './output/' + experiment_name

        posterior_params_folder = experiment_folder + 'Population_0/model_sim_params/'
        fit_species = [0, 1]


    elif int(sys.argv[2]) == 2:
        experiment_number = str(sys.argv[1])

        input_folder = './input_files_three_species_0/input_files/'

        experiment_name = 'three_species_stable_NUM/'.replace('NUM', experiment_number)
        experiment_folder = './output/' + experiment_name

        fit_species = [0, 1, 2]


    else:
        experiment_name = None
        experiment_number = None
        output_folder = None
        input_folder = None
        print("Please specify routine... exiting ")
        exit()

    posterior_params_folder = experiment_folder + 'Population_0/model_sim_params/'
    output_folder = experiment_folder + 'resample_posterior_plots/'

    try:
        os.mkdir(output_folder)

    except FileExistsError:
        pass

    model_list = []

    # Load models
    posterior_files_list = glob.glob(posterior_params_folder +'/*')
    for f in posterior_files_list:
        model_idx = int(os.path.basename(f).split("_")[1])
        posterior_df = pd.read_csv(f)

        posterior_df = posterior_df.loc[posterior_df['Accepted'] == True]

        if len(posterior_df) <= 1:
            continue

        input_params = input_folder + "params_" + str(model_idx) + ".csv"
        input_init_species = input_folder + "species_" + str(model_idx) + ".csv"

        init_params = import_input_file(input_params)
        init_species = import_input_file(input_init_species)

        model_new = Model(model_idx, init_params, init_species)
        alg_utils.make_posterior_kdes(model_new, posterior_df, init_params, init_species)

        # model_new.alt_generate_params_kde()
        model_list.append(model_new)

        # simple_sim = algorithms.SimpleSimulation(t_0, t_end, dt,
        #                                          model_list, batch_size=10, num_batches=1, fit_species=fit_species,
        #                                          out_dir=output_folder + 'model_' + str(model_idx) + '/')

        # simple_sim.simulate_and_plot()

        rejection_alg = algorithms.Rejection(t_0, t_end, dt, model_list, 1e6, 288, 2, 1, 3, output_folder)
        rejection_alg.run_rejection()
        print("")

        # Run ABC_rejecction algorithm


def simulate_and_plot():
    # Set time points
    t_0 = 0
    t_end = 1000
    dt = 0.5

    experiment_folder = experiment_name.replace('NUM', experiment_number)
    output_folder = output_folder + experiment_folder

    try:
        os.mkdir(output_folder)

    except FileExistsError:
        pass


    # Load models from input files
    model_list = []

    K_omega_T_space = np.linspace(np.log(1.0e21), np.log(1.0e22), num=10)
    idx = 0
    for x in range(len(K_omega_T_space)):
        for i in range(0, int((len(os.listdir(input_folder)) / 2))):
            input_init_species = input_folder + "species_" + str(i) + ".csv"
            input_params = input_folder + "params_" + str(i) + ".csv"

            init_params = import_input_file(input_params)
            init_species = import_input_file(input_init_species)
        

            if i != 0:
                continue

            for p in sorted(init_params, key=str.lower):
                print(p, init_params[p])

            model_new = Model(i, init_params, init_species)
            model_list.append(model_new)
            # init_params['D'] = [0.0, 0.0, 'uniform']

            # rejection_alg = algorithms.Rejection(t_0, t_end, dt, model_list, 1e6, 12, fit_species, 0, 3, output_folder)
            # rejection_alg.run_rejection()
            print("")

            simple_sim = algorithms.SimpleSimulation(t_0, t_end, dt,
                                                     model_list, batch_size=15, num_batches=1, fit_species=fit_species,
                                                     distance_function_mode=0, out_dir=output_folder + str(x) + "_")
            simple_sim.simulate_and_plot()
            # exit()

    print("")
    print("")




if __name__ == "__main__":
    # for i in range(50):
    #     steady_state_test(i)
    ABCSMC()
    # ABC_rejection()
    # simulate_and_plot()
    # resample_and_plot_posterior()
    # ABC_rejection()
    # eig_classification_test()
    # repressilator_test()
    # exit()
    # ABC_rejection()
