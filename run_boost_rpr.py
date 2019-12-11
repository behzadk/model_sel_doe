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

import tarfile
import yaml
import argparse


# Set time points

with open("experiment_config_spock_survival.yaml", 'r') as yaml_file:
    experiment_config = yaml.load(yaml_file, Loader=yaml.FullLoader)
    experiment_config['final_epsilon'] = [float(x) for x in experiment_config['final_epsilon']]

# Unpack config file
input_folder = experiment_config['inputs_folder']
output_folder = experiment_config['output_folder']
experiment_name = experiment_config['experiment_name']

t_0 = experiment_config['t_0']
t_end = experiment_config['t_end']
dt = experiment_config['dt']

distance_function_mode = experiment_config['distance_function_mode']
run_rejection = experiment_config['run_rejection']
run_SMC = experiment_config['run_SMC']


def import_input_file(input_path):
    data_dict = {}
    with open(input_path) as fin:
        reader = csv.reader(fin, skipinitialspace=True)
        for row in reader:
            data_dict[row[0]] = [float(i) for i in row[1:]]

    return data_dict


def ABCSMC_run_tests():
    experiment_folder = experiment_name.replace('NUM', str(experiment_number))
    exp_output_folder = output_folder + experiment_folder

    latest_pickle_path = alg_utils.find_latest_population_pickle(exp_output_folder)
    print(latest_pickle_path)
    ABC_algs = None
    
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
    fit_species = [0]
    # Run ABC_rejection algorithm
    ABC_algs = algorithms.ABC(t_0, t_end, dt, model_list, population_size=1000, n_sims_batch=150, fit_species=fit_species, 
        final_epsilon=final_epsilon, distance_function_mode=0, n_distances=1, out_dir=exp_output_folder)

    ABC_algs.run_model_selection_ABC_SMC(alpha=0.5, run_test=1)
    alg_utils.make_tarfile(exp_output_folder[0:-1] + "_pop_" + str(ABC_algs.population_number) + ".tar.gz", exp_output_folder)

def ABCSMC():
    experiment_folder = experiment_name.replace('NUM', str(experiment_number))
    exp_output_folder = output_folder + experiment_folder

    latest_pickle_path = alg_utils.find_latest_population_pickle(exp_output_folder)
    print(latest_pickle_path)
    ABC_algs = None
    
    try:
        os.mkdir(exp_output_folder)

    except FileExistsError:
        pass


    if 0:
        master_pickle_path = output_folder + "master_pickle_pop_0checkpoint.pickle"
        with open(master_pickle_path, 'rb') as handle:
            ABC_algs = pickle.load(handle)
            ABC_algs.out_dir = exp_output_folder
            ABC_algs.population_size = 250


    # ## reload previous population
    # elif latest_pickle_path:
    #     print("loading latest checkpoint... ")
    #     with open(latest_pickle_path, 'rb') as handle:
    #         ABC_algs = pickle.load(handle)
    #     # ABC_algs.run_model_selection_ABC_SMC(alpha=0.3)

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

            # if i!= 3307 and i != 521 and i != 677 and i != 511:
            #     continue

            model_list.append(model_new)

        # Run ABC_rejection algorithm
        ABC_algs = algorithms.ABC(t_0, t_end, dt, model_list=model_list, population_size=500, n_sims_batch=100,
            fit_species=fit_species, final_epsilon=final_epsilon, 
            distance_function_mode=distance_function_mode, n_distances=len(final_epsilon), out_dir=exp_output_folder)

    if run_rejection == "Y":
        ABC_algs.current_epsilon = final_epsilon

    ABC_algs.run_model_selection_ABC_SMC(alpha=0.3)
    alg_utils.make_tarfile(exp_output_folder[0:-1] + "_pop_" + str(ABC_algs.population_number) + ".tar.gz", exp_output_folder)


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
    t_end = 500
    dt = 0.5

    experiment_folder = experiment_name.replace('NUM', experiment_number)
    exp_output_folder = output_folder + experiment_folder

    try:
        os.mkdir(output_folder)

    except FileExistsError:
        pass


    # Load models from input files
    model_list = []

    idx = 0
    for i in range(int((len(os.listdir(input_folder)) / 2))):
        input_init_species = input_folder + "species_" + str(i) + ".csv"
        input_params = input_folder + "params_" + str(i) + ".csv"

        init_params = import_input_file(input_params)
        init_species = import_input_file(input_init_species)
    
        if i != 0:
            continue

        model_new = Model(i, init_params, init_species)
        model_list.append(model_new)
        # init_params['D'] = [0.0, 0.0, 'uniform']

        # rejection_alg = algorithms.Rejection(t_0, t_end, dt, model_list, 1e6, 12, fit_species, 0, 3, output_folder)
        # rejection_alg.run_rejection()
        print("")
        fit_species = [0, 1, 2, 3, 4]

        simple_sim = algorithms.SimpleSimulation(t_0, t_end, dt,
                                                 model_list, batch_size=15, num_batches=1, fit_species=fit_species,
                                                 distance_function_mode=0, out_dir=exp_output_folder + str(i) + "_")
        simple_sim.simulate_and_plot()

    print("")
    print("")

if __name__ == "__main__":
    # for i in range(50):
    #     steady_state_test(i)
    # ABCSMC_run_tests()
    # exit()
    # ABCSMC_run_gerlaud_test()
    ABCSMC()
    # ABC_rejection()
    # simulate_and_plot()
    # resample_and_plot_posterior()
    # ABC_rejection()
    # eig_classification_test()
    # repressilator_test()
    # exit()
    # ABC_rejection()
