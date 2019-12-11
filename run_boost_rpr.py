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

# Set time points
t_0 = 0
# t_end = 1000
t_end = 1500
dt = 0.5

restart = True

if int(sys.argv[2]) == 1:
    input_folder = './input_files/input_files_two_species_0/input_files/'
    output_folder = './output/'
    experiment_name = 'two_species_stable_NUM/'
    experiment_number = str(sys.argv[1])
    distance_function_mode = 2

    C = 1e12
    final_epsilon = [1e3 / C, 0.001, 1 / 0.001]

    fit_species = [0, 1]

elif int(sys.argv[2]) == 2:
    input_folder = './input_files/input_files_three_species_0/input_files/'
    output_folder = './output/'
    experiment_name = 'three_species_stable_NUM/'
    experiment_number = str(sys.argv[1])
    C = 1e12

    final_epsilon = [1e3 / C, 0.001, 1 / 0.001]

    fit_species = [0, 1, 2]


elif int(sys.argv[2]) == 4:
    input_folder = './input_files/input_files_one_species_0/input_files/'
    output_folder = './output/'
    experiment_name = 'one_species_stable_NUM/'
    experiment_number = str(sys.argv[1])

    fit_species = [0]

elif int(sys.argv[2]) == 5:
    input_folder = './input_files/input_files_two_species_auxos_0/input_files/'
    output_folder = './output/'
    experiment_name = 'two_species_auxo_stable_NUM/'
    experiment_number = str(sys.argv[1])

    C = 1e12
    final_epsilon = [1e3 / C, 0.001, 1 / 0.001]

    fit_species = [0, 1]

elif int(sys.argv[2]) == 5:
    input_folder = './input_files/input_files_two_species_auxos_0/input_files/'
    output_folder = './output/'
    experiment_name = 'two_species_auxo_stable_NUM/'
    experiment_number = str(sys.argv[1])

    C = 1e12
    final_epsilon = [1e3 / C, 0.001, 1 / 0.001]
    fit_species = [0, 1]

elif int(sys.argv[2]) == 6:
    input_folder = './input_files/input_files_test/input_files/'
    output_folder = './output/'
    experiment_name = 'weights_test/'
    experiment_number = str(sys.argv[1])
    final_epsilon = [3000]

elif int(sys.argv[2]) == 7:
    input_folder = './input_files/input_files_gerlaud_test/input_files/'
    output_folder = './output/'
    experiment_name = 'weights_test/'
    final_epsilon = [1, 1]

    experiment_number = str(sys.argv[1])

    fit_species = [0]


elif int(sys.argv[2]) == 8:
    input_folder = './input_files/input_files_two_species_spock_manu_2/input_files/'
    output_folder = './output/'
    experiment_name = 'spock_manu_stable_NUM/'
    experiment_number = str(sys.argv[1])
    final_epsilon = [500, 25000, 1 / 1e9]
    fit_species = [0, 1]
    distance_function_mode = 0
    # fit_species = [0, 1, 5, 6, 7]

elif int(sys.argv[2]) == 9:
    input_folder = './input_files/input_files_two_species_spock_manu_2/input_files/'
    output_folder = './output/'
    experiment_name = 'spock_manu_survive_NUM/'
    experiment_number = str(sys.argv[1])
    final_epsilon = [1 / 1e9]
    fit_species = [0, 1]
    distance_function_mode = 2

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
    ABC_algs = algorithms.ABC(t_0, t_end, dt, model_list, population_size=1e10, n_sims_batch=150, fit_species=fit_species, 
        final_epsilon=final_epsilon, distance_function_mode=0, n_distances=3, out_dir=exp_output_folder)
    ABC_algs.run_rejection()

    # rejection_alg.run_rejection()

    print("")
    print("")

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

def ABCSMC_run_gerlaud_test():
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
        if i != 1:
            print(i)
            continue
        model_list.append(model_new)

    fit_species = [0]
    # Run ABC_rejection algorithm
    ABC_algs = algorithms.ABC(t_0, t_end, dt, model_list=model_list, population_size=1000, n_sims_batch=1000, final_epsilon=final_epsilon,
        fit_species=fit_species, distance_function_mode=0, n_distances=2, out_dir=exp_output_folder)

    ABC_algs.run_model_selection_ABC_SMC(alpha=0.5, run_test=2)
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
