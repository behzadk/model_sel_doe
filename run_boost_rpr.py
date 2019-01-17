import algorithms
from model_space import Model
import xml.etree.ElementTree as ET
import csv
import os

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
            data_dict[row[0]] =  [float(i) for i in row[1:]]

    return data_dict


def test_param_dicts():

    model_rpr_prior_dict = {
        'alpha0': (0, 1),
        'alpha': (1000, 1000),
        'n': (1, 2),
        'beta': (1, 5),
    }

    model_lv_prior_dict = {
        'A': (0, 10),
        'B': (0, 10),
        'C': (0, 10),
        'D': (0, 10)
    }



  ## Parameters for oscillations
    model_spock_dict = {
        'D': (0.321967092171 , 0.321967092171 ),
        'mux_m': (0.745202817213 , 0.745202817213 ),
        'muc_m': (1.0342287987 , 1.0342287987),
        'Kx': (1.5e-5, 1.5e-5),
        'Kc': (1.5e-5, 1.5e-5),
        'omega_c_max': (0.940702659365, 0.940702659365 ),
        'K_omega': (8.06623324476e-07, 8.06623324476e-07),
        'n_omega': (1.94421248119, 1.94421248119),
        'S0': (4.0, 4.0),
        'gX': (1e12, 1e12),
        'gC': (1e12, 1e12),
        'C0L': (5.14386903636e-05, 5.14386903636e-05),
        'KDL': (7.84665370817e-08, 7.84665370817e-08 ),
        'nL': (3.43520850107, 3.43520850107 ),
        'K1L': (0.000288693934548, 0.000288693934548),
        'K2L': (3019.25026822 , 3019.25026822),
        'ymaxL': (3027.50877859, 3027.50877859),
        'K1T': (285.812160169, 285.812160169),
        'K2T': (75.9278560627, 75.9278560627),
        'ymaxT': (24.07333632, 24.07333632),
        'C0B': (2.64822867774e-05, 2.64822867774e-05),
        'LB': (0.067869016849, 0.067869016849),
        'NB': (0.853610815767, 0.853610815767),
        'KDB': (0.00568144617955, 0.00568144617955),
        'K1B': (0.0481781540971, 0.0481781540971),
        'K2B': (2784.26674844, 2784.26674844),
        'K3B': (34302.0707014, 34302.0707014),
        'ymaxB': (37.659017537, 37.659017537),
        'cgt': (0.088849069628, 0.088849069628),
        'k_alpha_max': (5.27450373726e-17, 5.27450373726e-17),
        'k_beta_max': (6.95020587806e-16, 6.95020587806e-16 ),
    }

    ## Spock init oscillations
    spock_init = {
        'X': (4.25939129081e+11, 4.25939129081e+11),
        'C': (2.81485982763e+11, 2.81485982763e+11),
        'S': (4, 4),
        'B': (0, 0),
        'A': (1e-10, 1e-10)
    }

    ## Spock Prior
    model_spock_dict = {
        'D': (0.01, 0.5),
        'mux_m': (0.4, 3.0),
        'muc_m': (0.4, 3.0),
        'Kx': (1.5e-5, 1.5e-5),
        'Kc': (1.5e-5, 1.5e-5),
        'omega_c_max': (0.5, 2),
        'K_omega': (1e-7, 1e-6),
        'n_omega': (1, 2),
        'S0': (4.0, 4.0),
        'gX': (1e12, 1e12),
        'gC': (1e12, 1e12),
        'C0L': (5e-5, 1e-4),
        'KDL': (3e-8, 8e-8),
        'nL': (2.0, 3.8),
        'K1L': (0.00015, 0.0004),
        'K2L': (2500, 5000),
        'ymaxL': (2500,  5000),
        'K1T': (0, 5000),
        'K2T': (0,  100),
        'ymaxT': (16, 27),
        'C0B': (2e-5, 1e-4),
        'LB': (0, 1e-2),
        'NB': (0.7, 1.1),
        'KDB': (0.002, 0.010),
        'K1B': (0.0, 0.08),
        'K2B': (800, 5000),
        'K3B': (1000, 50000),
        'ymaxB': (100, 500),
        'cgt': (0.0, 0.1),
        'k_alpha_max': (1e-22, 1e-15),
        'k_beta_max': (1e-22, 1e-15),
    }



def ABC_rejection():
    # Set time points
    t_0 = 0
    t_end = 5000
    dt = 1

    # Set input file and output directories
    input_folder = './input_files_two_species/'
    output_folder = './output_two_species_bigspace/'

    # Load models from input files
    model_list = []
    for i in range(int((len(os.listdir(input_folder))/2))):
        input_params = input_folder + "params_" + str(i) + ".csv"
        input_init_species = input_folder + "species_" + str(i) + ".csv"
        init_params = import_input_file(input_params)
        init_species = import_input_file(input_init_species)
        model_new = Model(i, init_params, init_species)
        model_list.append(model_new)

    # Run ABC_rejecction algorithm
    ABC = algorithms.ABC_rejection(t_0, t_end, dt, model_list, 1e6, 300, 2, 2, output_folder)
    ABC.run_abc_rejection()
    print("")

def ABCSMC():
    pass

if __name__ == "__main__":
    ABC_rejection()




