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
import seaborn as sns; sns.set()
from scipy.optimize import fsolve
import classification

import sys

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

def repressilator_test():
    # http://people.cs.uchicago.edu/~lebovitz/Eodesbook/stabeq.pdf
    # http: // www.math.psu.edu /tseng/class /Math251 / Notes-PhasePlane.pdf

    t_0 = 0
    t_end = 237
    dt = 0.01

    rpr_params_prior = {
        'alpha0': (1, 1),
        'alpha': (1000, 1000),
        'n': (2, 2),
        'beta': (5, 5),
    }

    init_states_prior = {
        'm1': (1, 1),
        'm2': (1, 1),
        'm3': (1, 1),
        'p1': (10, 10),
        'p2': (10, 10),
        'p3': (10, 10)
    }

    n_sims_batch = 1
    rpr_model = Model(0, rpr_params_prior, init_states_prior)
    model_space = ModelSpace([rpr_model])

    particle_models = model_space.sample_model_space(n_sims_batch)  # Model objects in this simulation

    init_states, input_params, model_refs = alg_utils.generate_particles(
        particle_models)  # Extract input parameters and model references

    pop_obj = population_modules.Population(n_sims_batch, t_0, t_end,
                                              dt, init_states, input_params, model_refs)
    pop_obj.generate_particles()
    pop_obj.simulate_particles()

    sol = pop_obj.get_particle_state_list(0)
    t = pop_obj.get_timepoints_list()

    out = pop_obj.get_particle_eigenvalues(0)

    eig_vals = out[0]
    eig_vec = out[1]

    n_species = len(init_states[0])
    eig_val_product = classification.eigenvalue_product(eig_vals)
    sum_eigenvalues = classification.sum_eigenvalues(eig_vals)
    trace = pop_obj.get_particle_trace(0)

    jac = np.asarray(pop_obj.get_particle_jacobian(0), dtype=np.float64)
    jac = np.reshape(jac, (n_species, n_species))

    sign, logdet = np.linalg.slogdet(jac)
    det = sign * np.exp(logdet)

    print("")
    print("Det: ", det)
    print("sum eig", sum_eigenvalues)
    print("Trace", trace)
    print("Eigen product", eig_val_product)
    print("")

    for idx, vec in enumerate(eig_vec):
        vec = np.reshape(vec, (n_species, 2))
        print("Eigenvalue: ", eig_vals[idx])
        print("Vector magnitude", np.linalg.norm(vec[:, 0], ord=1))
        print("")

    sol = np.reshape(sol, (len(t), len(init_states_prior)))

    m1 = sol[:, 0]
    m2 = sol[:, 1]
    m3 = sol[:, 2]

    p1 = sol[:, 3]
    p2 = sol[:, 4]
    p3 = sol[:, 5]

    plt.plot(t, m1)
    plt.plot(t, m2)
    plt.plot(t, m3)

    plt.plot(t, p1)
    plt.plot(t, p2)
    plt.plot(t, p3)
    plt.show()


def eig_test():
    # Set time points
    t_0 = 0
    t_end = 5000
    dt = 0.1

    input_folder = './input_files_two_species/'
    output_folder = './output/'
    experiment_name = 'two_species_big_NUM/'
    experiment_number = str(0)
    experiment_folder = experiment_name.replace('NUM', experiment_number)

    output_folder = output_folder + experiment_folder

    # Load models from input files
    model_list = []
    for i in range(int((len(os.listdir(input_folder))/2))):
        input_params = input_folder + "params_" + str(i) + ".csv"
        input_init_species = input_folder + "species_" + str(i) + ".csv"
        init_params = import_input_file(input_params)
        init_species = import_input_file(input_init_species)
        if i == 30:
            model_new = Model(i, init_params, init_species)
            model_list.append(model_new)

    model_space = ModelSpace(model_list)
    particle_models = model_space.sample_model_space(1)  # Model objects in this simulation

    init_states, input_params, model_refs = alg_utils.generate_particles(
        particle_models)  # Extract input parameters and model references

    n_sims_batch = 1

    pop_obj = population_modules.Population(n_sims_batch, t_0, t_end,
                                              dt, init_states, input_params, model_refs)
    pop_obj.generate_particles()
    pop_obj.simulate_particles()

    # 3. Calculate distances for population
    pop_obj.calculate_particle_distances()
    pop_obj.accumulate_distances()
    batch_distances = pop_obj.get_flattened_distances_list()
    batch_distances = np.reshape(batch_distances, (n_sims_batch, 2, 2))

    # 4. Accept or reject particles
    batch_part_judgements = alg_utils.check_distances(batch_distances, epsilon_array=[100, 10])

    n_species = len(init_states[0])

    out = pop_obj.get_particle_eigenvalues(0)

    eig_vals = out
    print(np.shape(eig_vals))
    [print(i[0]) for i in eig_vals]

    real_eigs = [i[0] for i in eig_vals]
    jac = np.asarray(pop_obj.get_particle_jacobian(0))
    jac = np.reshape(jac, (n_species, n_species))


    sign, logdet = np.linalg.slogdet(jac)
    py_det = sign * np.exp(logdet)
    cpp_det = pop_obj.get_particle_det(0)


    py_eig = np.linalg.eigvals(jac)
    py_trace = sum([jac[i][i] for i in range(n_species)])

    print("Python eigenvalue results:")
    print("\tpy_eig_product: ", np.product(py_eig))
    print("\tsum_py_eig: ", sum(py_eig))
    print("\tTrace: ", py_trace)
    print("\tDet: ", py_det)
    print("")


    eig_val_product = classification.eigenvalue_product(eig_vals)
    sum_eigenvalues = classification.sum_eigenvalues(eig_vals)
    trace = pop_obj.get_particle_trace(0)

    print("")

    print("cpp eigenvalue results: ")
    print("\tEigen product: ", eig_val_product)
    print("\tcpp eig real product: ", np.product(real_eigs))
    print("\tsum eig: ", sum_eigenvalues)
    print("\tTrace: ", trace)
    print("\tDet: ", cpp_det)
    print("\tratio: ", cpp_det/eig_val_product)
    print("")

    print("Python eigenvalues")
    for idx, vec in enumerate(py_eig):
        print("\tEigenvalue: ", py_eig[idx])
        # print("Vector magnitude", np.linalg.norm(vec[:, 0], ord=1))

    print("Cpp eigenvalues")
    for idx, vec in enumerate(eig_vals):
        # vec = np.reshape(vec, (n_species, 2))
        print("\tEigenvalue: ", eig_vals[idx])
        # print("Vector magnitude", np.linalg.norm(vec[:, 0], ord=1))
        print("")


    if batch_part_judgements[0]:
        sol = pop_obj.get_particle_state_list(0)
        t = pop_obj.get_timepoints_list()

        sol = np.reshape(sol, (len(t), n_species))
        N_1 = sol[:, 0]
        N_2 = sol[:, 1]

        plt.plot(t, N_1)
        plt.plot(t, N_2)
        plt.yscale('log')
        plt.show()

def eig_classification_test():
    # Set time points
    t_0 = 0
    t_end = 5000
    dt = 0.1

    input_folder = './input_files_two_species/'
    output_folder = './output/'
    experiment_name = 'two_species_big_NUM/'
    experiment_number = str(0)
    experiment_folder = experiment_name.replace('NUM', experiment_number)

    output_folder = output_folder + experiment_folder

    # Load models from input files
    model_list = []
    for i in range(int((len(os.listdir(input_folder))/2))):
        input_params = input_folder + "params_" + str(i) + ".csv"
        input_init_species = input_folder + "species_" + str(i) + ".csv"
        init_params = import_input_file(input_params)
        init_species = import_input_file(input_init_species)
        if i == 30:
            model_new = Model(i, init_params, init_species)
            model_list.append(model_new)

    model_space = ModelSpace(model_list)
    particle_models = model_space.sample_model_space(1)  # Model objects in this simulation

    init_states, input_params, model_refs = alg_utils.generate_particles(
        particle_models)  # Extract input parameters and model references

    n_sims_batch = 1

    pop_obj = population_modules.Population(n_sims_batch, t_0, t_end,
                                              dt, init_states, input_params, model_refs)
    pop_obj.generate_particles()
    pop_obj.simulate_particles()
    n_species = len(init_states[0])

    jac = np.asarray(pop_obj.get_particle_jacobian(0))
    jac = np.reshape(jac, (n_species, n_species))

    eigenvalues = np.linalg.eigvals(jac)
    eigenvalues = [ [i.real, i.imag] for i in eigenvalues]

    classification.classify_eigensystem(eigenvalues)

    sign, logdet = np.linalg.slogdet(jac)


    py_trace = sum([jac[i][i] for i in range(n_species)])




def steady_state_test(expnum):
    # Set time points
    t_0 = 0
    t_end = 250
    dt = 1

    input_folder = './input_files_two_species/'
    output_folder = './output/'
    experiment_name = 'two_species_big_NUM/'
    experiment_number = str(0)
    experiment_folder = experiment_name.replace('NUM', experiment_number)

    output_folder = output_folder + experiment_folder

    # Load models from input files
    model_list = []
    for i in range(int((len(os.listdir(input_folder)) / 2))):
        input_params = input_folder + "params_" + str(i) + ".csv"
        input_init_species = input_folder + "species_" + str(i) + ".csv"
        init_params = import_input_file(input_params)
        init_species = import_input_file(input_init_species)

        if i != 30:
            model_new = Model(i, init_params, init_species)
            model_list.append(model_new)

    model_space = ModelSpace(model_list)
    particle_models = model_space.sample_model_space(1)  # Model objects in this simulation

    init_states, input_params, model_refs = alg_utils.generate_particles(
        particle_models)  # Extract input parameters and model references

    n_sims_batch = 1

    pop_obj = population_modules.Population(n_sims_batch, t_0, t_end,
                                            dt, init_states, input_params, model_refs)
    pop_obj.generate_particles()
    pop_obj.simulate_particles()
    n_species = len(init_states[0])
    time_points = pop_obj.get_timepoints_list()
    state_list = pop_obj.get_particle_state_list(0)
    try:
        state_list = np.reshape(state_list, (len(time_points), n_species))
    except ValueError:
        return 0

    eigenvalues_t = []
    for i in range(1, len(time_points)):
        state = state_list[i]
        state = state.tolist()

        jac = pop_obj.get_particle_jacobian(state, 0)

        #
        # jac = pop_obj.get_particle_jacobian(state, 0)
        jac = np.reshape(jac, (n_species, n_species))
        try:
            eigenvalues = np.linalg.eigvals(jac)

        except(np.linalg.LinAlgError) as e:
            eigenvalues = [np.nan for i in range(n_species)]

        eigenvalues = [[i.real, i.imag] for i in eigenvalues]

        real_parts = [i[0] for i in eigenvalues]
        abs_real = [abs(i) for i in real_parts]
        try:
            max_idx = real_parts[np.nanargmax(abs_real)]
            eigenvalues_t.append(max_idx)

        except:
            print(time_points[i], "all nans")
            return 0
            eigenvalues_t.append(max_idx)


    const_eig_idx = None

    for idx, val in enumerate(eigenvalues_t):
        if idx == 0:
            continue

        if val == eigenvalues_t[idx-1]:
            const_eig_idx = time_points[idx]


    # Final eigenvalues
    final_state = state_list[-1]
    res = fsolve(alg_utils.fsolve_conversion, final_state, fprime=alg_utils.fsolve_jac_conversion,
                 args=(pop_obj, 0, n_species), full_output=True)

    steady_state = res[0]
    steady_state[0] = steady_state[0] + 0.1

    ier = res[2]
    fsolve_error = ier

    steady_state = steady_state.tolist()

    jac = pop_obj.get_particle_jacobian(steady_state, 0)
    jac = np.reshape(jac, (n_species, n_species))

    try:
        eigenvalues = np.linalg.eigvals(jac)

    except(np.linalg.LinAlgError) as e:
        return 0

    eigenvalues = [[i.real, i.imag] for i in eigenvalues]

    real_parts = [i[0] for i in eigenvalues]
    imag_parts = [i[1] for i in eigenvalues]

    all_negative =  all(i < 0 for i in real_parts)
    all_real = all(i==0.0 for i in imag_parts)
    num_conjugate_pairs = len(classification.get_conjugate_pairs(eigenvalues))


    plt.rcParams['figure.figsize'] = [15, 10]

    font = {'size': 15, }
    axes = {'labelsize': 'large', 'titlesize': 'large'}

    mpl.rc('font', **font)
    mpl.rc('axes', **axes)


    fig, (ax1, ax2) = plt.subplots(ncols=2)
    # ax1 = axes[0, 0]
    # ax2 = axes[0, 1]
    # ax3 = axes[1, 0]
    # ax4 = axes[1, 1]

    sns.lineplot(time_points[1:], eigenvalues_t, ax=ax1)

    sns.lineplot(time_points, state_list[:, 0], ax=ax2)
    sns.lineplot(time_points, state_list[:, 1], ax=ax2)
    # sns.lineplot(time_points, state_list[:, 2], ax=ax3)
    # sns.lineplot(time_points, state_list[:, 3], ax=ax4)

    print(type(const_eig_idx))

    ax1.set(yscale='symlog', xlabel='time', ylabel='Leading eigenvalue')
    ax2.set(yscale='symlog', xlabel='time', ylabel='Population (num cells)')
    text = ("all negative parts: ", str(all_negative), "\n", "all real parts: ", str(all_real), "\n",
            "num conjugate pairs: ", str(num_conjugate_pairs))
    ax1.text(0.85, 0.85, text, fontsize=10) #add text

    plt.suptitle('Time course max eig and two species population')

    plt.savefig("./output/"+str(expnum)+".pdf")
    plt.close()



def ABC_rejection():
    # Set time points
    t_0 = 0
    t_end = 1000
    dt = 1
    
    input_folder = './input_files_three_species/priors/'
    output_folder = './output/'
    experiment_name = 'three_species_stable_NUM/'
    experiment_number = str(sys.argv[1])
    
    experiment_folder = experiment_name.replace('NUM', experiment_number)

    output_folder = output_folder + experiment_folder

    fit_species = [0, 1, 2]

    try:
        os.mkdir(output_folder)
        
    except FileExistsError:
        pass

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
    rejection_alg = algorithms.Rejection(t_0, t_end, dt, model_list, 1e6, 288, fit_species, 3, output_folder)
    rejection_alg.run_rejection()
    print("")

def ABCSMC():
    # Set time points
    t_0 = 0
    t_end = 1000
    dt = 1

    input_folder = './input_files_three_species/'
    output_folder = './output/'
    experiment_name = 'three_species_stable_ABSMC_NUM/'
    experiment_number = str(0)
    experiment_folder = experiment_name.replace('NUM', experiment_number)

    output_folder = output_folder + experiment_folder

    try:
        os.mkdir(output_folder)

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
        if i == 30:
            model_list.append(model_new)

    # Run ABC_rejecction algorithm
    rejection_alg = algorithms.Rejection(t_0, t_end, dt, model_list, 1e6, 5000, 2, 3, output_folder)

    parameters_to_optimise = ['D', 'N_x', 'N_c']
    rejection_alg.run_paramter_optimisation(parameters_to_optimise)


def random_jacobian():
    # Set time points
    t_0 = 0
    t_end = 5000
    dt = 1

    input_folder = './input_files_two_species/'
    output_folder = './output/'
    experiment_name = 'two_species_stable_NUM/'
    experiment_number = str(5)
    experiment_folder = experiment_name.replace('NUM', experiment_number)

    output_folder = output_folder + experiment_folder

    try:
        os.mkdir(output_folder)

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

        if i == 23:
            model_list.append(model_new)

    # Run ABC_rejecction algorithm
    rejection_alg = algorithms.Rejection(t_0, t_end, dt, model_list, 1e6, 10000, 2, 3, output_folder)
    rejection_alg.run_rejection()
    print("")



if __name__ == "__main__":
    # for i in range(50):
    #     steady_state_test(i)
    # ABCSMC()
    ABC_rejection()
    # eig_classification_test()
    # repressilator_test()
    exit()
    ABC_rejection()




