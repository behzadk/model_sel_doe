import numpy as np
import pandas as pd

import glob
import pickle

from importlib.machinery import SourceFileLoader
import algorithm_utils as alg_utils

def compute_particle_weight(model_space, particle):
    model_params_prior = particle.curr_model._params_prior
    model_init_species_prior = particle.curr_model._init_species_prior

    particle_params = particle.curr_params
    particle_init_species = particle.curr_init_state

    this_model = particle.curr_model

    particle_prior_prob = 1
    for idx, id in enumerate(sorted(model_params_prior, key=str.lower)):
        x = 1.0
        this_param_prior = model_params_prior[id]

        if this_param_prior[0] == this_param_prior[1]:
            x = 1.0

        else:
            x = alg_utils.get_pdf_uniform(this_param_prior[0], this_param_prior[1], particle_params[idx])

        particle_prior_prob = particle_prior_prob * x

    for idx, id in enumerate(model_init_species_prior):
        x = 1.0

        this_init_species_prior = model_init_species_prior[id]

        if this_init_species_prior[0] == this_init_species_prior[1]:
            x = 1.0

        else:
            x = alg_utils.get_pdf_uniform(this_init_species_prior[0], this_init_species_prior[1],
                                      particle_init_species[idx])

        particle_prior_prob = particle_prior_prob * x

    # Model prior of 1 assumes uniform model prior
    numerator = particle_prior_prob

    s_1 = 0
    for m in model_space._model_list:
        s_1 += m.prev_margin * model_space.get_model_kernel_pdf(this_model, m, model_space.model_kernel,
                                                         len(model_space._model_list), model_space.dead_models_count)
    # Particle denominator
    s_2 = 0.0
    for p_j in model_space.accepted_particles:
        if this_model._model_ref == p_j.prev_model._model_ref:
            non_constant_idxs = [idx for idx, key in enumerate(sorted(model_params_prior, key=str.lower)) if model_params_prior[key][0] != model_params_prior[key][1]]
            
            kernel_params_pdf = alg_utils.get_parameter_kernel_pdf(particle_params, p_j.prev_params,
                                                                   this_model.param_kernels, non_constant_idxs, this_model._params_prior, p_j.param_aux_info, use_uniform_kernel=False, use_normal_kernel=True)
           
            non_constant_idxs = [idx for idx, key in enumerate(model_init_species_prior) if model_init_species_prior[key][0] != model_init_species_prior[key][1]]
            
            kernel_init_state_pdf = alg_utils.get_parameter_kernel_pdf(particle_init_species,
                                                                       p_j.prev_init_state,
                                                                       this_model.init_state_kernels, non_constant_idxs,  this_model._params_prior, p_j.init_state_aux_info, use_uniform_kernel=False, use_normal_kernel=True)
            print("kernel init state: ", kernel_init_state_pdf)
            print("kernel params pdf: ", kernel_params_pdf)

            kernel_pdf = kernel_params_pdf * kernel_init_state_pdf
            s_2 = s_2 + p_j.prev_weight * kernel_pdf

    print(numerator)
    print(s_1)
    print(s_2)
    particle.curr_weight = this_model.prev_margin * numerator / (s_1 * s_2)




def load_pop_1_pickle():
    pop_1_pickle_path = "/media/behzad/DATA/experiments_data/BK_manu_data/three_species_stable_SMC_5/BK_manu_three_species_SMC_3_3005_pop_2/Population_2/checkpoint.pickle"
    with open(pop_1_pickle_path, "rb") as input_file:
         pop_1 = pickle.load(input_file)

    # for p in pop_1.population_accepted_particles:
    #     compute_particle_weight(pop_1.model_space, p)

    # print("normalising weights")
    # pop_1.model_space.normalize_particle_weights()


    for p in pop_1.population_accepted_particles:
        if p.curr_model._model_ref == 2194:
            print("normlizd weight: ", p.curr_weight)
            compute_particle_weight(pop_1.model_space, p)
            print(p.curr_weight)
            print("")



def find_latest_population_path(exp_folder):
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
                return f

        idx += 1

    return None


def find_latest_pickles_path(exp_folder):
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
                return pickles

        idx += 1

    return None

def check_model_2194():
    data_dir = '/media/behzad/DATA/experiments_data/BK_manu_data/three_species_stable_SMC_5/'
    
    exp_dirs = glob.glob(data_dir + "**/")
    print(exp_dirs)
    population_pickles_list = []

    pickle_path_list = []

    experiment_name = "chunk_"
    exp_dirs = [x for x in exp_dirs if "chunk" not in x][:60]
    split_pickle_paths = np.array_split(exp_dirs, 3)

    total_acc = 0
    model_marginals = []
    n_accepteds = []



    for idx, d in enumerate(split_pickle_paths[0]):
        if "chunk_" in d:
            continue

        if idx == 5:
            print(d)
            continue
        n_dir = find_latest_population_path(d)
        pickle_path_list.append(n_dir)
        x = pd.read_csv(n_dir + 'model_space_report.csv')
        model_marginal = x.loc[x['model_idx'] == 2194]['model_marginal'].values[0]
        n_accepted = x.loc[x['model_idx'] == 2194]['accepted_count'].values[0]


        model_marginals.append(model_marginal)
        n_accepteds.append(n_accepted)
        # accepted = model
        print(model_marginal)

    print("")
    print(np.mean(model_marginals))
    print(np.mean(n_accepteds))

    exit()
    pickle_path_list = pickle_path_list
    split_pickle_paths = np.array_split(pickle_path_list, 3)

    print(split_pickle_paths)


def check_model_2194_pickle():
    data_dir = '/media/behzad/DATA/experiments_data/BK_manu_data/three_species_stable_SMC_5/'
    exp_dirs = glob.glob(data_dir + "**/")
    population_pickles_list = []

    pickle_path_list = []

    experiment_name = "chunk_"
    exp_dirs = [x for x in exp_dirs if "chunk" not in x][:60]
    split_pickle_paths = np.array_split(exp_dirs, 3)

    total_acc = 0
    model_marginals = []
    n_accepteds = []



    for idx, d in enumerate(split_pickle_paths[0]):
        if "chunk_" in d:
            continue

        if idx == 5:
            p_path = find_latest_pickles_path(d)[0]
            print(p_path)
            with open(p_path, "rb") as input_file:
                 e = pickle.load(input_file)
            pickle_path_list.append(n_dir)


if __name__ == "__main__":
    load_pop_1_pickle()