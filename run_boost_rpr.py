from particle_sim import p_sim
import population_modules
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy import stats
import itertools


class Model:
    ##
    # Each model object possesses all the information needed to simulate the model.
    # Priors should be loaded from input files
    # model ref refers to the index of the model in the cpp module.
    ##
    def __init__(self, model_ref, prior):
        self._model_ref = model_ref
        self._prior = prior
        self._n_params = len(prior)
        self._param_kdes = []

    def sample_from_prior(self, n_sims):
        model_sim_params = []
        for sim in range(0, n_sims):
            sim_params = []

            for param in self._prior:
                lwr_bound = self._prior[param][0]
                upr_bound = self._prior[param][1]
                param_val = np.random.uniform(lwr_bound, upr_bound)
                sim_params.append(param_val)

            model_sim_params.append(sim_params)

        return model_sim_params

    def sample_from_kde(self, n_sims):
        model_sim_params = np.empty(shape=[n_sims, self._n_params])

        for idx, kern in enumerate(self._param_kdes):
            new_params = kern.resample(n_sims)
            model_sim_params[:, idx] = new_params

    def generate_kde(self, params_list):
        self._param_kdes = []
        params_list = np.asarray(params_list)
        for idx, param in enumerate(range(0, self._n_params)):
            param_vals = params_list[:, idx]
            kernel = stats.gaussian_kde(param_vals)
            self._param_kdes.append(kernel)


class ModelSpace:
    ##
    # Model space is initialised with a list of Model objects.
    #
    # This class controls sampling of models, and generation of a population of particles to be simulated
    #
    ##
    def __init__(self, model_list):
        self._model_list = model_list
        
        self._model_refs_list = model_refs_list

    def generate_input_params_from_prior(self):
        for m in _model_list:
            sample_fr






def check_distances(particle_distances, epsilon_array):
    particle_judgements = []

    for particle in particle_distances:
        particle_accept = True
        for species in particle:
            for epsilon_idx, dist in enumerate(species):
                if dist >= epsilon_array[epsilon_idx]:
                    particle_accept = False

        particle_judgements.append(particle_accept)

    return particle_judgements


def gen_model_refs(n_sims):
    model_refs = []
    for i in range(0, n_sims):
        model_refs.append(0)

    return model_refs


def abc_rejection():
    n_sims = 6
    t_0 = 0
    t_end = 1000
    dt = 0.01

    model_rpr_prior_dict = {
        'alpha0': (0, 10),
        'alpha': (0, 3000),
        'n': (0, 10),
        'beta': (0, 10),
    }

    model_lv_prior_dict = {
        'A': (0, 10),
        'B': (0, 10),
        'C': (0, 10),
        'D': (0, 10)
    }

    prior_list = [model_lv_prior_dict, model_rpr_prior_dict]
    model_refs_list = [0, 1]
    
    lv_model = Model(model_lv_prior_dict)
    rpr_model = model_space.Model(model_rpr_prior_dict)


    input_params = rpr_model.sample_from_prior(n_sims)



    model_refs_list = gen_model_refs(n_sims)
    model_space = ModelSpace()


    # input_params = np.asarray(input_params)
    population_num = 0

    eps = 1000

    while eps != 0:
        print(population_num)
        p = population_modules.Population(n_sims, t_0, t_end,
                                          dt, input_params, model_refs_list)

        p.generate_particles()
        p.simulate_particles()
        p.calculate_particle_distances()
        p.accumulate_distances()
        p.get_population_distances()

        particle_distances = p.get_flattened_distances_list()
        # Shape spec, n_sims, n_species, n_distances
        particle_distances = np.reshape(particle_distances, (n_sims, 3, 2))
        epsilon_array = [eps, eps]
        judge_array = check_distances(particle_distances, epsilon_array)

        # Collect parameters of accepted particles
        i = itertools.compress(input_params, judge_array)
        accepted_parameters = [x for x in i]

        if len(accepted_parameters) == 0:
            print("no accepted particles")
            continue

        # Generate new distributions
        rpr_model.generate_kde(accepted_parameters)

        # Sample new parameters
        rpr_model.sample_from_kde(n_sims)

        eps = eps-200
        population_num +=1




if __name__ == "__main__":
    # n_sims = int(sys.argv[1])
    abc_rejection()


    exit()
    sim_array = np.array(p.get_particle_state_list(0))
    sim_array = np.reshape(sim_array, (100001, 6))
    print(np.shape(sim_array[:, 0]))

    # t_array = np.arange(t_0, t_end+dt, dt)
    # print(np.shape(t_array))

    # res = p_sim(n_sim_per_run, t_0, t_end, dt, all_params)
    # print("done")
    # print(np.shape(res))
    # print(np.shape(res[0]))
    # print(np.shape(res[0][0]))
