from model_space import ModelSpace
import algorithm_utils as alg_utils
import population_modules
import matplotlib.pyplot as plt
import numpy as np
import random
import sys
from scipy import stats
import itertools


def abc_rejection(t_0, t_end, dt, model_list, particles_per_population, n_sims, n_species_fit, n_distances):
    epsilon_array = [ [100, 100], [50, 50], [30, 30], [20, 20], [10, 10], [5, 5], [1, 1] ]

    model_space = ModelSpace(model_list)

    n_simultaneuous_sims = n_sims

    population = 0

    for epsilon in epsilon_array:
        print(population)
        accepted_particles = 0
        all_judgements = []
        all_inputs = []
        all_particles_simmed = []

        while accepted_particles < particles_per_population:
            particle_models = model_space.sample_model_space(n_simultaneuous_sims)  # Model objects in this simulation
            input_params, model_refs = alg_utils.generate_particles(particle_models)          # Extract input parameters and model references

            p = population_modules.Population(n_simultaneuous_sims, t_0, t_end,
                                              dt, input_params, model_refs)

            p.generate_particles()
            p.simulate_particles()
            p.calculate_particle_distances()
            p.accumulate_distances()
            pop_distances = p.get_flattened_distances_list()

            pop_distances = np.reshape(pop_distances, (n_sims, n_species_fit, n_distances))
            part_judgements = alg_utils.check_distances(pop_distances, epsilon_array=epsilon)

            accepted_particles += sum(part_judgements)

            all_inputs = all_inputs + input_params
            all_judgements = all_judgements + part_judgements
            all_particles_simmed = all_particles_simmed + particle_models.tolist()

            print("Accepted particles: ", accepted_particles)

        model_space.generate_model_kdes(all_judgements, all_particles_simmed, all_inputs)
        model_space.update_model_data(all_particles_simmed, all_judgements)

        population += 1
