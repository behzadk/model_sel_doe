from model_space import ModelSpace
import algorithm_utils as alg_utils
import population_modules
import numpy as np
import os
import csv
import matplotlib.pyplot as plt

class ABC_rejection:
    def __init__(self, t_0, t_end, dt, model_list, population_size, n_sims_batch, n_species_fit, n_distances):
        self.t_0 = t_0
        self.t_end = t_end
        self.dt = dt
        self.model_list = model_list

        self.population_size = population_size
        self.n_sims_batch = n_sims_batch
        self.n_species_fit = n_species_fit
        self.n_distances = n_distances

        self.epsilon_array = [ [1e6, 1e6], [1e5, 1e5], [1e4, 1e4], [1e3, 1e3], [1e2, 1e2], [1e1, 1e1] ]

        # Init model space
        self.model_space = ModelSpace(model_list)

    def test_generate_init_states(self):
        spock_init = {
            'X': (1e11, 1e11),
            'C': (1e11, 1e12),
            'S': (4, 4),
            'B': (0, 0),
            'A': (0, 0)
        }

        rpr_init = {
            'X': (1e11, 1e11),
            'C': (0, 0),
            'S': (0, 0),
            'B': (0, 2),
            'G': (0, 5),
            'L': (0, 10),
        }

        init_state = []

        for s in range(self.n_sims_batch):
            sim_init = []
            for param in spock_init:
                lwr_bound = spock_init[param][0]
                upr_bound = spock_init[param][1]
                param_val = np.random.uniform(lwr_bound, upr_bound)
                sim_init.append(param_val)

            init_state.append(sim_init)


        return init_state

    def run_abc_rejection(self):
        population = 0

        for epsilon in self.epsilon_array:
            print(population)
            accepted_particles = 0
            all_judgements = []
            all_inputs = []
            all_particles_simmed = []

            while accepted_particles < self.population_size:

                # 1. Sample from model space
                particle_models = self.model_space.sample_model_space(self.n_sims_batch)  # Model objects in this simulation

                # 2. Sample particles for each model
                init_state, input_params, model_refs = alg_utils.generate_particles(particle_models)          # Extract input parameters and model references
                # input_state_init = self.test_generate_init_states()

                # 3. Simulate population
                p = population_modules.Population(self.n_sims_batch, self.t_0, self.t_end,
                                                  self.dt, init_state, input_params, model_refs)


                p.generate_particles()

                p.simulate_particles()

                # 3. Calculate distances for population
                p.calculate_particle_distances()
                print("calculated distances")
                p.accumulate_distances()
                pop_distances = p.get_flattened_distances_list()

                # 4. Accept or reject particles
                pop_distances = np.reshape(pop_distances, (self.n_sims_batch, self.n_species_fit, self.n_distances))
                part_judgements = alg_utils.check_distances(pop_distances, epsilon_array=epsilon)

                accepted_particles += sum(part_judgements)

                all_inputs = all_inputs + input_params
                all_judgements = all_judgements + part_judgements
                all_particles_simmed = all_particles_simmed + particle_models.tolist()

                print("Accepted particles: ", accepted_particles)
                # print(pop_distances)
                sim = p.get_particle_state_list(0)
                print(np.shape(sim))
                n_species = np.shape(init_state)[1]
                print(len(sim))
                print(n_species)

                sim = np.reshape(sim, (int(len(sim)/(n_species)), n_species))
                time_points = np.arange(self.t_0, self.t_end, self.dt)

                plt.plot(time_points, sim[:,0])
                plt.plot(time_points, sim[:,1])
                plt.yscale('log')
                plt.show()


                exit()
            # 5. Generate new distributions for models
            self.model_space.generate_model_kdes(all_judgements, all_particles_simmed, all_inputs)
            self.model_space.update_model_data(all_particles_simmed, all_judgements)
            self.model_space.model_space_report()

            # 6. Write the parameters of accepted particles
            folder_name = "./Population_" + str(population)
            try:
                os.mkdir(folder_name)
            except FileExistsError:
                pass

            self.model_space.write_accepted_particle_params(folder_name, all_particles_simmed, all_judgements, all_inputs)
            population += 1

