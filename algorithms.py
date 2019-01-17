from model_space import ModelSpace
import algorithm_utils as alg_utils
import population_modules
import numpy as np
import os
import csv
from timeit import default_timer as timer
import sys
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import plotting

def blockPrinting(func):
    def func_wrapper(*args, **kwargs):
        # block all printing to the console
        sys.stdout = open(os.devnull, 'w')
        # call the method in question
        func(*args, **kwargs)
        # enable all printing to the console
        sys.stdout = sys.__stdout__

    return func_wrapper


class ABC_rejection:
    def __init__(self, t_0, t_end, dt,
                 model_list, population_size, n_sims_batch,
                 n_species_fit, n_distances, out_dir):
        self.t_0 = t_0
        self.t_end = t_end
        self.dt = dt
        self.model_list = model_list

        self.population_size = population_size
        self.n_sims_batch = n_sims_batch
        self.n_species_fit = n_species_fit
        self.n_distances = n_distances

        self.epsilon_array = [ [100, 10] ]
        # self.epsilon_array = [ [1e12, 1e12], [1e9, 1e12], [1e7, 1e12], [1e6, 1e12], [1e5, 1e12], [1e4, 1e12], [1e3, 1e12], [1e2, 1e12], [1e1, 1e12] ]


        # Init model space
        self.model_space = ModelSpace(model_list)

        self.out_dir = out_dir


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

    def plot_accepted_particles(self, out_dir, pop_num, batch_num, part_judgements, init_states, model_refs):
        out_path = out_dir + "Population_" + str(pop_num) + "_batch_" + str(batch_num) + "_plots.pdf"

        # Make new pdf
        pdf = PdfPages(out_path)

        # Iterate all particles in batch
        for idx, is_accepted in enumerate(part_judgements):
            if is_accepted:
                state_list = self.pop_obj.get_particle_state_list(idx)
                time_points = self.pop_obj.get_timepoints_list()

                state_list = np.reshape(state_list, (len(time_points), len(init_states[idx])))
                model_ref = model_refs[idx]

                plot_species = [i for i in range(self.n_species_fit)]
                plotting.plot_simulation(pdf, model_ref, state_list, time_points, plot_species)
        pdf.close()


    def plot_all_particles(self, out_dir, pop_num, batch_num, part_judgements, init_states, model_refs):
        out_path = out_dir + "Population_" + str(pop_num) + "_batch_" + str(batch_num) + "_plots.pdf"

        # Make new pdf
        pdf = PdfPages(out_path)

        # Iterate all particles in batch
        for idx, is_accepted in enumerate(part_judgements):
            state_list = self.pop_obj.get_particle_state_list(idx)
            time_points = self.pop_obj.get_timepoints_list()

            try:
                state_list = np.reshape(state_list, (len(time_points), len(init_states[idx])))

            except(ValueError):
                continue

            model_ref = model_refs[idx]

            plot_species = [i for i in range(self.n_species_fit)]
            plotting.plot_simulation(pdf, model_ref, state_list, time_points, plot_species)
        pdf.close()

    def write_accepted_particle_distances(self, out_dir, model_refs, part_judgments, distances):
        out_path = out_dir + "distances.csv"
        with open(out_path, 'a') as out_csv:
            wr = csv.writer(out_csv)
            for idx, is_accepted in enumerate(part_judgments):
                record_vals = [idx, model_refs[idx]]
                if is_accepted:
                    for n in range(self.n_species_fit):
                        for d in distances[idx][n]:
                            record_vals.append(d)

                    wr.writerow(record_vals)

    def write_particle_distances(self, out_dir, model_refs, distances):
        out_path = out_dir + "distances.csv"
        with open(out_path, 'a') as out_csv:
            wr = csv.writer(out_csv)
            for idx, m_ref in enumerate(model_refs):
                record_vals = [idx, m_ref]
                for n in range(self.n_species_fit):
                    for d in distances[idx][n]:
                        record_vals.append(d)

                wr.writerow(record_vals)

    def run_abc_rejection(self):
        population_number = 0

        for epsilon in self.epsilon_array:
            folder_name = self.out_dir + "Population_" + str(population_number) + "/"

            try:
                os.mkdir(folder_name)
            except FileExistsError:
                pass

            accepted_particles_count = 0
            total_sims = 0

            all_judgements = []
            all_inputs = []
            all_particles_simmed = []

            batch_num = 0

            while accepted_particles_count < self.population_size:

                # 1. Sample from model space
                particle_models = self.model_space.sample_model_space(self.n_sims_batch)  # Model objects in this simulation

                # 2. Sample particles for each model
                init_states, input_params, model_refs = alg_utils.generate_particles(particle_models)          # Extract input parameters and model references

                # 3. Simulate population
                self.pop_obj = population_modules.Population(self.n_sims_batch, self.t_0, self.t_end,
                                                  self.dt, init_states, input_params, model_refs)
                self.pop_obj.generate_particles()
                self.pop_obj.simulate_particles()

                all_particles_simmed = all_particles_simmed + particle_models.tolist()

                # 3. Calculate distances for population
                self.pop_obj.calculate_particle_distances()

                self.pop_obj.accumulate_distances()
                batch_distances = self.pop_obj.get_flattened_distances_list()


                if len(batch_distances) == 0:
                    continue

                print("")

                # 4. Accept or reject particles
                batch_distances = np.reshape(batch_distances, (self.n_sims_batch, self.n_species_fit, self.n_distances))
                batch_part_judgements = alg_utils.check_distances(batch_distances, epsilon_array=epsilon)

                self.write_accepted_particle_distances(folder_name, model_refs, batch_part_judgements, batch_distances)

                # Write accepted parameters every batch
                self.model_space.write_accepted_particle_params(folder_name+'model_accepted_params/',
                                                                particle_models.tolist(),
                                                                batch_part_judgements, input_params)



                # all_inputs = all_inputs + input_params
                # all_judgements = all_judgements + batch_part_judgements

                # plot accepted particles
                if sum(batch_part_judgements) > 0:
                    self.plot_accepted_particles(folder_name + 'simulation_plots/', population_number,
                                                 batch_num, batch_part_judgements,
                                                 init_states, model_refs)

                # self.plot_all_particles(folder_name, population_number,
                #                                  batch_num, batch_part_judgements,
                #                                  init_states, model_refs)

                # print(pop_distances)
                accepted_particles_count += sum(batch_part_judgements)
                total_sims += len(model_refs)

                print("Population: ", population_number, "Accepted particles: ", accepted_particles_count, "Total simulations: ", total_sims)

                self.model_space.update_model_population_sample_data(particle_models.tolist(), batch_part_judgements)
                self.model_space.model_space_report(folder_name, batch_num)

                batch_num += 1


            # 5. Generate new distributions for models
            self.model_space.generate_model_kdes(all_judgements, all_particles_simmed, all_inputs)
            self.model_space.update_model_sample_probabilities()

            # 6. Write the parameters of accepted particles
            try:
                os.mkdir(folder_name)
            except FileExistsError:
                pass

            self.model_space.model_space_report(folder_name)
            self.model_space.write_accepted_particle_params(folder_name, all_particles_simmed, all_judgements, all_inputs)
            population_number += 1

class ABC_SMC:
    pass