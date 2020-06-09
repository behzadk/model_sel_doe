from model_space import ModelSpace
import algorithm_utils as alg_utils

# import population_modules
import ode_sim as population_modules
import numpy as np
import os
import csv
from timeit import default_timer as timer
import sys
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import plotting
from scipy.optimize import fsolve
import copy
import pickle
import math
import time

target_data = np.random.geometric(0.5, 100)
target_data_s1 = np.sum(target_data)
target_data_t1 = np.sum(np.log([float(np.math.factorial(x)) for x in target_data]))

def blockPrinting(func):
    def func_wrapper(*args, **kwargs):
        # block all printing to the console
        sys.stdout = open(os.devnull, 'w')
        # call the method in question
        func(*args, **kwargs)
        # enable all printing to the console
        sys.stdout = sys.__stdout__

    return func_wrapper


class ABC:
    def __init__(self, t_0, t_end, dt, exp_num,
                 model_list, population_size, n_sims_batch,
                 fit_species, final_epsilon, initial_epsilon, distance_function_mode, n_distances, abs_tol, rel_tol, out_dir):
        self.t_0 = t_0
        self.t_end = t_end
        self.dt = dt
        self.model_list = model_list

        self.exp_num = exp_num

        self.population_size = population_size
        self.population_accepted_count = 0
        self.population_total_simulations = 0
        self.n_sims_batch = n_sims_batch
        self.fit_species = fit_species
        self.distance_function_mode = distance_function_mode
        self.n_distances = n_distances

        self.population_accepted_particle_distances = []
        self.population_accepted_particles_count = 0

        self.population_model_refs = []
        self.population_judgements = []

        self.population_accepted_particles = []

        self.abs_tol = abs_tol
        self.rel_tol = rel_tol

        # self.epsilon = [100, 10, 1e4]
        # self.epsilon = [0.01, 0.001, 1e-5]

        self.final_epsilon = final_epsilon

        self.current_epsilon = initial_epsilon
        print("Initial epsilon: ", self.current_epsilon)

        # Init model space
        self.model_space = ModelSpace(model_list)

        self.out_dir = out_dir
        self.population_number = 0
        self.finished = False

    def save_object_pickle(self, output_dir):
        pickle_name = output_dir + "checkpoint.pickle"
        with open(pickle_name , 'wb') as handle:
            pickle.dump(self, handle, protocol=-1)

    def plot_accepted_particles(self, out_dir, pop_num, batch_num, part_judgements, init_states, model_refs):
        out_path = out_dir + "Population_" + str(pop_num) + "_batch_" + str(batch_num) + "_accepted_plots.pdf"

        if sum(part_judgements) == 0:
            return 0

        # Make new pdf
        with PdfPages(out_path) as pdf:
            # Iterate all particles in batch
            for sim_idx, is_accepted in enumerate(part_judgements):
                if is_accepted:
                    state_list = self.pop_obj.get_particle_state_list(sim_idx)
                    time_points = self.pop_obj.get_timepoints_list()

                else:
                    continue

                try:
                    state_list = np.reshape(state_list, (len(time_points), len(init_states[sim_idx])))

                except(ValueError):
                    # print(len(state_list)/len(init_states[sim_idx]))
                    time_points = range(int(len(state_list) / len(init_states[sim_idx])))
                    state_list = np.reshape(state_list, (len(time_points), len(init_states[sim_idx])))

                model_ref = model_refs[sim_idx]

                plot_species = self.fit_species
                error_msg = self.pop_obj.get_particle_integration_error(sim_idx)

                plotting.plot_simulation(pdf, sim_idx, model_ref, state_list, time_points, plot_species, error_msg)

        # pdf.close()

    def plot_all_particles(self, out_dir, pop_num, batch_num, init_states, model_refs):
        out_path = out_dir + "/simulation_plots/Population_" + str(pop_num) + "_batch_" + str(
            batch_num) + "all_plots.pdf"

        # Make new pdf
        pdf = PdfPages(out_path)

        negative_count = 0
        # Iterate all particles in batch
        for sim_idx, m_ref in enumerate(model_refs):
            state_list = self.pop_obj.get_particle_state_list(sim_idx)
            time_points = self.pop_obj.get_timepoints_list()

            try:
                state_list = np.reshape(state_list, (len(time_points), len(init_states[sim_idx])))

            except(ValueError):
                # print(len(state_list)/len(init_states[sim_idx]))
                time_points = range(int(len(state_list) / len(init_states[sim_idx])))
                state_list = np.reshape(state_list, (len(time_points), len(init_states[sim_idx])))

            if np.min(state_list) < 0 or np.isnan(state_list).any():
                negative_count += 1

            model_ref = model_refs[sim_idx]
            error_msg = self.pop_obj.get_particle_integration_error(sim_idx)

            plot_species = [i for i in self.fit_species]
            # plot_species = [i for i in range(np.shape(state_list)[1])]

            plotting.plot_simulation(pdf, sim_idx, model_ref, state_list, time_points, plot_species, error_msg)

        print("negative count: ", negative_count)
        pdf.close()

    def write_all_particle_state_lists(self, out_dir, pop_num, batch_num, init_states, model_refs):
        for sim_idx, m_ref in enumerate(model_refs):
            out_path = out_dir + "/simulation_states/Population_" + str(pop_num) + "_batch_" + \
                       str(batch_num) + "_idx_" + str(sim_idx) + "_state.csv"

            state_list = self.pop_obj.get_particle_state_list(sim_idx)
            time_points = self.pop_obj.get_timepoints_list()

            tp = int(len(state_list) / len(init_states[sim_idx]))
            state_list = np.reshape(state_list, (tp, len(init_states[sim_idx])))
            np.savetxt(out_path, state_list, delimiter=',')

    def write_accepted_particle_distances(self, out_dir, model_refs, part_judgments, distances):
        out_path = out_dir + "distances.csv"
        with open(out_path, 'a') as out_csv:
            wr = csv.writer(out_csv)
            for idx, is_accepted in enumerate(part_judgments):
                record_vals = [idx, model_refs[idx]]
                if is_accepted:
                    for n in self.fit_species:
                        for d in distances[idx][n]:
                            record_vals.append(d)

                    wr.writerow(record_vals)

    def write_particle_distances(self, out_dir, model_refs, batch_num, population_num, judgement_array, distances, only_accepted=False):
        out_path = out_dir + "distances.csv"

        # If file doesn't exist, write header
        if not os.path.isfile(out_path):
            col_header = ['sim_idx', 'batch_idx', 'population_num', 'exp_num', 'model_ref', 'Accepted', 'integ_error']
            idx = 1
            for n in self.fit_species:
                for d in self.current_epsilon:
                    col_header.append('d' + str(idx))
                    idx += 1

            with open(out_path, 'a') as out_csv:
                wr = csv.writer(out_csv, quoting=csv.QUOTE_NONNUMERIC)
                wr.writerow(col_header)

        # Write distances
        with open(out_path, 'a') as out_csv:
            wr = csv.writer(out_csv, quoting=csv.QUOTE_NONNUMERIC)

            for idx, m_ref in enumerate(model_refs):
                if only_accepted and judgement_array[idx] == False:
                    continue

                error_msg = self.pop_obj.get_particle_integration_error(idx)
                row_vals = [idx, batch_num, population_num, self.exp_num, m_ref, judgement_array[idx], error_msg]


                for n_idx, n in enumerate(self.fit_species):
                    for d in distances[idx][n_idx]:
                        row_vals.append(d)

                wr.writerow(row_vals)

    def write_particle_params(self, out_dir, batch_num, simulated_particles,
                              input_params, input_init_species, judgement_array, particle_weights):
        for m in self.model_space._model_list:
            out_path = out_dir + "model_" + str(m.get_model_ref()) + "_all_params"

            # Add header if file does not yet exist
            if not os.path.isfile(out_path):
                col_header = ['sim_idx', 'batch_idx', 'Accepted', 'particle_weight'] + list(sorted(m._params_prior, key=str.lower)) + list(
                    m._init_species_prior)
                with open(out_path, 'a') as out_csv:
                    wr = csv.writer(out_csv)
                    wr.writerow(col_header)

            # Write data
            with open(out_path, 'a') as out_csv:
                wr = csv.writer(out_csv)
                for idx, particle in enumerate(simulated_particles):
                    if m is particle:
                        wr.writerow(
                            [idx] + [batch_num] + [judgement_array[idx]] + [particle_weights[idx]] + input_params[idx] + input_init_species[idx])

    def write_population_particle_params(self, out_dir):
        for m in self.model_space._model_list:
            out_path = out_dir + "model_" + str(m.get_model_ref()) + "_population_all_params"

            # Add header if file does not yet exist
            if not os.path.isfile(out_path):
                col_header = ['sim_idx', 'batch_idx', 'population_num', 'exp_num', 'particle_weight'] + list(sorted(m._params_prior, key=str.lower)) + list(
                    m._init_species_prior)
                with open(out_path, 'a') as out_csv:
                    wr = csv.writer(out_csv)
                    wr.writerow(col_header)

            # Write data
            with open(out_path, 'a') as out_csv:
                wr = csv.writer(out_csv)
                for idx, particle in enumerate(self.population_accepted_particles):
                    if m._model_ref is particle.curr_model._model_ref:
                        wr.writerow(
                            [particle.sim_idx] + [particle.batch_idx] + [self.population_number] + [self.exp_num] + [particle.curr_weight] + particle.curr_params + particle.curr_init_state)


    def write_particle_sum_stdevs(self, out_dir, model_refs, batch_num, simulated_particles, from_timepoint):
        out_path = out_dir + "sum_stdev.csv"

        if not os.path.isfile(out_path):
            col_header = ['sim_idx', 'batch_num', 'model_ref', 'sum_stdev']

            with open(out_path, 'a') as out_csv:
                wr = csv.writer(out_csv)
                wr.writerow(col_header)

        with open(out_path, 'a') as out_csv:
            wr = csv.writer(out_csv)

            for sim_idx, m_ref in enumerate(model_refs):
                integ_error = self.pop_obj.get_particle_integration_error(sim_idx)

                if integ_error == 'species_decayed' or integ_error == 'no_progress_error' or integ_error == 'step_adjustment_error':
                    sum_stdev = np.nan

                else:
                    sum_stdev = self.pop_obj.get_particle_sum_stdev(sim_idx, from_timepoint)

                row_vals = [sim_idx, batch_num, m_ref, sum_stdev]

                wr.writerow(row_vals)

    def write_particle_sum_grads(self, out_dir, model_refs, batch_num, simulated_particles, from_timepoint):
        out_path = out_dir + "sum_grad.csv"

        if not os.path.isfile(out_path):
            col_header = ['sim_idx', 'batch_num', 'model_ref', 'sum_grad']

            with open(out_path, 'a') as out_csv:
                wr = csv.writer(out_csv)
                wr.writerow(col_header)

        with open(out_path, 'a') as out_csv:
            wr = csv.writer(out_csv)

            for sim_idx, m_ref in enumerate(model_refs):
                integ_error = self.pop_obj.get_particle_integration_error(sim_idx)

                if integ_error == 'species_decayed' or integ_error == 'no_progress_error' or integ_error == 'step_adjustment_error':
                    sum_grad = np.nan

                else:
                    sum_grad = self.pop_obj.get_particle_sum_grad(sim_idx)

                row_vals = [sim_idx, batch_num, m_ref, sum_grad]

                wr.writerow(row_vals)

    def write_particle_grads(self, out_dir, model_refs, batch_num, simulated_particles, from_timepoint):
        out_path = out_dir + "all_grads.csv"
        if not os.path.isfile(out_path):
            col_header = ['sim_idx', 'batch_num', 'model_ref', 'grad_N1', 'grad_N2', 'grad_S', 'grad_x_4', 'grad_x_5',
                          'grad_x_6', 'grad_x_7']

            with open(out_path, 'a') as out_csv:
                wr = csv.writer(out_csv)
                wr.writerow(col_header)

        with open(out_path, 'a') as out_csv:
            wr = csv.writer(out_csv)

            for sim_idx, m_ref in enumerate(model_refs):
                integ_error = self.pop_obj.get_particle_integration_error(sim_idx)

                if integ_error == 'species_decayed' or integ_error == 'no_progress_error' or integ_error == 'step_adjustment_error':
                    all_grads = []

                else:
                    all_grads = self.pop_obj.get_particle_grads(sim_idx)

                row_vals = [sim_idx, batch_num, m_ref] + all_grads

                wr.writerow(row_vals)

    def write_epsilon(self, out_dir, epsilon):
        out_path = out_dir + "epsilon.txt"

        if not os.path.isfile(out_path):
            col_header = ['e_' + str(idx) for idx, _ in enumerate(epsilon)]

            with open(out_path, 'a') as out_csv:
                wr = csv.writer(out_csv)
                wr.writerow(col_header)
                wr.writerow(epsilon)

    def write_eigenvalues(self, out_dir, model_refs, batch_num, simulated_particles,
                          end_state=False, init_state=False, do_fsolve=False):

        if end_state == True:
            out_path = out_dir + "eigenvalues_end_state.csv"

        elif init_state == True:
            out_path = out_dir + "eigenvalues_init_state.csv"

        elif do_fsolve == True:
            out_path = out_dir + "eigenvalues_do_fsolve_state.csv"

        else:
            print("State to use for jacobian not specified, exiting...")
            exit()

        # If file doesn't exist, write header
        if not os.path.isfile(out_path):
            col_header = ['sim_idx', 'batch_num', 'model_ref', 'integ_error', 'fsolve_error']
            for i in range(10):
                str_eig_real = 'eig_#I#_real'.replace('#I#', str(i))
                str_eig_imag = 'eig_#I#_imag'.replace('#I#', str(i))
                col_header = col_header + [str_eig_real] + [str_eig_imag]

            with open(out_path, 'a') as out_csv:
                wr = csv.writer(out_csv)
                wr.writerow(col_header)

        with open(out_path, 'a') as out_csv:
            wr = csv.writer(out_csv)

            for sim_idx, m_ref in enumerate(model_refs):
                fsolve_error = 0

                n_species = len(simulated_particles[sim_idx]._init_species_prior)

                jac = []
                if end_state == True:
                    jac = self.pop_obj.get_particle_end_state_jacobian(sim_idx)

                elif init_state == True:
                    jac = self.pop_obj.get_particle_init_state_jacobian(sim_idx)

                elif do_fsolve == True:
                    final_state = self.pop_obj.get_particle_final_species_values(sim_idx)
                    res = fsolve(alg_utils.fsolve_conversion, final_state, fprime=alg_utils.fsolve_jac_conversion,
                                 args=(self.pop_obj, sim_idx, n_species), full_output=True)
                    steady_state = res[0]

                    ier = res[2]
                    fsolve_error = ier

                    steady_state = steady_state.tolist()

                    jac = self.pop_obj.get_particle_jacobian(steady_state, sim_idx)

                jac = np.reshape(jac, (n_species, n_species))

                try:
                    eigenvalues = np.linalg.eigvals(jac)

                except(np.linalg.LinAlgError) as e:
                    eigenvalues = [np.nan for i in range(n_species)]

                eigenvalues = [[i.real, i.imag] for i in eigenvalues]

                real_parts = [i[0] for i in eigenvalues]
                imag_parts = [i[1] for i in eigenvalues]

                integ_error = self.pop_obj.get_particle_integration_error(sim_idx)
                row_vals = [sim_idx, batch_num, m_ref, integ_error, fsolve_error]

                for idx_e, i in enumerate(eigenvalues):
                    row_vals = row_vals + [real_parts[idx_e]] + [imag_parts[idx_e]]

                wr.writerow(row_vals)

    def write_time_to_stability(self, out_dir, model_refs, batch_num, simulated_particles):

        out_path = out_dir + "time_to_stab.csv"

        # If file doesn't exist, write header
        if not os.path.isfile(out_path):
            col_header = ['sim_idx', 'batch_num', 'model_ref', 'time_to_stab']

            with open(out_path, 'a') as out_csv:
                wr = csv.writer(out_csv)
                wr.writerow(col_header)

        time_points = self.pop_obj.get_timepoints_list()

        with open(out_path, 'a') as out_csv:
            wr = csv.writer(out_csv)

            for sim_idx, m_ref in enumerate(model_refs):
                integ_error = self.pop_obj.get_particle_integration_error(sim_idx)

                time_to_stab = 0

                if integ_error == 'species_decayed' or integ_error == 'no_progress_error' or integ_error == 'step_adjustment_error':
                    time_to_stab = np.nan

                else:

                    final_state = self.pop_obj.get_particle_final_species_values(sim_idx)
                    state_list = self.pop_obj.get_particle_state_list(sim_idx)

                    n_species = len(simulated_particles[sim_idx]._init_species_prior)

                    state_list = np.reshape(state_list, (len(time_points), n_species))

                    for t_idx, t in enumerate(time_points):
                        lists_equal = list(state_list[t_idx]) == list(final_state)

                        if lists_equal:
                            time_to_stab = t
                            break

                row_vals = [sim_idx, batch_num, m_ref, time_to_stab]
                wr.writerow(row_vals)

    

    def run_model_selection_ABC_SMC(self, alpha=0.5, run_test=0):
        
        # abs_tol = 1e-20
        # rel_tol = 1e-12

        # abs_tol = 1e-15
        # rel_tol = 1e-5

        while not self.finished:
            folder_name = self.out_dir + "Population_" + str(self.population_number) + "/"

            try:
                os.mkdir(folder_name)
            except FileExistsError:
                pass

            sim_params_folder = folder_name + 'model_sim_params/'
            try:
                os.mkdir(sim_params_folder)
            except FileExistsError:
                pass

            try:
                os.mkdir(folder_name + 'simulation_plots')
            except FileExistsError:
                pass

            try:
                os.mkdir(folder_name + 'simulation_states')

            except FileExistsError:
                pass

            if self.current_epsilon == self.final_epsilon:
                print("Running final epsilon")
                self.finished = True

            # Reset for new population
            total_sims = 0
            self.batch_num = 0


            while self.population_accepted_count < self.population_size:
                print("")
                start_time = time.time()
                start_time_sampling = time.time()

                # 1. Sample from model space
                if self.population_number == 0:
                    particles = self.model_space.sample_particles_from_prior(self.n_sims_batch)
                    init_weights = 1/len(particles)
                    for sim_idx, p in enumerate(particles):
                        p.curr_weight = init_weights
                        p.prev_weight = init_weights
                        p.batch_idx = self.batch_num
                        p.sim_idx = sim_idx


                else:
                    print("Sampling new population")
                    # Sample and perturb particles from previous population
                    particles = self.model_space.sample_particles_from_previous_population(self.n_sims_batch)
                    for sim_idx, p in enumerate(particles):
                        p.batch_idx = self.batch_num
                        p.sim_idx = sim_idx

                end_time_sampling = time.time()
                print("Particle sampling time elapsed: ", end_time_sampling - start_time_sampling)

                init_states = [copy.deepcopy(p.curr_init_state) for p in particles]
                input_params = [copy.deepcopy(p.curr_params) for p in particles]
                model_refs = [copy.deepcopy(p.curr_model._model_ref) for p in particles]
                particle_weights = [p.prev_weight for p in particles]

                particle_models = [p.curr_model for p in particles]

                alg_utils.rescale_parameters(input_params, init_states, particle_models)


                if run_test==0:
                    self.pop_obj = population_modules.Population(self.n_sims_batch, self.t_0, self.t_end,
                                                                 self.dt, init_states, input_params, model_refs,
                                                                 self.fit_species, self.abs_tol, self.rel_tol)
            
                    # print("gen particles")
                    self.pop_obj.generate_particles()
                    # print("sim particles")
                    start_time_sim = time.time()
                    

                    self.pop_obj.simulate_particles()
                    end_time_sim = time.time()

                    print("Particle simulation time elapsed: ", end_time_sim - start_time_sim)

            
                    # 3. Calculate distances for population
                    # print("calculating distances")
                    self.pop_obj.calculate_particle_distances(self.distance_function_mode)
                    # print("got distances")

                    self.pop_obj.accumulate_distances()
                    # print("distances accumulated")

                    batch_distances = self.pop_obj.get_flattened_distances_list()
                    batch_distances = np.reshape(batch_distances,
                                                 (self.n_sims_batch, len(self.fit_species), self.n_distances))

                    # 4. Accept or reject particles
                    if self.distance_function_mode != 1:
                        batch_part_judgements = alg_utils.check_distances_generic(batch_distances,
                                                                                 epsilon_array=self.current_epsilon)
                    elif self.distance_function_mode == 1:
                        batch_part_judgements = alg_utils.check_distances_osc(batch_distances,
                                                                              epsilon_array=self.current_epsilon)

                    else:
                        batch_part_judgements = None
                        print("Invalid distance function set, quitting... ")
                        exit()


                if run_test == 1:
                    print("Running test ")
                    batch_part_judgements = []
                    batch_distances = []
                    for p in particles:
                        d1 = np.random.uniform(0, 1)

                        batch_distances.append(d1)
                        if d1 < self.current_epsilon[0]:
                            batch_part_judgements.append(True)

                        else:
                            batch_part_judgements.append(False)


                    print(np.shape(batch_distances))
                    batch_distances = np.reshape(batch_distances, 
                        (self.n_sims_batch, 1, 1))

                if run_test == 2:
                    if self.population_number == 0:
                        self.current_epsilon = [1e900, 1e900]

                    # gen_data = 
                    n = 100
                    print("Running test ")
                    batch_part_judgements = []
                    batch_distances = []
                    factorial_all = lambda x: np.math.factorial(x)
                    v_factorial_all = np.vectorize(factorial_all)

                    for p in particles:
                        s1 = None
                        t1 = None

                        # Poission model
                        if p.curr_model._model_ref == 0:
                            if self.population_number == 0:
                                p.curr_params[0] = np.random.exponential(1)
                            
                            M1_sim = np.random.poisson(p.curr_params[0], n)

                            s1 = np.sum(M1_sim)

                            factorials = [math.factorial(x) for x in M1_sim]
                            log_factorials = [math.log(x) for x in factorials]
                            t1 = np.sum(log_factorials)

                        # Geometric model
                        if p.curr_model._model_ref == 1:
                            M2_sim = np.random.geometric(p.curr_params[0], n)
                            s1 = np.sum(M2_sim)

                            factorials = [math.factorial(x)  for x in M2_sim]
                            log_factorials = [math.log(x)for x in factorials]
                            t1 = np.sum(log_factorials)

                        sim_dists = [abs(target_data_s1 - s1), abs(target_data_t1 - t1)]
                        batch_distances.append(sim_dists)


                        if sim_dists[0] < self.current_epsilon[0] and sim_dists[1] < self.current_epsilon[1]:
                            batch_part_judgements.append(True)
                        
                        else:
                            batch_part_judgements.append(False)

                    batch_distances = np.reshape(batch_distances, 
                    (self.n_sims_batch, len(self.fit_species), self.n_distances))

                if run_test == 3:
                    for n in range(self.n_sims_batch):
                        ode_sim.run_sim(self.t_0, self.t_end, self.dt, init_states[n], input_params[n])



                self.population_accepted_particle_distances += [part_d for part_d, judgement in
                                                zip(batch_distances, batch_part_judgements)
                                                if judgement]

                accepted_particles = [p for p, judgement in zip(particles, batch_part_judgements) if judgement]


                self.population_accepted_particles = self.population_accepted_particles + accepted_particles

                # print("Writing data")
                self.write_epsilon(folder_name, self.current_epsilon)

                # Write data

                if run_test == 0:
                    start_time_write_distance = time.time()

                    if self.final_epsilon == self.current_epsilon:
                        self.write_particle_distances(folder_name, model_refs, self.batch_num, self.population_number,
                            batch_part_judgements, batch_distances, only_accepted=True)

                    end_time_write_distance = time.time()

                print("Write distance time elapsed: ", end_time_write_distance - start_time_write_distance)

                # print("Doing the rest")

                self.population_accepted_count += sum(batch_part_judgements)
                self.population_total_simulations += len(model_refs)

                self.model_space.update_population_sample_data_v2(model_refs, batch_part_judgements)

                print("Population: ", self.population_number, "Accepted particles: ", self.population_accepted_count,
                      "Total simulations: ", self.population_total_simulations)

                if not run_test:
                    del self.pop_obj

                sys.stdout.flush()
                self.batch_num += 1
                end_time = time.time()

                print("batch time elapsed: ", end_time - start_time)


            self.population_accepted_particles = self.population_accepted_particles[: self.population_size]
            # self.population_model_refs = self.population_model_refs[: self.population_size]
            # self.population_judgements = self.population_judgements[: self.population_size]

            self.model_space.accepted_particles = self.population_accepted_particles
            # self.model_space.update_population_sample_data(self.population_model_refs, self.population_judgements)

            if self.population_number == 0:
                for p in self.model_space.accepted_particles:
                    p.curr_weight = 1

            else:
                self.model_space.compute_particle_weights()

            self.model_space.normalize_particle_weights()
            self.model_space.update_model_marginals()

            if self.final_epsilon == self.current_epsilon:
                self.save_object_pickle(folder_name)
                self.write_population_particle_params(sim_params_folder)

            self.model_space.prepare_next_population()

            print("generating model kernels")
            self.model_space.generate_model_kernels(self.population_accepted_particles, self.population_number)
            print("generating aux info")

            self.model_space.generate_kernel_aux_info()
            self.model_space.count_dead_models()

            print("generating model space report")
            print(folder_name)

            self.model_space.model_space_report(folder_name, self.batch_num, use_sum=False)

            if self.current_epsilon != self.final_epsilon: 
                self.current_epsilon = alg_utils.update_epsilon(self.current_epsilon, self.final_epsilon,
                                                           self.population_accepted_particle_distances, alpha)

                print("Current epsilon: ", self.current_epsilon)
                print("Starting new population... ")
                print("")

                self.population_number += 1
                self.population_accepted_count = 0
                self.population_total_simulations = 0
                self.population_accepted_particle_distances = []
                self.population_model_refs = []
                self.population_judgements = []
                self.population_accepted_particles = []
