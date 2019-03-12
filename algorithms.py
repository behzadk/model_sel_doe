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
from scipy.optimize import fsolve


def blockPrinting(func):
    def func_wrapper(*args, **kwargs):
        # block all printing to the console
        sys.stdout = open(os.devnull, 'w')
        # call the method in question
        func(*args, **kwargs)
        # enable all printing to the console
        sys.stdout = sys.__stdout__

    return func_wrapper


class Rejection:
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

        self.epsilon = [100, 10, 1e4]

        # Init model space
        self.model_space = ModelSpace(model_list)

        self.out_dir = out_dir

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


    def plot_all_particles(self, out_dir, pop_num, batch_num, init_states, model_refs):
        out_path = out_dir + "/simulation_plots/Population_" + str(pop_num) + "_batch_" + str(batch_num) + "_plots.pdf"

        # Make new pdf
        pdf = PdfPages(out_path)

        # Iterate all particles in batch
        for sim_idx, m_ref in enumerate(model_refs):
            state_list = self.pop_obj.get_particle_state_list(sim_idx)
            time_points = self.pop_obj.get_timepoints_list()

            try:
                state_list = np.reshape(state_list, (len(time_points), len(init_states[sim_idx])))

            except(ValueError):
                continue

            model_ref = model_refs[sim_idx]

            plot_species = [i for i in range(self.n_species_fit)]
            plotting.plot_simulation(pdf, sim_idx, model_ref, state_list, time_points, plot_species)
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
                    for n in range(self.n_species_fit):
                        for d in distances[idx][n]:
                            record_vals.append(d)

                    wr.writerow(record_vals)

    def write_particle_distances(self, out_dir, model_refs, batch_num, simulated_particles, judgement_array, distances):
        out_path = out_dir + "distances.csv"

        # If file doesn't exist, write header
        if not os.path.isfile(out_path):
            col_header = ['sim_idx', 'batch_num', 'model_ref', 'Accepted']
            idx = 1
            for n in range(self.n_species_fit):
                for d in self.epsilon:
                    col_header.append('d'+str(idx))
                    idx +=1

            with open(out_path, 'a') as out_csv:
                wr = csv.writer(out_csv, quoting=csv.QUOTE_NONNUMERIC)
                wr.writerow(col_header)

        # Write distances
        with open(out_path, 'a') as out_csv:
            wr = csv.writer(out_csv, quoting=csv.QUOTE_NONNUMERIC)

            for idx, m_ref in enumerate(model_refs):
                row_vals = [idx, batch_num, m_ref, judgement_array[idx]]

                for n in range(self.n_species_fit):
                    for d in distances[idx][n]:
                        row_vals.append(d)

                wr.writerow(row_vals)

    def write_particle_params(self, out_dir, batch_num, simulated_particles,
                                  input_params, input_init_species,  judgement_array):
        for m in self.model_space._model_list:
            out_path = out_dir + "model_" + str(m.get_model_ref()) + "_all_params"

            # Add header if file does not yet exist
            if not os.path.isfile(out_path):
                col_header = ['sim_idx', 'batch_num', 'Accepted'] + list(m._params_prior) + list(m._init_species_prior)
                with open(out_path, 'a') as out_csv:
                    wr = csv.writer(out_csv)
                    wr.writerow(col_header)

            # Write data
            with open(out_path, 'a') as out_csv:
                wr = csv.writer(out_csv)
                for idx, particle in enumerate(simulated_particles):
                    if m is particle:
                        wr.writerow([idx] + [batch_num] + [judgement_array[idx]] + input_params[idx] + input_init_species[idx])

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
            col_header = ['sim_idx', 'batch_num', 'model_ref', 'grad_N1', 'grad_N2', 'grad_S', 'grad_x_4', 'grad_x_5', 'grad_x_6', 'grad_x_7']

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




    def run_rejection(self):
        population_number = 0

        folder_name = self.out_dir + "Population_" + str(population_number) + "/"

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

        accepted_particles_count = 0
        total_sims = 0
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

            # 3. Calculate distances for population
            print("calculating distances")
            self.pop_obj.calculate_particle_distances()
            print("got distances")

            self.pop_obj.accumulate_distances()
            batch_distances = self.pop_obj.get_flattened_distances_list()
            batch_distances = np.reshape(batch_distances, (self.n_sims_batch, self.n_species_fit, self.n_distances))

            # 4. Accept or reject particles
            batch_part_judgements = alg_utils.check_distances(batch_distances, epsilon_array=self.epsilon)


            # Write data
            self.write_particle_params(sim_params_folder, batch_num, particle_models.tolist(),
                                                   input_params, init_states, batch_part_judgements)

            self.write_particle_distances(folder_name, model_refs, batch_num, particle_models.tolist(),
                                          batch_part_judgements, batch_distances)

            self.write_eigenvalues(folder_name, model_refs, batch_num, particle_models.tolist(), do_fsolve=True)
            # self.write_all_particle_state_lists(folder_name, population_number, batch_num, init_states, model_refs)

            accepted_particles_count += sum(batch_part_judgements)
            total_sims += len(model_refs)

            print("Population: ", population_number, "Accepted particles: ", accepted_particles_count, "Total simulations: ", total_sims)
            self.model_space.update_model_population_sample_data(particle_models.tolist(), batch_part_judgements)
            self.model_space.model_space_report(folder_name, batch_num)

            sys.stdout.flush()
            batch_num += 1

    def find_eigensystems(self):
        population_number = 0

        folder_name = self.out_dir + "Population_" + str(population_number) + "/"

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

        accepted_particles_count = 0
        total_sims = 0
        batch_num = 0

        plot_states = []
        while accepted_particles_count < self.population_size:
            # 1. Sample from model space
            particle_models = self.model_space.sample_model_space(self.n_sims_batch)  # Model objects in this simulation

            # 2. Sample particles for each model
            init_states, input_params, model_refs = alg_utils.generate_particles(particle_models)          # Extract input parameters and model references

            # 3. Simulate population
            self.pop_obj = population_modules.Population(self.n_sims_batch, self.t_0, self.t_end,
                                              self.dt, init_states, input_params, model_refs)
            self.pop_obj.generate_particles()

            # Iterate init states, accepting all negative systems
            acc_init_states = []
            acc_input_params = []
            acc_model_refs = []
            acc_particle_models = []
            for sim_idx, m_ref in enumerate(model_refs):
                n_species = len(init_states[sim_idx])

                root = fsolve(alg_utils.fsolve_conversion, init_states[sim_idx], fprime=alg_utils.fsolve_jac_conversion,
                             args=(self.pop_obj, sim_idx, n_species), full_output=True)

                steady_state = root[0]

                ier = root[2]
                fsolve_error = ier

                steady_state = steady_state.tolist()
                steady_state = root[0]

                ier = root[2]
                fsolve_error = ier

                steady_state = steady_state.tolist()

                if not all(i > 0 for i in steady_state):
                    continue

                jac = self.pop_obj.get_particle_jacobian(steady_state, sim_idx)

                jac = np.reshape(jac, (n_species, n_species))

                try:
                    eigenvalues = np.linalg.eigvals(jac)

                except(np.linalg.LinAlgError) as e:
                    eigenvalues = [np.nan for i in range(n_species)]

                eigenvalues = [[i.real, i.imag] for i in eigenvalues]

                real_parts = [i[0] for i in eigenvalues]
                imag_parts = [i[1] for i in eigenvalues]

                all_neg_real_parts = all(i < 0 for i in real_parts)
                only_real_parts = all(i == 0.0 for i in imag_parts)

                if all_neg_real_parts and only_real_parts:
                    plot_states.append(steady_state)
                    acc_input_params.append(input_params[sim_idx])
                    acc_model_refs.append(model_refs[sim_idx])
                    acc_particle_models.append(particle_models.tolist()[sim_idx])


            total_sims += 1
            print(total_sims)
            if total_sims > 1:
                break

        plot_states = np.array(plot_states)
        print(np.shape(plot_states))
        plt.scatter(plot_states[:, 0], plot_states[:, 1], s=0.8)

        plt.xscale('log')
        plt.yscale('log')
        plt.show()

    def run_paramter_optimisation(self, parameters_to_optimise):
        population_number = 0

        folder_name = self.out_dir + "Population_" + str(population_number) + "/"

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

        accepted_particles_count = 0
        total_sims = 0
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

            # 3. Calculate distances for population
            print("calculating distances")
            self.pop_obj.calculate_particle_distances()
            print("got distances")

            self.pop_obj.accumulate_distances()
            batch_distances = self.pop_obj.get_flattened_distances_list()
            batch_distances = np.reshape(batch_distances, (self.n_sims_batch, self.n_species_fit, self.n_distances))

            # 4. Accept or reject particles
            batch_part_judgements = alg_utils.check_distances(batch_distances, epsilon_array=self.epsilon)
            num_true = sum(batch_part_judgements)
            total = len(batch_part_judgements)
            print(num_true / total)

            self.model_space.alt_generate_model_param_kdes(batch_part_judgements, particle_models.tolist(), input_params, parameters_to_optimise)
            self.model_space.alt_generate_model_init_species_kdes(batch_part_judgements, particle_models.tolist(), init_states, parameters_to_optimise)

            self.wr
            # # Write data
            # self.write_particle_params(sim_params_folder, batch_num, particle_models.tolist(),
            #                                        input_params, init_states, batch_part_judgements)
            #
            # self.write_particle_distances(folder_name, model_refs, batch_num, particle_models.tolist(),
            #                               batch_part_judgements, batch_distances)
            #
            # self.write_eigenvalues(folder_name, model_refs, batch_num, particle_models.tolist(), do_fsolve=True)
            #
            # accepted_particles_count += sum(batch_part_judgements)
            # total_sims += len(model_refs)
            #
            # print("Population: ", population_number, "Accepted particles: ", accepted_particles_count, "Total simulations: ", total_sims)
            # self.model_space.update_model_population_sample_data(particle_models.tolist(), batch_part_judgements)
            # self.model_space.model_space_report(folder_name, batch_num)


    def run_random_jacobian(self):
        population_number = 0

        folder_name = self.out_dir + "Population_" + str(population_number) + "/"

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

        accepted_particles_count = 0
        total_sims = 0
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
            self.plot_all_particles(folder_name, 0, batch_num, init_states, model_refs)

            # Iterate init states, accepting all negative systems
            acc_init_states = []
            acc_input_params = []
            acc_model_refs = []
            acc_particle_models = []
            for sim_idx, m_ref in enumerate(model_refs):
                n_species = len(init_states[sim_idx])

                root = fsolve(alg_utils.fsolve_conversion, init_states[sim_idx], fprime=alg_utils.fsolve_jac_conversion,
                             args=(self.pop_obj, sim_idx, n_species), full_output=True)

                steady_state = root[0]

                ier = root[2]
                fsolve_error = ier

                steady_state = steady_state.tolist()

                if not all(i > 0 for i in steady_state):
                    continue

                jac = self.pop_obj.get_particle_jacobian(steady_state, sim_idx)

                jac = np.reshape(jac, (n_species, n_species))

                try:
                    eigenvalues = np.linalg.eigvals(jac)

                except(np.linalg.LinAlgError) as e:
                    eigenvalues = [np.nan for i in range(n_species)]

                eigenvalues = [[i.real, i.imag] for i in eigenvalues]

                real_parts = [i[0] for i in eigenvalues]
                imag_parts = [i[1] for i in eigenvalues]

                all_neg_real_parts = all(i < 0 for i in real_parts)
                only_real_parts = all(i == 0.0 for i in imag_parts)

                if all_neg_real_parts and only_real_parts:
                    acc_init_states.append(steady_state)
                    acc_input_params.append(input_params[sim_idx])
                    acc_model_refs.append(model_refs[sim_idx])
                    acc_particle_models.append(particle_models.tolist()[sim_idx])

            if len(acc_init_states) == 0:
                continue


            n_sims_batch = len(acc_init_states)

            # Simulate all systems that pass the eigenstability test
            succ_init_states = []
            succ_input_params = []
            succ_model_refs = []
            succ_particle_models = []

            self.pop_obj = population_modules.Population(n_sims_batch, 0, 100,
                                              self.dt, acc_init_states, acc_input_params, acc_model_refs)
            self.pop_obj.generate_particles()
            self.pop_obj.simulate_particles()

            for sim_idx, m_ref in enumerate(acc_model_refs):
                n_species = len(init_states[sim_idx])
                final_state = self.pop_obj.get_particle_final_species_values(sim_idx)

                if not all(i > 0 for i in final_state):
                    continue

                jac = self.pop_obj.get_particle_jacobian(final_state, sim_idx)

                jac = np.reshape(jac, (n_species, n_species))

                try:
                    eigenvalues = np.linalg.eigvals(jac)

                except(np.linalg.LinAlgError) as e:
                    eigenvalues = [np.nan for i in range(n_species)]

                eigenvalues = [[i.real, i.imag] for i in eigenvalues]

                real_parts = [i[0] for i in eigenvalues]
                imag_parts = [i[1] for i in eigenvalues]

                all_neg_real_parts = all(i < 0 for i in real_parts)
                only_real_parts = all(i == 0.0 for i in imag_parts)

            self.plot_all_particles(folder_name, 1, batch_num, acc_init_states, acc_model_refs)
            # self.write_eigenvalues(folder_name, model_refs, batch_num, particle_models.tolist(), end_state=True)
            # self.write_eigenvalues(folder_name, model_refs, batch_num, particle_models.tolist(), init_state=True)

            total_sims += len(model_refs)
            integ_errors = self.pop_obj.get_all_particle_integration_errors()

            print("Population: ", population_number, "Accepted particles: ", accepted_particles_count, "Total simulations: ", total_sims)
            # self.model_space.update_model_population_sample_data(acc_particle_models, batch_part_judgements)
            self.model_space.model_space_report(folder_name, batch_num)

            sys.stdout.flush()
            batch_num += 1


class ABC_SMC:
    pass