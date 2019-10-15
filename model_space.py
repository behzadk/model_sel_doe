import numpy as np
from scipy import stats
import pandas as pd
import algorithm_utils as alg_utils
import copy

log_decimals = 4

class Particle:
    def __init__(self, model):
        self.curr_model = model
        self.prev_model = None

        self.prev_weight = None
        self.prev_params = None
        self.prev_init_state = None

        self.curr_weight = 1
        self.curr_params = []
        self.curr_init_state = []

    def sample_params_from_prior(self):
        self.curr_params = self.curr_model.sample_particle()

    def sample_init_state_from_prior(self):
        self.curr_init_state = self.curr_model.sample_init_state()

    def perturb_particle(self):
        new_params = []
        new_init_state = []

        # Perturb particles
        for idx, id in enumerate(sorted(self.prev_model._params_prior, key=str.lower)):
            kernel = self.prev_model.param_kernels[idx]

            kern_lwr = kernel[0]
            kern_upr = kernel[1]

            previous_param = self.prev_params[idx]

            prior_min = self.prev_model._params_prior[id][0]
            prior_max = self.prev_model._params_prior[id][1]

            if prior_min == prior_max:
                new_params.append(previous_param)
                continue

            low_flag = (previous_param + kern_lwr) < prior_min
            high_flag = (previous_param + kern_upr) > prior_max

            if low_flag == True:
                kern_lwr = -(previous_param - prior_min)

            if high_flag == True:
                kern_upr = prior_max - previous_param

            if low_flag == False and high_flag == False:
                delta = np.random.uniform(kern_lwr, kern_upr)

            # Choose negative or positive perturbation
            else:
                pos_perturb_flag = np.random.uniform(0, 1) > abs(kern_lwr) / (abs(kern_lwr) + kern_upr)

                if pos_perturb_flag:
                    delta = np.random.uniform(0, kern_upr)

                else:
                    delta = np.random.uniform(kern_lwr, 0)

            perturbed_param = previous_param + delta
            new_params.append(perturbed_param)

        # Perturb init state
        for idx, id in enumerate(self.prev_model._init_species_prior):
            kernel = self.prev_model.init_state_kernels[idx]

            kern_lwr = kernel[0]
            kern_upr = kernel[1]

            previous_init_species = self.prev_init_state[idx]

            prior_min = self.prev_model._init_species_prior[id][0]
            prior_max = self.prev_model._init_species_prior[id][1]

            if prior_min == prior_max:
                new_init_state.append(previous_init_species)
                continue

            low_flag = (previous_init_species + kern_lwr) < prior_min
            high_flag = (previous_init_species + kern_upr) > prior_max

            if low_flag == True:
                kern_lwr = -(previous_init_species - self.prev_model._init_species_prior[id][0])

            if high_flag == True:
                kern_upr = self.prev_model._init_species_prior[id][1] - previous_init_species

            if low_flag == False and high_flag == False:
                delta = np.random.uniform(kern_lwr, kern_upr)

            # Choose negative or positive perturbation
            else:
                positive = np.random.uniform(0, 1) > abs(kern_lwr) / (abs(kern_lwr) + kern_upr)
                # print(kern_lwr)
                # print(kern_upr)
                # print("")

                if positive:
                    delta = np.random.uniform(0, kern_upr)

                else:
                    delta = np.random.uniform(kern_lwr, 0)

            perturbed_init_species = previous_init_species + delta
            new_init_state.append(perturbed_init_species)

        new_particle = copy.deepcopy(self)

        new_particle.curr_weight = None
        new_particle.curr_model = self.prev_model
        new_particle.curr_params = new_params
        new_particle.curr_init_state = new_init_state

        return new_particle




class Model:
    ##
    # Each model object possesses all the information needed to simulate the model.
    # Priors should be loaded from input files
    # model ref refers to the index of the model in the cpp module.
    ##
    def __init__(self, model_ref, params_prior, init_species_prior):
        self._model_ref = model_ref
        self._params_prior = params_prior
        self._init_species_prior = init_species_prior
        self._n_params = len(params_prior)

        self._param_kdes = [None for i in range(self._n_params)]
        self._init_state_kdes = [None for i in range(len(self._init_species_prior))]

        self._param_has_kde = [False for i in range(self._n_params)]
        self._init_species_has_kde = [False for i in range(len(self._init_species_prior))]

        self._prev_sample_probability = None
        self._current_sample_probability = 1

        # Model data. Number of times this model was sampled in each population and number of times it was accepted
        self.population_sample_count = []
        self.population_accepted_count = []

        self.param_kernels = []
        self.init_state_kernels = []

        self.curr_margin = 0
        self.prev_margin = 0


        for param_id in self._params_prior:
            param = self._params_prior[param_id]
            # if abs(np.log10(param[0]) - np.log10(param[1])) > log_decimals:
            if (param[0] < 1e-4 and param[0] != 0) or (param[0] > 1e4):
                self._params_prior[param_id].append('log')
                self._params_prior[param_id][0] = np.log(self._params_prior[param_id][0])
                self._params_prior[param_id][1] = np.log(self._params_prior[param_id][1])

            else:
                self._params_prior[param_id].append('uniform')

        for species_id in self._init_species_prior:
            species = self._init_species_prior[species_id]

            # if abs(np.log10(species[0]) - np.log10(species[1])) > log_decimals:
            if (species[0] < 1e-4 and species[0] != 0) or (species[0] > 1e4):
                self._init_species_prior[species_id].append('log')
                self._init_species_prior[species_id][0] = np.log(self._init_species_prior[species_id][0])
                self._init_species_prior[species_id][1] = np.log(self._init_species_prior[species_id][1])

            else:
                self._init_species_prior[species_id].append('uniform')



    def alt_generate_params_kde(self, param_values, fit_params_key):

        for idx, param_key in enumerate(list(self._params_prior.keys())):
            if param_key in fit_params_key:
                try:
                    kernel = stats.gaussian_kde(param_values)
                    self._param_kdes[idx] = (kernel)
                    self._param_has_kde[idx] = True

                except np.linalg.linalg.LinAlgError:
                    self._param_kdes[idx] = (param_values[0])
                    self._param_has_kde[idx] = False

            else:
                self._param_has_kde[idx] = False

        self._has_kde = True

    def alt_generate_species_kde(self, species_list, fit_params_id):
        species_list = np.asarray(species_list)
        for idx, species_key in enumerate(list(self._init_species_prior.keys())):
            if species_key in fit_params_id:
                param_vals = species_list[:, idx]
                try:
                    kernel = stats.gaussian_kde(param_vals)
                    self._init_state_kdes[idx] = (kernel)
                    self._init_species_has_kde[idx] = True

                except np.linalg.linalg.LinAlgError:
                    self._init_state_kdes[idx] = (param_vals[0])
                    # self._init_species_has_kde[idx] = False

            else:
                self._init_species_has_kde[idx] = False

    ##
    # Generates a KDE from a list of parameters, where each row is a complete set of parameters
    # required for a simulation.
    #
    # List is transformed into an ndarray, where each column is a single parameter. Each column is used to generate
    # a KDE
    ##
    def generate_kde(self, params_list):
        self._param_kdes = []
        params_list = np.asarray(params_list)
        for idx, param in enumerate(range(0, self._n_params)):
            param_vals = params_list[:, idx]
            try:
                kernel = stats.gaussian_kde(param_vals)
                self._param_kdes.append(kernel)

            except np.linalg.linalg.LinAlgError:
                self._param_kdes.append(param_vals[0])


    def generate_kernels(self, params, init_states):
        params = np.array(params)
        init_states = np.array(init_states)
        # params = np.array([np.array(p)for p in params])
        # init_states = np.array([np.array(s)for s in init_states])

        # params = np.array(params, dtype=object)
        # init_states = np.array(init_states, dtype=object)

        self.param_kernels = []
        self.init_state_kernels = []

        use_prior = False
        if use_prior:
            params_prior = self._params_prior
            init_species_prior = self._init_species_prior


            for idx, id in enumerate(sorted(self._params_prior, key=str.lower)):
                min_param = params_prior[id][0]
                max_param = params_prior[id][1]
                scale = max_param - min_param
                # print(id, scale)

                self.param_kernels.append([-scale / 2.0, scale / 2.0])

            for idx, s in enumerate(self._init_species_prior):
                min_init_state = init_species_prior[s][0]
                max_init_state = init_species_prior[s][1]
                
                scale = max_init_state - min_init_state

                self.init_state_kernels.append([-scale / 2.0, scale / 2.0])


        else:
            if np.shape(params)[0] == 1:
                for p_idx in range(np.shape(params)[1]):
                    self.param_kernels.append([-1, 1])

                for s_idx in range(np.shape(init_states)[1]):
                    self.init_state_kernels.append([-1, 1])

            else:
                for p_idx in range(np.shape(params)[1]):
                    min_param = min(params[:, p_idx])
                    max_param = max(params[:, p_idx])

                    scale = max_param - min_param

                    self.param_kernels.append([-scale / 2.0, scale / 2.0])

                for s_idx in range(np.shape(init_states)[1]):
                    min_init_state = min(init_states[:, s_idx])
                    max_init_state = max(init_states[:, s_idx])

                    scale = max_init_state - min_init_state

                    self.init_state_kernels.append([-scale / 2.0, scale / 2.0])

    ##
    # Samples simulation parameters.
    # If KDE has been generated, parameters are sampled from KDE
    # Else, parameters are sampled from the prior
    ##
    def sample_particle(self):
        sim_params = []

        for idx, id in enumerate(sorted(self._params_prior, key=str.lower)):

            # Check if parameter has a kde
            if self._param_has_kde[idx]:
                new_params = self._param_kdes[idx].resample(1)

                # Prevent sampling of negative parameter values
                while new_params < 0:
                    new_params = self._param_kdes[idx].resample(1)

                sim_params.append(new_params[0][0])

            else:  # Sample from uniform prior
                lwr_bound = self._params_prior[id][0]
                upr_bound = self._params_prior[id][1]
                param_val = np.random.uniform(lwr_bound, upr_bound)

                sim_params.append(param_val)

        return sim_params

    def sample_init_state(self):
        init_species = []

        for idx, s in enumerate(self._init_species_prior):
            if self._init_species_has_kde[idx]:
                species_val = self._init_state_kdes[idx].resample(1)

                while species_val[0][0] < 0:
                    species_val = self._init_state_kdes[idx].resample(1)

                init_species.append(species_val[0][0])

            else:
                lwr_bound = self._init_species_prior[s][0]
                upr_bound = self._init_species_prior[s][1]


                species_val = np.random.uniform(lwr_bound, upr_bound)
                init_species.append(species_val)

        return init_species

    def get_model_ref(self):
        return self._model_ref


##
# Model space is initialised with a list of Model objects.
#
# This class controls sampling of models, and generation of a population of particles to be simulated.
# Used to apply operations to all models within a model space, and reports on all models in the model space.
##
class ModelSpace:
    def __init__(self, model_list):
        self._model_list = model_list
        self._model_refs_list = [m.get_model_ref() for m in self._model_list]  # Extract model reference from each model
        self.accepted_particles = []
        self.dead_models_count = 0
        self.model_kernel = 0.1

    def count_dead_models(self):
        count = 0
        for m in self._model_list:
            if m.population_accepted_count[-1] == 0:
                count += 1

        self.dead_models_count = count

    def alt_generate_model_param_kdes(self, judgement_array, simulated_particles, input_params, fit_parameters):
        unique_models = self._model_list

        for m in unique_models:
            accepted_params = []
            for idx, part in enumerate(simulated_particles):
                if part is m and judgement_array[idx]:
                    accepted_params.append(input_params[idx])

            if len(accepted_params) > 1:
                m.alt_generate_params_kde(accepted_params, fit_parameters)

            else:
                continue

    def alt_generate_model_init_species_kdes(self, judgement_array, simulated_particles, input_init_state,
                                             fit_parameters):
        unique_models = self._model_list

        for m in unique_models:
            accepted_params = []
            for idx, part in enumerate(simulated_particles):
                if part is m and judgement_array[idx]:
                    accepted_params.append(input_init_state[idx])

            if len(accepted_params) > 1:
                m.alt_generate_species_kde(accepted_params, fit_parameters)

            else:
                continue

    ##
    # Generates new kdes for simulated models from the input parameters used for accepted particles
    ##
    def generate_model_kdes(self, judgement_array, simulated_particles, input_params):
        unique_models = self._model_list

        for m in unique_models:
            accepted_params = []
            for idx, part in enumerate(simulated_particles):
                if part is m and judgement_array[idx]:
                    accepted_params.append(input_params[idx])

            if len(accepted_params) > 1:
                m.generate_kde(accepted_params)

            else:
                continue

    def generate_model_kernels(self, accepted_particles, pop_num):
        unique_models = self._model_list

        for m in unique_models:
            accepted_params = []
            accepted_init_states = []

            for idx, particle in enumerate(accepted_particles):
                if particle.prev_model is m:
                    accepted_params.append(particle.prev_params)
                    accepted_init_states.append(particle.prev_init_state)

            if pop_num == 0:
                if len(accepted_params) >= 1:
                    m.generate_kernels(accepted_params, accepted_init_states)

            elif len(accepted_params) > 5:
                m.generate_kernels(accepted_params, accepted_init_states)

            else:
                continue

    ##
    # Appends a new entry for the counts of times sampled and times accepted in a population
    # Each time this is called
    ##
    def update_population_sample_data(self, simulated_particle_refs, judgement_array):
        unique_models = self._model_list

        for m in unique_models:
            sampled_count = 0
            accepted_count = 0

            for idx, particle_ref in enumerate(simulated_particle_refs):
                if particle_ref is m._model_ref:
                    sampled_count += 1

                    if judgement_array[idx]:
                        accepted_count += 1
            
            m.population_sample_count.append(sampled_count)
            m.population_accepted_count.append(accepted_count)

    def update_model_weights_naive(self, sigma=0.5):

        unique_models = self._model_list

        for this_model in self._model_list:
            n_sims = this_model.population_sample_count[-1]
            n_accepted = this_model.population_accepted_count[-1]
            self._current_sample_probability = n_accepted / n_sims

    ##
    # Samples model space based on???
    # Returns a list of models to be simulated
    ##
    def sample_model_space(self, n_sims):
        model_probabilities = [m._current_sample_probability for m in
                               self._model_list]  # Extract model sample probabilities

        # model probabilities greater than 1 at start first population, therefore we sample uniformally
        if sum(model_probabilities) > 1:
            model_probabilities = None

        sampled_models = np.random.choice(self._model_list, n_sims, p=model_probabilities)

        return sampled_models

    def sample_particles_from_prior(self, n_sims):
        model_probabilities = [m._current_sample_probability for m in
                               self._model_list]  # Extract model sample probabilities

        # model probabilities greater than 1 at start first population, therefore we sample uniformally
        if sum(model_probabilities) > 1:
            model_probabilities = None

        sampled_models = np.random.choice(self._model_list, n_sims, p=model_probabilities)

        particles = []

        for model in sampled_models:
            new_particle = Particle(model)
            new_particle.sample_params_from_prior()
            new_particle.sample_init_state_from_prior()
            particles.append(new_particle)

        return particles

    ##
    #   Samples particles from previous population based on model marginals, followed by 
    #   particle perturbation
    ##
    def sample_particles_from_previous_population(self, n_sims):

        # Sample model based on marginals
        model_marginals = [m.prev_margin for m in self._model_list]

        sampled_models = np.random.choice(self._model_list, n_sims, p=model_marginals)
        prev_models = sampled_models[:]

        all_alive_models = [m for m in self._model_list if m.prev_margin > 0]

        # Perturb models
        if self.dead_models_count > len(self._model_list) - 1:
            for i in range(n_sims):
                u = np.random.uniform(0, 1)

                if u > self.model_kernel:
                    # Randomly sample from alive models
                    perturbed_model = np.random.choice(all_alive_models)
                    sampled_models[i] = perturbed_model

        # Sample a particle from the previous population that has the corresponding model
        sampled_particles = []
        for i in range(n_sims):
            this_model = sampled_models[i]
            corresponding_particles = [p for p in self.prev_accepted_particles if p.prev_model == this_model]
            sampled_particles.append(np.random.choice(corresponding_particles))

        # Perturb particle parameters and init states
        perturbed_particles = []
        for i in range(n_sims):
            p_p = sampled_particles[i].perturb_particle()
            perturbed_particles.append(p_p)

        return perturbed_particles

    def compute_particle_weights(self):
        for particle in self.accepted_particles:

            model_params_prior = particle.curr_model._params_prior
            model_init_species_prior = particle.curr_model._init_species_prior

            particle_params = particle.curr_params
            particle_init_species = particle.curr_init_state

            this_model = particle.curr_model

            particle_prior = 1
            for idx, id in enumerate(sorted(model_params_prior, key=str.lower)):
                x = 1.0

                this_param_prior = model_params_prior[id]

                if this_param_prior[0] == this_param_prior[1]:
                    x = 1.0

                else:
                    x = alg_utils.get_pdf_uniform(this_param_prior[0], this_param_prior[1], particle_params[idx])

                particle_prior = particle_prior * x

            for idx, id in enumerate(model_init_species_prior):
                x = 1.0

                this_init_species_prior = model_init_species_prior[id]

                if this_init_species_prior[0] == this_init_species_prior[1]:
                    x = 1.0

                else:
                    x = alg_utils.get_pdf_uniform(this_init_species_prior[0], this_init_species_prior[1],
                                              particle_init_species[idx])

                particle_prior = particle_prior * x

            # Model prior of 1 assumes uniform model prior
            model_prior = 1
            numerator = model_prior * particle_prior

            s_1 = 0
            for m in self._model_list:
                s_1 += m.prev_margin * self.get_model_kernel_pdf(this_model, m, self.model_kernel,
                                                                 len(self._model_list), self.dead_models_count)

            s_2 = 0
            for p_j in self.prev_accepted_particles:
                if this_model == p_j.prev_model:
                    non_constant_idxs = [idx for idx, key in enumerate(sorted(model_params_prior, key=str.lower)) if model_params_prior[key][0] != model_params_prior[key][1]]
                    kernel_params_pdf = alg_utils.get_parameter_kernel_pdf(particle_params, p_j.prev_params,
                                                                           this_model.param_kernels, non_constant_idxs)

                    non_constant_idxs = [idx for idx, key in enumerate(model_init_species_prior) if model_init_species_prior[key][0] != model_init_species_prior[key][1]]
                    # kernel_init_state_pdf = alg_utils.get_parameter_kernel_pdf(particle_init_species,
                    #                                                            p_j.prev_init_state,
                    #                                                            this_model.init_state_kernels, non_constant_idxs)
                    kernel_init_state_pdf = 1
                    kernel_pdf = kernel_params_pdf * kernel_init_state_pdf

                    s_2 += p_j.prev_weight * kernel_pdf


            # print(this_model.prev_margin)
            # print(numerator)
            # print(s_1)
            # print(s_2)

            particle.curr_weight = this_model.prev_margin * numerator / (s_1 * s_2)

            # print("particle.curr_weight", particle.curr_weight)
            # print(particle.prev_model._model_ref)
            # print(particle.curr_model._model_ref)
            # print(particle.curr_model.prev_margin)
            # print(particle.prev_model.prev_margin)
            # print(numerator)
            # print(s_1)
            # print(s_2)
            # print("")


    def update_model_marginals(self):
        for m in self._model_list:
            m.curr_margin = 0
            for particle in self.accepted_particles:
                if particle.curr_model == m:
                    m.curr_margin += particle.curr_weight

    def normalize_particle_weights(self):
        sum_weights = sum([p.curr_weight for p in self.accepted_particles])
        for p in self.accepted_particles:
            p.curr_weight = p.curr_weight / sum_weights

    def get_model_kernel_pdf(self, new_model, old_model, model_k, num_models, dead_models):
        """
        Returns the probability of model number m0 being perturbed into model number m (assuming neither is dead).

        This assumes a uniform model perturbation kernel: with probability modelK the model is not perturbed; with
        probability (1-modelK) it is replaced by a model randomly chosen from the non-dead models (including the current
        model).

        Parameters
        ----------
        new_model : index of next model
        old_model : index of previous model
        model_k : model (non)-perturbation probability
        num_models : total number of models
        dead_models : number of models which are 'dead'
        """

        num_dead_models = self.dead_models_count

        if num_dead_models == num_models - 1:
            return 1.0
        else:
            if new_model == old_model:
                return model_k
            else:
                return (1 - model_k) / (num_models - num_dead_models)

    def model_space_report(self, output_dir, batch_num, use_sum=False):
        file_path = output_dir + 'model_space_report.csv'
        column_names = ['model_idx', 'accepted_count', 'simulated_count', 'model_marginal']
        models_data = []
        for m in self._model_list:
            if use_sum:
                models_data.append([m.get_model_ref(), sum(m.population_accepted_count), sum(m.population_sample_count),
                                    m.prev_margin])

            else:
                models_data.append([m.get_model_ref(), m.population_accepted_count[-1], m.population_sample_count[-1],
                                    m.prev_margin])

        new_df = pd.DataFrame(data=models_data, columns=column_names)
        new_df.to_csv(file_path)

    def prepare_next_population(self):
        for particle in self.accepted_particles:
            particle.prev_weight = particle.curr_weight
            particle.prev_params = particle.curr_params[:]
            particle.prev_init_state = particle.curr_init_state[:]
            particle.prev_model = particle.curr_model

            particle.curr_weight = None
            particle.curr_params = None
            particle.curr_init_state = None
            particle.curr_model = None


        for model in self._model_list:
            model.prev_margin = model.curr_margin
            model.curr_margin = 0


        self.prev_accepted_particles = self.accepted_particles[:]
        self.accepted_particles = []