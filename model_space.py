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

        use_special_uniform = False
        use_component_normal = True

        # Perturb particles
        if use_special_uniform:
            for idx, id in enumerate(sorted(self.prev_model._params_prior, key=str.lower)):
                kernel = self.prev_model.param_kernels[idx]
                kern_lwr = -kernel
                kern_upr = kernel

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

                kern_lwr = - kernel
                kern_upr = kernel

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

        elif use_component_normal:
            for idx, id in enumerate(sorted(self.prev_model._params_prior, key=str.lower)):
                prior_min = self.prev_model._params_prior[id][0]
                prior_max = self.prev_model._params_prior[id][1]

                if self.prev_model._params_prior[id][2] == "constant":
                    perturbed_param = prior_max

                else:
                    param_illegal = True
                    while param_illegal:
                        previous_param = self.prev_params[idx]
                        kernel = self.prev_model.param_kernels[idx]
                        perturbed_param = np.random.normal(loc=previous_param, scale=np.sqrt(kernel))
                        x = alg_utils.get_pdf_uniform(prior_min, prior_max, perturbed_param)
                        
                        if x > 0:
                            param_illegal = False
                
                new_params.append(perturbed_param)

            for idx, id in enumerate(self.prev_model._init_species_prior):
                prior_min = self.prev_model._init_species_prior[id][0]
                prior_max = self.prev_model._init_species_prior[id][1]

                if self.prev_model._init_species_prior[id][2] == "constant":
                    perturbed_param = prior_max

                else:
                    param_illegal = True

                    while param_illegal:
                        previous_init_species = self.prev_init_state[idx]
                        kernel = self.prev_model.init_state_kernels[idx]
                        perturbed_param = np.random.normal(loc=previous_init_species, scale=np.sqrt(kernel))
                        x = alg_utils.get_pdf_uniform(prior_min, prior_max, perturbed_param)

                        if x > 0:
                            param_illegal = False


                new_init_state.append(perturbed_param)

        # Calculate liklihood
        prior_prob = 1
        for idx, id in enumerate(sorted(self.prev_model._params_prior, key=str.lower)):
            prior_min = self.prev_model._params_prior[id][0]
            prior_max = self.prev_model._params_prior[id][1]

            if self.prev_model._params_prior[id][2] == "constant":
                x = 1.0

            else:
                x = alg_utils.get_pdf_uniform(prior_min, prior_max, new_params[idx])

            prior_prob = prior_prob * x

        # print("")
        for idx, id in enumerate(self.prev_model._init_species_prior):
            prior_min = self.prev_model._init_species_prior[id][0]
            prior_max = self.prev_model._init_species_prior[id][1]

            if self.prev_model._init_species_prior[id][2] == "constant":
                x = 1.0


            else:
                x = alg_utils.get_pdf_uniform(prior_min, prior_max, new_init_state[idx])


            prior_prob = prior_prob * x


        if prior_prob == 0:
            return False

        else:
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

            if param[0] == param[1]:
                self._params_prior[param_id].append('constant')

            # if abs(np.log10(param[0]) - np.log10(param[1])) > log_decimals:
            elif (param[0] < 1e-4 and param[0] != 0) or (param[0] > 1e4):
                self._params_prior[param_id].append('log')
                self._params_prior[param_id][0] = np.log(self._params_prior[param_id][0])
                self._params_prior[param_id][1] = np.log(self._params_prior[param_id][1])

            else:
                self._params_prior[param_id].append('uniform')

        for species_id in self._init_species_prior:
            species = self._init_species_prior[species_id]

            if species[0] == species[1]:
                self._init_species_prior[species_id].append('constant')

            # if abs(np.log10(species[0]) - np.log10(species[1])) > log_decimals:
            elif (species[0] < 1e-4 and species[0] != 0) or (species[0] > 1e4):
                self._init_species_prior[species_id].append('log')
                self._init_species_prior[species_id][0] = np.log(self._init_species_prior[species_id][0])
                self._init_species_prior[species_id][1] = np.log(self._init_species_prior[species_id][1])

            else:
                self._init_species_prior[species_id].append('uniform')


    def generate_kernels(self, params, init_states, weights):
        params = np.array(params)
        init_states = np.array(init_states)
        # params = np.array([np.array(p)for p in params])
        # init_states = np.array([np.array(s)for s in init_states])

        # params = np.array(params, dtype=object)
        # init_states = np.array(init_states, dtype=object)

        self.param_kernels = []
        self.param_kern_aux = []
        self.init_state_kernels = []
        self.init_state_kern_aux = []

        use_prior = False
        use_uniform = False
        use_normal = True


        if np.shape(params)[0] == 1:
            use_prior = True
            use_uniform = False
            use_normal = False

        if use_prior:
            params_prior = self._params_prior
            init_species_prior = self._init_species_prior


            for idx, id in enumerate(sorted(self._params_prior, key=str.lower)):
                min_param = params_prior[id][0]
                max_param = params_prior[id][1]
                scale = abs(max_param - min_param) / 2.0
                # print(id, scale)

                self.param_kernels.append(scale)

            for idx, s in enumerate(self._init_species_prior):
                min_init_state = init_species_prior[s][0]
                max_init_state = init_species_prior[s][1]
                
                scale = abs(max_init_state - min_init_state) / 2.0

                self.init_state_kernels.append(scale)


        elif use_uniform:
            if np.shape(params)[0] == 1:
                for p_idx in range(np.shape(params)[1]):
                    self.param_kernels.append(1)

                for s_idx in range(np.shape(init_states)[1]):
                    self.init_state_kernels.append(1)
                

            else:
                for p_idx in range(np.shape(params)[1]):
                    min_param = min(params[:, p_idx])
                    max_param = max(params[:, p_idx])


                    scale = abs(max_param - min_param) / 2.0
                    self.param_kernels.append(scale)

                for s_idx in range(np.shape(init_states)[1]):
                    min_init_state = min(init_states[:, s_idx])
                    max_init_state = max(init_states[:, s_idx])

                    scale = abs(max_init_state - min_init_state) / 2.0

                    self.init_state_kernels.append(scale)

        elif use_normal:
            for p_idx in range(np.shape(params)[1]):
                s2w = alg_utils.get_wt_var(params[:, p_idx], weights)
                self.param_kernels.append(s2w *2)
                if np.isnan(s2w):
                    print(s2w)
                    print(weights)
                    print(params[:, p_idx])
                    exit()


            for s_idx in range(np.shape(init_states)[1]):
                s2w = alg_utils.get_wt_var(init_states[:, s_idx], weights)

                self.init_state_kernels.append(s2w * 2)


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
        self.model_kernel = 0.7

    def count_dead_models(self):
        count = 0
        for m in self._model_list:
            if m.population_accepted_count[-1] == 0:
                count += 1

        self.dead_models_count = count



    def generate_model_kernels(self, accepted_particles, pop_num):
        unique_models = self._model_list

        for m in unique_models:
            accepted_params = []
            accepted_init_states = []
            accepted_weights = []

            for idx, particle in enumerate(accepted_particles):
                if particle.prev_model._model_ref is m._model_ref:
                    accepted_params.append(particle.prev_params)
                    accepted_init_states.append(particle.prev_init_state)
                    accepted_weights.append(particle.prev_weight)

            if pop_num == 0:
                if len(accepted_params) >= 1:
                    m.generate_kernels(accepted_params, accepted_init_states, accepted_weights)

            elif len(accepted_params) >= 1:
                m.generate_kernels(accepted_params, accepted_init_states, accepted_weights)

            else:
                continue



    ##
    # Appends a new entry for the counts of times sampled and times accepted in a population
    # Each time this is called
    ##
    def update_population_sample_data(self, simulated_particle_refs, judgement_array):
        unique_models = self._model_list

        sampled_count = [0 for i in range(len(unique_models))]
        accepted_count = [0 for i in range(len(unique_models))]

        for idx, particle_ref in enumerate(simulated_particle_refs):
            sampled_count[particle_ref] = sampled_count[particle_ref] + 1

            if judgement_array[idx]:
                accepted_count[particle_ref] = accepted_count[particle_ref] + 1


        for m in unique_models:
            m.population_sample_count.append(sampled_count[m._model_ref])
            m.population_accepted_count.append(accepted_count[m._model_ref])

    # ##
    # # Appends a new entry for the counts of times sampled and times accepted in a population
    # # Each time this is called
    # ##
    # def update_population_sample_data(self, simulated_particle_refs, judgement_array):
    #     unique_models = self._model_list

    #     for m in unique_models:
    #         sampled_count = 0
    #         accepted_count = 0

    #         for idx, particle_ref in enumerate(simulated_particle_refs):

    #             if particle_ref == m._model_ref:
    #                 sampled_count += 1

    #                 if judgement_array[idx]:
    #                     accepted_count += 1

    #         m.population_sample_count.append(sampled_count)
    #         m.population_accepted_count.append(accepted_count)

    def update_model_weights_naive(self, sigma=0.5):

        unique_models = self._model_list

        for this_model in self._model_list:
            n_sims = this_model.population_sample_count[-1]
            n_accepted = this_model.population_accepted_count[-1]
            self._current_sample_probability = n_accepted / n_sims


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
            p_p = False
            
            while p_p == False:
                p_p = sampled_particles[i].perturb_particle()

            p_p.curr_model = sampled_particles[i].prev_model
            p_p.prev_model = sampled_particles[i].prev_model

            perturbed_particles.append(p_p)

        return perturbed_particles

    def compute_particle_weights(self):
        for particle in self.accepted_particles:

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
            for m in self._model_list:
                s_1 += m.prev_margin * self.get_model_kernel_pdf(this_model, m, self.model_kernel,
                                                                 len(self._model_list), self.dead_models_count)
            # Particle denominator
            s_2 = 0.0
            for p_j in self.accepted_particles:
                if this_model._model_ref == p_j.prev_model._model_ref:
                    non_constant_idxs = [idx for idx, key in enumerate(sorted(model_params_prior, key=str.lower)) if model_params_prior[key][0] != model_params_prior[key][1]]
                    
                    kernel_params_pdf = alg_utils.get_parameter_kernel_pdf(particle_params, p_j.prev_params,
                                                                           this_model.param_kernels, non_constant_idxs, this_model._params_prior, p_j.param_aux_info, use_uniform_kernel=False, use_normal_kernel=True)
                   
                    non_constant_idxs = [idx for idx, key in enumerate(model_init_species_prior) if model_init_species_prior[key][0] != model_init_species_prior[key][1]]
                    
                    kernel_init_state_pdf = alg_utils.get_parameter_kernel_pdf(particle_init_species,
                                                                               p_j.prev_init_state,
                                                                               this_model.init_state_kernels, non_constant_idxs,  this_model._params_prior, p_j.init_state_aux_info, use_uniform_kernel=False, use_normal_kernel=True)
                    kernel_pdf = kernel_params_pdf * kernel_init_state_pdf
                    s_2 = s_2 + p_j.prev_weight * kernel_pdf


            particle.curr_weight = this_model.prev_margin * numerator / (s_1 * s_2)

            # print(this_model.prev_margin)
            # print(numerator)
            # print(s_1)
            # print(s_2)
            # if this_model._model_ref == 3:
            #     print(count)
            #     print(this_model.param_kernels)
            #     print("model_ref: ", particle.curr_model._model_ref)
            #     print("particle.curr_weight: ", particle.curr_weight)
            #     print("prev margin: ", particle.curr_model.prev_margin)
            #     print("prior prob", particle_prior_prob)
            #     print("numer", numerator)
            #     print("ratio", numerator/sum_pdfs)

            #     exit()


    def update_model_marginals(self):
        new_model_weights = [0 for i in range(len(self._model_list))]

        for particle in self.accepted_particles:
            new_model_weights[particle.curr_model._model_ref] = new_model_weights[particle.curr_model._model_ref] + particle.curr_weight


        for m in self._model_list:
            m.curr_margin = new_model_weights[m._model_ref]

        # for m in self._model_list:
        #     m.curr_margin = 0
        #     for particle in self.accepted_particles:
        #         if particle.curr_model._model_ref == m._model_ref:
        #             m.curr_margin += particle.curr_weight


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


    def generate_kernel_aux_info(self):
        for particle in self.prev_accepted_particles:
            param_aux_info = []
            init_state_aux_info = []
            
            for idx, id in enumerate(sorted(particle.prev_model._params_prior, key=str.lower)):
                prior_max = particle.prev_model._params_prior[id][1]
                prior_min = particle.prev_model._params_prior[id][0]

                if prior_min == prior_max:
                    param_aux_info.append(1.0)

                else:
                    mean = particle.prev_params[idx]
                    scale = np.sqrt(particle.prev_model.param_kernels[idx])
                    aux = stats.norm.cdf(prior_max, mean, scale) - stats.norm.cdf(prior_min, mean, scale)
                    param_aux_info.append(aux)

            for idx, id in enumerate(particle.prev_model._init_species_prior):
                prior_max = particle.prev_model._init_species_prior[id][1]
                prior_min = particle.prev_model._init_species_prior[id][0]

                if prior_min == prior_max:
                    init_state_aux_info.append(1.0)

                else:
                    mean = particle.prev_init_state[idx]
                    scale = np.sqrt(particle.prev_model.init_state_kernels[idx])
                    aux = stats.norm.cdf(prior_max, mean, scale) - stats.norm.cdf(prior_min, mean, scale)

                    init_state_aux_info.append(aux)

            particle.param_aux_info = param_aux_info
            particle.init_state_aux_info = init_state_aux_info


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


        self.prev_accepted_particles = self.accepted_particles[:]
        self.accepted_particles = []