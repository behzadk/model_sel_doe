import numpy as np
from scipy import stats

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
        self._has_kde = False

        # Model data. Number of times this model was sampled in each population and number of times it was accepted
        self.population_sample_count = []
        self.population_accepted_count = []

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

        self._has_kde = True

    ##
    # Samples simulation parameters for n_sims using the model KDEs
    ##
    def sample_particle(self):
        if self._has_kde:  # Sample from kde
            model_sim_params = []
            for idx, kern in enumerate(self._param_kdes):
                if isinstance(kern, stats.kde.gaussian_kde):
                    new_params = kern.resample(1)
                    model_sim_params.append(new_params[0][0])

                elif isinstance(kern, np.float64):
                    model_sim_params.append(kern)

                else:
                    raise ("Type not recognised")


            return model_sim_params

        else:  # Sample from uniform prior
            sim_params = []

            for param in self._prior:
                lwr_bound = self._prior[param][0]
                upr_bound = self._prior[param][1]
                param_val = np.random.uniform(lwr_bound, upr_bound)
                sim_params.append(param_val)

            return sim_params


    def get_model_ref(self):
        return self._model_ref


##
# Model space is initialised with a list of Model objects.
#
# This class controls sampling of models, and generation of a population of particles to be simulated
##
class ModelSpace:
    def __init__(self, model_list):
        self._model_list = model_list
        self._model_refs_list = [m.get_model_ref() for m in self._model_list]  # Extract model reference from each model

    ##
    # Samples model space based on???
    # Returns a list of models to be simulated
    ##
    def sample_model_space(self, n_sims):
        sampled_models = np.random.choice(self._model_list, n_sims)

        return sampled_models

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
                print("accepted params <=  1")


    ##
    # Appends a new entry for the counts of times sampled and times accepted in a population
    # Each time this is called
    ##
    def update_model_data(self, simulated_particles, judgement_array):
        unique_models = self._model_list

        for m in unique_models:
            sampled_count = 0
            accepted_count = 0

            for idx, particle in enumerate(simulated_particles):
                if particle is m:
                    sampled_count +=1

                if judgement_array[idx]:
                    accepted_count +=1

            m.population_sample_count.append(sampled_count)
            m.population_accepted_count.append(accepted_count)