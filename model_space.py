import numpy as np
from scipy import stats
import csv
import pandas as pd
import os

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
        self._param_kdes = []
        self._has_kde = False
        self._sample_probability = 1

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
    # Samples simulation parameters.
    # If KDE has been generated, parameters are sampled from KDE
    # Else, parameters are sampled from the prior
    ##
    def sample_particle(self):
        if self._has_kde:  # Sample from kde
            model_sim_params = []
            for idx, kern in enumerate(self._param_kdes):
                if isinstance(kern, stats.kde.gaussian_kde):
                    new_params = kern.resample(1)

                    # Prevent sampling of negative parameter values
                    while new_params < 0:
                        new_params = kern.resample(1)

                    model_sim_params.append(new_params[0][0])

                elif isinstance(kern, np.float64):
                    model_sim_params.append(kern)

                else:
                    raise("Type not recognised")

            return model_sim_params

        else:  # Sample from uniform prior
            sim_params = []

            for param in self._params_prior:
                lwr_bound = self._params_prior[param][0]
                upr_bound = self._params_prior[param][1]
                param_val = np.random.uniform(lwr_bound, upr_bound)
                sim_params.append(param_val)

            return sim_params

    def sample_init_state(self):
        init_species = []

        for s in self._init_species_prior:
            lwr_bound = self._init_species_prior[s][0]
            upr_bound = self._init_species_prior[s][1]
            param_val = np.random.uniform(lwr_bound, upr_bound)
            init_species.append(param_val)

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

    ##
    # Appends a new entry for the counts of times sampled and times accepted in a population
    # Each time this is called
    ##
    def update_model_population_sample_data(self, simulated_particles, judgement_array):
        unique_models = self._model_list

        for m in unique_models:
            sampled_count = 0
            accepted_count = 0

            for idx, particle in enumerate(simulated_particles):
                if particle is m:
                    sampled_count += 1

                    if judgement_array[idx]:
                        accepted_count += 1

            m.population_sample_count.append(sampled_count)
            m.population_accepted_count.append(accepted_count)

    ##
    #   Updates the model weights based upon the acceptance rate of the model in the most recent population:
    #   P = #Accepted / #Sampled
    ##
    def update_model_sample_probabilities(self):
        sum_accepted = sum(m.population_accepted_count[-1] for m in self._model_list)

        for m in self._model_list:
            num_accepted = m.population_accepted_count[-1]
            m._sample_probability = num_accepted / sum_accepted

    ##
    # Samples model space based on???
    # Returns a list of models to be simulated
    ##
    def sample_model_space(self, n_sims):
        model_probabilities = [m._sample_probability for m in self._model_list] # Extract model sample probabilities

        # model probabilities greater than 1 at start first population, therefore we sample uniformally
        if sum(model_probabilities) > 1:
            model_probabilities = None

        print(model_probabilities)
        sampled_models = np.random.choice(self._model_list, n_sims, p=model_probabilities)

        return sampled_models

    def model_space_report(self, output_dir, batch_num):
        file_path = output_dir + 'model_space_report.csv'
        column_names = ['model_idx', 'accepted_count', 'simulated_count']
        models_data = []
        for m in self._model_list:
            models_data.append([m.get_model_ref(), sum(m.population_accepted_count), sum(m.population_sample_count)])

        new_df = pd.DataFrame(data=models_data, columns=column_names)
        new_df.to_csv(file_path)