import numpy as np
import pandas as pd

def reload_experiment_from_posterior(posterior_path, init_species, init_params, sim_idx, batch_num):
    df = pd.read_csv(posterior_path)
    df = df.loc[df['sim_idx'] == sim_idx]
    df = df.loc[df['batch_num'] == batch_num]
    df = df.iloc[0]
    print(df)
    for param, _ in init_params.items():
        init_params[param] = [df[param], df[param]]

    for species, _ in init_species.items():
        init_params[species] = [df[species], df[species]]


    return init_species, init_params

##
# Generates particles consisting of a initial state and parameter sample from a list of models.
#
##
def generate_particles(models_list):
    pop_model_refs = [m.get_model_ref() for m in models_list]
    pop_params_list = []
    init_state_list = []


    for m in models_list:
        params = m.sample_particle()
        init_state = m.sample_init_state()

        init_state_list.append(init_state)
        pop_params_list.append(params)

    return init_state_list, pop_params_list, pop_model_refs


##
# Compares the distances of each species of each particle to the epsilons in the distance
# array.
#
# Returns a list of bools True - accept particle, False - reject particle
##
def check_distances_stable(particle_distances, epsilon_array):
    particle_judgements = []

    for part_distance in particle_distances:
        particle_accept = True

        for species_distances in part_distance:
            if np.isnan(species_distances).any() or any(p > 1e300 for p in species_distances):
                particle_accept = False
                
            for epsilon_idx, dist in enumerate(species_distances):
                if dist > epsilon_array[epsilon_idx]:
                    particle_accept = False

        particle_judgements.append(particle_accept)

    return particle_judgements

def check_distances_osc(particle_distances, epsilon_array):
    particle_judgements = []
    for part_distance in particle_distances:
        particle_accept = True

        for species_distances in part_distance:
            # Check if any of the distances for the fitted species are infinite or overflow
            if np.isnan(species_distances).any() or any(p > 1e300 for p in species_distances):
                particle_accept = False

            for epsilon_idx, dist in enumerate(species_distances):
                # threshold_amplitudes_count, 
                if epsilon_idx == 0 and dist < epsilon_array[epsilon_idx]:
                    particle_accept = False


                # final_amplitude
                elif epsilon_idx == 1 and dist < epsilon_array[epsilon_idx]:
                    particle_accept = False

                # signal_period_freq
                elif epsilon_idx == 2 and dist > epsilon_array[epsilon_idx]:
                    particle_accept = False

        particle_judgements.append(particle_accept)
        # print(particle_judgements)

    return particle_judgements


def update_epsilon(current_epsilon, final_epsilon, accepted_particle_distances, alpha):
    n_epsilon = len(final_epsilon)

    n_keep = int(alpha * len(accepted_particle_distances))
    new_epsilon = [0 for x in range(n_epsilon)]

    # Iterate species in each accepted simulation
    for e_idx in range(n_epsilon):

        # List of distances for this particular epsilon
        epsilon_accepted_distances = []

        # Iterate particles, keeping the largest distance in the species list
        for part_dists in accepted_particle_distances:
            epsilon_accepted_distances.append(max(part_dists[:, e_idx]))

        epsilon_accepted_distances.sort()
        epsilon_accepted_distances = epsilon_accepted_distances[:n_keep]
        new_epsilon[e_idx] = epsilon_accepted_distances[-1]

    for e_idx, new_e in enumerate(new_epsilon):
        if new_e < final_epsilon[e_idx]:
            new_epsilon[e_idx] = final_epsilon[e_idx]

    return new_epsilon

def fsolve_conversion(y, pop_obj, particle_ref, n_species):
    if type(y) == np.ndarray:
        y = y.tolist()

    sol = pop_obj.py_model_func(y, particle_ref)

    return sol

def fsolve_jac_conversion(y, pop_obj, particle_ref, n_species):
    if type(y) == np.ndarray:
        y = y.tolist()

    jac = pop_obj.get_particle_jacobian(y, particle_ref)

    jac = np.reshape(jac, (n_species, n_species))

    return jac

def make_posterior_kdes(model, posterior_df, init_params, init_species):
    for param_key, _ in init_params.items():
        posterior_values = posterior_df[param_key].values
        model.alt_generate_params_kde(posterior_values, param_key)

    for species_key, _ in init_species.items():
        posterior_values = posterior_df[species_key].values
        model.alt_generate_params_kde(posterior_values, species_key)
