import numpy as np

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
# Compares the distances of each spiecies of each particle to the epsilons in the distance
# array.
#
# Returns a list of bools True - accept particle, False - reject particle
##
def check_distances(particle_distances, epsilon_array):
    particle_judgements = []

    for part_distance in particle_distances:
        particle_accept = True

        if np.isnan(part_distance).any():
            particle_accept = False
            continue

        else:
            for species in part_distance:
                for epsilon_idx, dist in enumerate(species):
                    if dist >= epsilon_array[epsilon_idx]:
                        particle_accept = False

        particle_judgements.append(particle_accept)

    return particle_judgements
