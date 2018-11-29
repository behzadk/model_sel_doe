



##
# Generates particles corresponding to each model in the model list
##
def generate_particles(models_list):
    pop_model_refs = [m.get_model_ref() for m in models_list]
    pop_params_list = []

    for m in models_list:
        params = m.sample_particle()
        pop_params_list.append(params)

    return pop_params_list, pop_model_refs


##
# Compares the distances of each spiecies of each particle to the epsilons in the distance
# array.
#
# Returns a list of bools True - accept particle, False - reject particle
##
def check_distances(particle_distances, epsilon_array):
    particle_judgements = []

    for particle in particle_distances:
        particle_accept = True
        for species in particle:
            for epsilon_idx, dist in enumerate(species):
                if dist >= epsilon_array[epsilon_idx]:
                    particle_accept = False

        particle_judgements.append(particle_accept)

    return particle_judgements
