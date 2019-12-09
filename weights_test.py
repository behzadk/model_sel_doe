import numpy as np
import model_space

def make_KP(previous_params):
    pass


def numerical_test():
    kernel = 0.1

    numer = 0.2

    a = (numer * 50) / (kernel * 50)
    b = (numer * 5) / (kernel * 5)



def get_pdf_uniform(lower_bound, upper_bound, x):
    if (x > upper_bound) or (x < lower_bound):
        return 0.0

    else:
        return (1 / (upper_bound - lower_bound))


def weight_problem_example():
    # Two priors, one small one large
    prior_a = [[1, 5]]
    model_a = "a"

    prior_b = [[1, 5], [1, 5]]
    model_b = "b"

    n_particles = 2000
    # Previous population had equal weights
    prev_weight_a = 1/2000
    prev_weight_b = 1/2000

    # Sampled parameters are within prior range
    particle_a_params = [np.random.uniform(x[0], x[1]) for x in prior_a]
    particle_b_params = [np.random.uniform(x[0], x[1]) for x in prior_b]

    particle_a_prev_params = [np.random.uniform(x[0], x[1]) for x in prior_a]
    particle_b_prev_params = [np.random.uniform(x[0], x[1]) for x in prior_b]



    packed_a = [prior_a, particle_a_params, particle_a_prev_params, prev_weight_a, model_a]
    packed_b = [prior_b, particle_b_params, particle_b_prev_params, prev_weight_b, model_b]

    particle_a_scales = []
    all_p = []
    for p_idx, _ in enumerate(prior_a):
        print(p_idx)
        print(type(packed_a[2][p_idx]))
        all_p.append(packed_a[2][p_idx])

    print(all_p)
    print(np.max(all_p))
    print(np.min(all_p))
    exit()
    particle_a_scales.append((np.max(all_p) - np.min(all_p) )/ 2)

    particle_b_scales = []
    for p_idx, _ in enumerate(prior_b):
        all_p = []
        all_p.append(packed_b[2][p_idx])
        particle_b_scales.append((np.max(all_p) - np.min(all_p) )/ 2)

    packed_a.append(particle_a_scales)
    packed_b.append(particle_b_scales)

    particles = []
    particles = particles + [packed_a for x in range(1000)]
    particles =  particles + [packed_b for x in range(1000)]
    model_a_weights = []
    model_b_weights = []
    for part in particles:
        # unpack particle data
        prior = part[0]
        params = part[1]
        prev_params = part[2]
        prev_weight = part[3]
        model = part[4]

        # Numerator
        particle_prior_prob = 1
        for idx, p in enumerate(params):
            particle_prior_prob = particle_prior_prob * get_pdf_uniform(prior[idx][0], prior[idx][1], p)

        # Assume equal model probability
        particle_prior_prob = particle_prior_prob * 1


        # Denominator, assuming we only have two particles
        s = 0
        for theta in particles:
            if theta[4] == model:
                theta_prior = theta[0]
                theta_params = theta[1]
                theta_prev_params = theta[2]
                theta_prev_weight = theta[3]
                theta_model = theta[4]
                theta_scales = theta[5]
                print(theta_scales)

                k_prob = 1
                for theta_idx, theta_p in enumerate(theta_prev_params):
                    # get parameter kernel pdf
                    x = get_pdf_uniform(
                        theta_p - theta_scales[theta_idx]/2, 
                        theta_p + theta_scales[theta_idx]/2, 
                        params[theta_idx])

                    # if theta_model == "b":
                    print(theta_p)
                    print(theta_p - theta_scales[theta_idx][0]/2)
                    print(theta_p + theta_scales[theta_idx][1]/2)
                    print(params[theta_idx])
                    print(x)
                    print("")

                    k_prob = k_prob * x

                # print(s)
                s += ( k_prob)
                # print(s)

        part_weight = particle_prior_prob/s

        if model == "a":
            model_a_weights.append(part_weight)

        if model == "b":
            model_b_weights.append(part_weight)

    print(np.mean(model_a_weights))
    print(np.mean(model_b_weights))


def weight_problem_example_2():
    # Two priors, one small one large
    prior_a = [[1, 5]]
    model_a = "a"

    prior_b = [[1, 5], [1, 5], [1, 5], [1, 5], [1, 5], [1, 5]]
    model_b = "b"

    
    n_particles = 100

    accepted_particles = []

    # Sample some from prior evenly
    for x in range(int(n_particles/2)):
        new_particle = model_space.Particle("A")
        new_particle.curr_params = [np.random.uniform(x[0], x[1]) for x in prior_a]
        new_particle.prior = prior_a
        new_particle.curr_weight = 1/n_particles
        accepted_particles.append(new_particle)

    for x in range(int(n_particles/2)):
        new_particle = model_space.Particle("B")
        new_particle.curr_params = [np.random.uniform(x[0], x[1]) for x in prior_b]
        new_particle.prior = prior_b
        new_particle.curr_weight = 1/n_particles
        accepted_particles.append(new_particle)


    ## Population 0    
    # Assume all particles are accepted
    for part in accepted_particles:
        part.prev_params = part.curr_params[:]
        part.curr_params = []
        part.prev_weight = part.curr_weight

    # Make param kernels for each model
    model_A_param_kernel_scales = []
    model_B_param_kernel_scales = []

    # Model A
    for p_idx, _ in enumerate(prior_a):
        pop_params = []
        for part in accepted_particles:
            if part.curr_model == "A":
                pop_params.append(part.prev_params[p_idx])

        model_A_param_kernel_scales.append((np.max(pop_params) - np.min(pop_params))/2)
    
    # Model B
    for p_idx, _ in enumerate(prior_b):
        pop_params = []
        for part in accepted_particles:
            if part.curr_model == "B":
                pop_params.append(part.prev_params[p_idx])

        model_B_param_kernel_scales.append((np.max(pop_params) - np.min(pop_params))/2)

    # assign scales to models
    for part in accepted_particles:
        if part.curr_model == "A":
            part.scales = model_A_param_kernel_scales

        if part.curr_model == "B":
            part.scales = model_B_param_kernel_scales


    ## Population 1

    # Resample from the prior, imagining it is a new population 
    for part in accepted_particles:
        part.curr_params = [np.random.uniform(x[0], x[1]) for x in part.prior]


    # Calculate weights
    for this_part in accepted_particles:
        
        # Prior probability
        particle_prior_prob = 1
        for param_idx, param_prior in enumerate(this_part.prior):
            get_pdf_uniform(param_prior[0], param_prior[1], this_part.curr_params[param_idx])

        # Perturbation probability
        S = 0
        count = 0 
        for p_j in accepted_particles:
            if p_j.curr_model == this_part.curr_model:
                kernel_pdf = 1
                for param_idx, _ in enumerate(p_j.prev_params):
                    count +=1
                    kernel_pdf = kernel_pdf * get_pdf_uniform(p_j.prev_params[param_idx] - this_part.scales[param_idx], 
                        p_j.prev_params[param_idx] + this_part.scales[param_idx], this_part.curr_params[param_idx])

                S += kernel_pdf * (p_j.prev_weight)

        this_part.curr_weight = particle_prior_prob / S


    model_A_weights = []
    model_B_weights = []

    for part in accepted_particles:
        if part.curr_model == "A":
            model_A_weights.append(part.curr_weight)

        if part.curr_model == "B":
            model_B_weights.append(part.curr_weight)


    print(np.sum(model_A_weights))
    print(np.sum(model_B_weights))



if __name__ == "__main__":
    # numerical_test()
    # exit()
    # weight_problem_example()
    weight_problem_example_2()
