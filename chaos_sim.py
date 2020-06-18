import population_modules

def chaos_sim(ABC, init_states, init_params, model_refs, theta=10**-8):
	init_t0 = 0.0
	init_t_end = 5000.0
	init_dt = ABC.dt

	init_epsilon = 0.001

	# Generate particles for initial trajectories
    pop_obj = population_modules.Population(ABC.n_sims_batch, init_t0, init_t_end,
                                                 init_dt, init_states, init_params, model_refs,
                                                 ABC.fit_species, ABC.abs_tol, ABC.rel_tol)
    # Simulate particles
    self.pop_obj.generate_particles()
    start_time_sim = time.time()
    self.pop_obj.simulate_particles()
    end_time_sim = time.time()

    # Get final states of successful sims
    self.pop_obj.calculate_particle_distances(self.distance_function_mode)
    self.pop_obj.accumulate_distances()

    batch_distances = self.pop_obj.get_flattened_distances_list()
    batch_distances = np.reshape(batch_distances,
                                 (self.n_sims_batch, len(self.fit_species), self.n_distances))

    batch_part_judgements = alg_utils.check_distances_generic(batch_distances,
                                                             epsilon_array=init_epsilon)
    accepted_particles = [p for p, judgement in zip(particles, batch_part_judgements) if judgement]

    # Generate new particles 
    particles = accepted_particles
    init_states = [copy.deepcopy(p.curr_init_state) for p in particles]
    input_params = [copy.deepcopy(p.curr_params) for p in particles]

    particle_models = [p.curr_model for p in particles]
