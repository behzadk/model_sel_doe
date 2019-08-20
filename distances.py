import numpy as np

def osc_dist():
	sim_distances = []

	max_dist = np.inf

	max_distances = [max_dist, max_dist, max_dist];

	if (integration_failed) {
		for (auto it = species_to_fit.begin(); it != species_to_fit.end(); it++) {
			sim_distances.push_back(max_distances);
		}

		return sim_distances;
	}


	int from_time_index = 900;
	std::cout << "Oscillatory distances" << std::endl;

	for (auto it = species_to_fit.begin(); it != species_to_fit.end(); it++) {
		std::vector<double> signal = extract_species_to_fit(state_vec, *it, from_time_index);
		std::vector<double> signal_gradient = get_signal_gradient(signal);
		
		std::vector<int> peak_indexes;
		std::vector<int> trough_indexes;

		find_signal_peaks_and_troughs(signal_gradient, peak_indexes, trough_indexes);
		// test_fft(signal);
	}

	return sim_distances;
}
