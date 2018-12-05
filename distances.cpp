#include <array>
#include <iostream>
#include <vector>
#include <math.h>
// #include "particle_sim_opemp.h"
#include "distances.h"
#include <algorithm>
#include <numeric>

DistanceFunctions::DistanceFunctions() {

}


std::vector<double> DistanceFunctions::extract_species_to_fit(std::vector<state_type>& state_vec, int species_idx)
{
	std::vector<double> species_val_vec;
	for (auto tp_iter = state_vec.begin(); tp_iter != state_vec.end(); tp_iter++) {
        state_type sim_vec = *tp_iter;
        species_val_vec.push_back(sim_vec[species_idx]);
    }

    return species_val_vec;
}

std::vector<double> DistanceFunctions::get_signal_gradient(std::vector<double>& signal)
{
	std::vector<double> signal_gradient;

	for (int i = 1; i <= signal.size(); i++) {
		double grad = signal[i] - signal[i-1];
		signal_gradient.push_back(grad);
	}

	return signal_gradient;

}

double DistanceFunctions::calculate_final_gradient(std::vector<double>& species_vals)
{
	double i = abs(species_vals.end()[-2]);
	double j = abs(species_vals.end()[-1]);

	return (abs(j - i));
}


/*! \brief appends indexes of signal for each peak and trough, for a given signal gradient.
 *         
 *
 *  Iterates through the signal gradient identifying changes between positive and negative
 *	gradients. Differences in sign of the two current and previous gradient indicate a 
 *	peak or a trough.
 *
 *	Appends the indexes of each peak and trough to the supplied vectors
 */	
void DistanceFunctions::find_signal_peaks_and_troughs(std::vector<double>& signal_gradient, std::vector<int>& peak_idx, std::vector<int>& trough_idx)
{
	double current_gradient;
	double previous_gradient;
	

	for (int i = 1; i <= signal_gradient.size(); i++) {
		// Set current and previous gradients
		current_gradient = signal_gradient[i];
		previous_gradient = signal_gradient[i-1];

		
		/* 
		A peak is when positive state precedes negative state.
		A trough is when negative state precedes positive state.
		*/
		if (current_gradient < 0 && previous_gradient > 0) {
			peak_idx.push_back(i);
		} else if (current_gradient > 0 && previous_gradient < 0) {
			trough_idx.push_back(i);
		} else {
			continue;
		}

	}
}

/*! \brief Calculates standard deviation of a signal.
 *        
 *	Calculates standard deviation of a signal. Be careful of sizes of numbers, might hit upper limits
 *	for some systems, possibly I should scale down the signal first?
 */	
double DistanceFunctions::standard_deviation(std::vector<double>& signal) {

	double mean = std::accumulate(signal.begin(), signal.end(), 0.0) / signal.size();

	auto sq_diff_mean = [&](double time_point, double mean){return (double) pow(time_point - mean, 2); };
	auto sq_sum = [&](std::vector<double>& sig, double& mean) {return (double) std::accumulate(sig.begin(), sig.end(), 
		mean, sq_diff_mean); };

	double stdev = sqrt(sq_sum(signal, mean) / (signal.size() -1) );

	return stdev;
}

bool DistanceFunctions::has_negative_species(std::vector<state_type>& state_vec) {
	for (auto tp_iter = state_vec.begin(); tp_iter != state_vec.end(); tp_iter++) {
		std::vector<double> sim_vec = *tp_iter;
		auto min_value = *std::min_element(sim_vec.begin(),sim_vec.end());

		if (min_value < 0) {
			return true;
		}
	}

	return false;
}


/*! \brief Calculates distances for stable objective. Returns vector of distances for each species
 *        
 *	Calculates standard deviation of a signal. Be careful of sizes of numbers, might hit upper limits
 *	for some systems, possibly I should scale down the signal first?
 */	
std::vector<std::vector<double>> DistanceFunctions::stable_dist(std::vector<state_type>& state_vec, std::vector<int> species_to_fit, bool integration_failed) {
	std::vector<std::vector<double>> sim_distances;
	double max_dist = std::numeric_limits<double>::max();

	if (integration_failed or has_negative_species(state_vec)) {
		for (auto it = species_to_fit.begin(); it != species_to_fit.end(); it++) {
			std::vector<double> max_distances = {max_dist, max_dist};
			sim_distances.push_back(max_distances);
		}
		return sim_distances;
	}


	// Iterate through all species to fit. Extract data.
	for (auto it = species_to_fit.begin(); it != species_to_fit.end(); it++) {
		std::vector<double> signal = extract_species_to_fit(state_vec, *it);

		std::vector<double> signal_gradient = get_signal_gradient(signal);

		double stdev = standard_deviation(signal);
		double final_gradient = abs(signal_gradient.end()[-1]);

		std::vector<double> signal_distances = {final_gradient, stdev};
		sim_distances.push_back(signal_distances);
	}

	return sim_distances;

}

