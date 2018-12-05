// Header guard
#ifndef __DISTANCE_H_INCLUDED__
#define __DISTANCE_H_INCLUDED__
#include "particle_sim_opemp.h"

class DistanceFunctions {
	
	public:
		DistanceFunctions();
		std::vector<double> extract_species_to_fit(std::vector<state_type>&, int);
		double calculate_final_gradient(std::vector<double>&);
		std::vector<double> get_signal_gradient(std::vector<double> &);
		void find_signal_peaks_and_troughs(std::vector<double>&, std::vector<int>&, std::vector<int>&);
		double standard_deviation(std::vector<double>&);
		bool has_negative_species(std::vector<state_type>&);

		std::vector<std::vector<double>> stable_dist(std::vector<state_type>&, std::vector<int>, bool);

};


#endif
