#include <array>
#include <iostream>
#include <vector>
#include <math.h>
// #include "particle_sim_opemp.h"
#include "distances.h"
#include <algorithm>
#include <numeric>

extern "C" {
	#include "kissfft/kiss_fft.h"
	#include "kissfft/kiss_fftr.h"
}

DistanceFunctions::DistanceFunctions() {
}

std::vector<double> DistanceFunctions::arange(int start, int stop, float step) {
    std::vector<double> values;

    for (double value = start; value < stop; value += step) {
        values.push_back(value);
    }

    return values;
}


/*! \brief Return the Discrete Fourier Transform sample frequencies.
 *         
 *
 *  
 *	See np.fft.fftfreq
 *	
 */
void DistanceFunctions::fft_freq(double * results, int n, float step_size) {
    float val = 1.0 / (n * step_size);
    // double results[n];

    int N;
    if (n % 2 == 0) {
	    N = (n) / 2 - 1 + 1;
    }
    else {
	    N = (n-1)/2 + 1;
    }

    std::vector<double> p1 = arange(0, N, 1);
    
    for (int i=0; i <= N; i++) {
	    results[i] = p1[i] * val;
    }

    std::vector<double> p2 = arange(-N, 0, 1);
	for (int i=0; i < p2.size(); i++) {
	    results[N+i] = p2[i] * val;
    }
}

/*! \brief Returns the period frequency of a signal
 *         
 *
 *  
 *	
 */
double DistanceFunctions::get_period_frequency(std::vector<double>& signal, const float dt)
{
	int n_samples = signal.size();



  	kiss_fft_scalar signal_scalar[n_samples];
	for (int i = 0; i < n_samples; i++) {
		signal_scalar[i] = signal[i];
	}


  	kiss_fft_cpx out[(n_samples /2) + 1];
    double period;

  	kiss_fftr_cfg cfg;

  	if ((cfg = kiss_fftr_alloc(n_samples, 0/*is_inverse_fft*/, NULL, NULL)) != NULL) 
  	{
	    size_t x;

	    kiss_fftr(cfg, signal_scalar, out);
	    free(cfg);


	    double max_real_part = pow((out[1].r + out[1].i), 2);

	   	int max_arg = 1;
	    for (x = 2; x < (n_samples/2 ) + 1; x++)
	    {
	    	double mag  = pow((out[x].r + out[x].i), 2);
	    	if (mag > max_real_part) {
	    		if ( isinf(mag)) {
	    			continue;
	    		}
	    		else {
		    		max_real_part = mag;
				   	max_arg = x;
		    	}
	    	}
	    }

	    double freq_bins[n_samples];

	    fft_freq(freq_bins, n_samples, dt);

	   	double max_freq = 2 * freq_bins[max_arg];
	   	period = 1/max_freq;

   	}

    return period;
}

void DistanceFunctions::test_fft(float f, int amp, int t_end, float step_size) {
	std::cout << "starting fft... " << std::endl;
	// Make sine wave

	std::vector<double> time_vec = arange(0, t_end, step_size);
	int n_samples = time_vec.size();

  	kiss_fft_scalar sin_wav[n_samples];
  	kiss_fft_cpx out[(n_samples /2) + 1];


	for (int i = 0; i < time_vec.size(); i++) {
        double y = amp * cos(M_PI * f * time_vec[i]);
		sin_wav[i] = y;
	}

  	kiss_fftr_cfg cfg;
  	if ((cfg = kiss_fftr_alloc(n_samples, 0/*is_inverse_fft*/, NULL, NULL)) != NULL)
  	{
	    size_t x;

	    kiss_fftr(cfg, sin_wav, out);
	    free(cfg);

	    double max_real_part = pow((out[1].r + out[1].i), 2); 
	   	int max_arg = 1;
	    for (x = 2; x < (n_samples/2 ) + 1; x++)
	    {

	    	double mag  = pow((out[x].r + out[x].i), 2);
	    	if (mag > max_real_part) {
	    		if ( isinf(mag)) {
	    			continue;
	    		}
	    		else {
		    		max_real_part = mag;
    			   	max_arg = x;
		    	}

	    	}
	    }

	    double freq_bins[n_samples];
	    fft_freq(freq_bins, n_samples, step_size);
	    std::cout << n_samples << std::endl;
	    std::cout << "" << std::endl;

	    std::cout << "max real parts = "<< max_real_part << std::endl;
	    std::cout << "max arg = " << max_arg << std::endl;

	   	double max_freq = 2 * freq_bins[max_arg];
	   	double period = 1/max_freq;
   	    std::cout << "fre_bin  = "<<  freq_bins[max_arg] << std::endl;
   	    std::cout << "freq  = "<<  max_freq << std::endl;
   	    std::cout << "period  = "<<  period << std::endl;

	    std::cout << "" << std::endl;

	}

}


std::vector<double> DistanceFunctions::extract_species_to_fit(std::vector<state_type>& state_vec, int species_idx, int from_time_index=0)
{
	std::vector<double> species_val_vec;

	for (auto tp_iter = state_vec.begin() + from_time_index; tp_iter != state_vec.end(); tp_iter++) {
        state_type sim_vec = *tp_iter;
        species_val_vec.push_back(sim_vec[species_idx]);
    }

    return species_val_vec;
}


std::vector<double> DistanceFunctions::get_signal_gradient(std::vector<double>& signal)
{
	std::vector<double> signal_gradient;

	for (int i = 1; i < signal.size(); i++) {
		double grad = signal[i] - signal[i-1];
		signal_gradient.push_back(grad);
	}

	return signal_gradient;

}

long double DistanceFunctions::calculate_final_gradient(std::vector<double>& species_vals)
{
	double i = species_vals.end()[-2];
	double j = species_vals.end()[-1];

	long double grad = j - i;
	return grad;
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
	
	for (int i = 1; i < signal_gradient.size(); i++) {
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

/*! \brief Returns a vector of amplitudes, calculated from a given signal and indexes for the peaks and troughs.
 *        
 *	Calculates standard deviation of a signal. Be careful of sizes of numbers, might hit upper limits
 *	for some systems, possibly I should scale down the signal first?
 */	
std::vector<double> DistanceFunctions::get_amplitudes(std::vector<double>& signal, std::vector<int>& peak_idx, std::vector<int>& trough_idx) 
{
	int num_peaks = peak_idx.size();
	int num_troughs = trough_idx.size();

	int max_iterations = min(num_peaks, num_troughs);

	std::vector<double> amplitudes;

	for (int i = 0; i < max_iterations; i++)
	{	
		int peak_i = peak_idx[i];
		int trough_i = trough_idx[i];

		double amp = abs(signal[peak_i] - signal[trough_i]);
		amplitudes.push_back(amp);
	}

	return amplitudes;
}

/*! \brief Calculates standard deviation of a signal.
 *        
 *	Calculates standard deviation of a signal. Be careful of sizes of numbers, might hit upper limits
 *	for some systems, possibly I should scale down the signal first?
 */	
double DistanceFunctions::standard_deviation(std::vector<double>& signal) {
	double mean;
	double sum_signal = 0.0;

	for (int i = 0; i < signal.size(); i++) {
		double val = signal[i];
		sum_signal = sum_signal + signal[i];
	}

	mean = sum_signal / signal.size();

	double sq_sum = 0.0;

	// auto sq_diff_mean = [&](double time_point, double mean){return pow(time_point - mean, 2); };
	//sum square
	for (int i = 0; i < signal.size(); i++) {
		double val = std::pow( (signal[i] - mean) , 2);
		sq_sum += val;
	}

	// for (auto it = signal.begin(); it != signal.end(); it++) {
	// 	sq_sum += sq_diff_mean(*it, mean);
	// }

	// auto sq_sum = [&](std::vector<double>& sig, double& mean) {return  std::accumulate(sig.begin(), sig.end(), 
	// 	mean, sq_diff_mean); };

	if (sq_sum == 0){
		return 0;
	}
	double stdev = sqrt( sq_sum / (signal.size()) );


	return stdev;
}

bool DistanceFunctions::has_negative_species(std::vector<state_type>& state_vec) {
	for (auto tp_iter = state_vec.begin(); tp_iter != state_vec.end(); tp_iter++) {
		std::vector<double> sim_vec = *tp_iter;
		double min_value = *std::min_element(sim_vec.begin(),sim_vec.end());

		if (min_value < 0) {
			return true;
		}
	}

	return false;
}


double DistanceFunctions::get_sum_stdev(std::vector<state_type>& state_vec, int n_species, int from_time_index) {
	double sum_stdev = 0;

	for (int i = 0; i < n_species; i++) {
		std::vector<double> signal = extract_species_to_fit(state_vec, i, from_time_index);
		double stdev = standard_deviation(signal);
		sum_stdev = sum_stdev + stdev;
	}

	return sum_stdev;
}

long double DistanceFunctions::get_sum_grad(std::vector<state_type>& state_vec, int n_species) {

	long double sum_grad = 0;
	for (int i = 0; i < n_species; i++) {
		std::vector<double> signal = extract_species_to_fit(state_vec, n_species, 0);

		// std::vector<double> signal_gradient = get_signal_gradient(signal);

		long double final_gradient = calculate_final_gradient(signal);
		sum_grad = sum_grad + final_gradient;
	}

	return sum_grad;

}

boost::python::list DistanceFunctions::get_all_species_grads(std::vector<state_type>& state_vec, int n_species) {
	boost::python::list all_grads;

	for (int i = 0; i < n_species; i++) {
		std::vector<double> signal = extract_species_to_fit(state_vec, n_species, 0);

		std::vector<double> signal_gradient = get_signal_gradient(signal);

		long double final_gradient = calculate_final_gradient(signal);
		all_grads.append(final_gradient);
	}

	return all_grads;

}

/*! \brief Calculates distances for oscillatory objective. Returns vector of distances for each species
 *        
 *	Distances: Number of peaks, final peak amplitude and period frequency.
 *	
 */	
std::vector<std::vector<double>> DistanceFunctions::osc_dist(std::vector<state_type>& state_vec, std::vector<int> species_to_fit, bool integration_failed, const float dt) {
	std::vector<std::vector<double>> sim_distances;

	double max_dist = std::numeric_limits<double>::max();

	std::vector<double> max_distances = {max_dist, max_dist, max_dist};

	// If integration has failed, return all distances as maximum
	if (integration_failed) {
		for (auto it = species_to_fit.begin(); it != species_to_fit.end(); it++) {
			sim_distances.push_back(max_distances);
		}
		return sim_distances;
	}

	int from_time_index = 900;
	int amplitude_threshold = 1e3;

	for (auto it = species_to_fit.begin(); it != species_to_fit.end(); it++) {
		std::vector<double> signal = extract_species_to_fit(state_vec, *it, from_time_index);
		std::vector<double> signal_gradient = get_signal_gradient(signal);
		
		std::vector<int> peak_indexes;
		std::vector<int> trough_indexes;

		find_signal_peaks_and_troughs(signal_gradient, peak_indexes, trough_indexes);
		
		double signal_period_freq = get_period_frequency(signal, dt);

		std::vector<double> singal_amplitudes = get_amplitudes(signal, peak_indexes, trough_indexes);

		double threshold_amplitudes_count;
		double final_amplitude;

		// If no amplitudes, set threshold and final amplitude to zero
		if (singal_amplitudes.size() == 0) {
			threshold_amplitudes_count = 0;
			final_amplitude = 0;
		}
		else {
			double threshold_amplitudes_count = 0;

			// Count threshold amplitudes
			for (auto amp_it = singal_amplitudes.begin(); amp_it != singal_amplitudes.end(); amp_it++) {
				if (*amp_it > amplitude_threshold) {
					threshold_amplitudes_count = threshold_amplitudes_count + 1;
				}
			}

			// Set final amplitude
			double final_amplitude = singal_amplitudes[singal_amplitudes.size() - 1];
		}


		std::vector<double> signal_distances = {threshold_amplitudes_count, final_amplitude, signal_period_freq};

		sim_distances.push_back(signal_distances);

	}

	return sim_distances;
}


/*! \brief Calculates distances for stable objective. Returns vector of distances for each species
 *        
 *	Calculates standard deviation of a signal. Be careful of sizes of numbers, might hit upper limits
 *	for some systems, possibly I should scale down the signal first?
 */	
std::vector<std::vector<double>> DistanceFunctions::stable_dist(std::vector<state_type>& state_vec, std::vector<int> species_to_fit, bool integration_failed) {
	std::vector<std::vector<double>> sim_distances;

	double max_dist = std::numeric_limits<double>::max();

	std::vector<double> max_distances = {max_dist, max_dist, max_dist};

	if (integration_failed) {
		for (auto it = species_to_fit.begin(); it != species_to_fit.end(); it++) {
			sim_distances.push_back(max_distances);
		}

		return sim_distances;
	}


	int from_time_index = 900;
	for (auto it = species_to_fit.begin(); it != species_to_fit.end(); it++) {
		std::vector<double> signal = extract_species_to_fit(state_vec, *it, from_time_index);
	}

	// Iterate through all species to fit. Extract data.
	for (auto it = species_to_fit.begin(); it != species_to_fit.end(); it++) {
		std::vector<double> signal = extract_species_to_fit(state_vec, *it, from_time_index);

		std::vector<double> signal_gradient = get_signal_gradient(signal);

		double stdev = standard_deviation(signal);
		double final_gradient = fabs(signal_gradient.end()[-1]);
		double final_value = signal.end()[-1];

		std::vector<double> signal_distances = {final_gradient, stdev, final_value};

		sim_distances.push_back(signal_distances);
	}

	return sim_distances;

}
