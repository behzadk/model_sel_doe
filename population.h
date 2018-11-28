// Header guard
#ifndef __POPULATION_H_INCLUDED__
#define __POPULATION_H_INCLUDED__
#include "particle_sim_opemp.h"


typedef std::vector<std::vector<std::vector<long>>> PopDistances;
class Population {
	public:
		Population(const int, const int, const int, 
			const float, boost::python::list, boost::python::list);
		std::vector< std::vector<double> > unpack_parameters(boost::python::list);
		std::vector<int> unpack_model_references(boost::python::list);
		void generate_particles();
		void simulate_particles();
		void calculate_particle_distances();
		void accumulate_distances();
		PopDistances get_population_distances() {return _all_distances;};
		
		boost::python::list get_flattened_distances_list();
		boost::python::list get_particle_state_list(int);

	private:
		int _n_sims;
		int  _t_0;
		int _t_end;
		float _dt;
		std::vector<int> _model_refs;
		PopDistances _all_distances;
		std::vector<std::vector<double>> _all_params;
		std::vector<Particle> _particle_vector;

};




#endif
