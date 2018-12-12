#include <iostream>
#include "model.h"

#include <boost/numeric/odeint.hpp>
#include <boost/python.hpp>
#include <boost/python/args.hpp>
#include <omp.h>
#include <boost/numeric/odeint/external/openmp/openmp.hpp>
#include "particle_sim_opemp.h"
#include <boost/python/numpy.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include "population.h"
#include "distances.h"


using namespace std;
using namespace boost::numeric::odeint;
using namespace boost::python;

/*
 * Population class is used to generate a population of particles
 * to be simulated. This class is used as an interface with python.
*/
Population::Population(const int n_sims, const int t_0, 
    const int t_end, const float dt, boost::python::list state_init_list,
    boost::python::list params_list, boost::python::list model_ref_list) {

	_n_sims = n_sims;
	_t_0 = t_0;
	_t_end = t_end;
	_dt = dt;

	// Fill time points array
    for(double i=_t_0; i <_t_end; i+=_dt){
        _time_array.push_back(i);
    }

	_all_params = unpack_parameters(params_list);
	_all_state_init = unpack_parameters(state_init_list);
	_model_refs = unpack_model_references(model_ref_list);
}


/*
 * Generates vector of particle objects with their parameters and reference
 * to the model which should be simulated.
*/
void Population::generate_particles(){
	Models m = Models();

	std::vector<Particle> particle_vector;
    for (int i=0; i < _n_sims; ++i) {
        _particle_vector.push_back(Particle(_all_state_init[i], _all_params[i], m, _model_refs[i]));
    }
}

/*
 * Performs simulation of all particles in the population
*/
void Population::simulate_particles() {
	#pragma omp parallel for schedule(runtime)
	for (int i=0; i < _n_sims; ++i) {
		try{ 
        // _particle_vector[i].simulate_particle( _dt, _time_array);
        _particle_vector[i].simulate_particle_rosenbrock( _dt, _time_array);

    	} catch (boost::exception_detail::clone_impl<boost::exception_detail::error_info_injector<boost::numeric::odeint::no_progress_error> >) {
    		std::cout <<"integration_failed" << std::endl;
    		_particle_vector[i].integration_failed = true;
    	}
	}

	std::cout << "particles simulated" << std::endl;
}


/*
 * Calculates the distances of all particles in the population.
*/
void Population::calculate_particle_distances(){
	std::vector<int> fit_species = {0, 1};
	DistanceFunctions dist = DistanceFunctions();

	// #pragma omp parallel for schedule(runtime)
    for (int i=0; i < _n_sims; ++i) {
    	_particle_vector[i].get_state_vec();
    	_particle_vector[i].set_distance_vector(dist.stable_dist( _particle_vector[i].get_state_vec(), fit_species, _particle_vector[i].integration_failed));
    }
}

/*
 * Extracts the distances from all particles into a member vector.
 */
void Population::accumulate_distances(){
	for (int i=0; i < _n_sims; ++i) {
		_all_distances.push_back( _particle_vector[i].get_sim_distances() );
	}
}


/*
 * Flattens all the particle distances into a list for output to python.
 */
boost::python::list Population::get_flattened_distances_list() {
	boost::python::list py_list_distances;

	for (int i=0; i < _all_distances.size(); ++i) { //Iter sim
		auto sim = _all_distances[i];
		for (int j=0; j<_all_distances[i].size(); ++j) { //Iter species
			auto species = sim[j];

			for(int k=0; k<species.size(); ++k ) {
				double dist_val = species[k];
				py_list_distances.append(dist_val);
			}
		}
	}

	return py_list_distances;
}


/*
 * Unpacks the nested list of parameters to C++ compatible vector of vectors.
 * Input vector contains a vector of parameters for each simulation.
 */
std::vector< std::vector<double> > Population::unpack_parameters(boost::python::list nested_parameters) {
    std::vector< std::vector<double> > all_params;
    for (int i = 0; i < _n_sims; ++i){
        std::vector<double> params_temp;
        boost::python::list temp_sim_params = boost::python::extract<boost::python::list>(nested_parameters[i]);
        for (int j = 0; j < len(temp_sim_params); ++j){
            params_temp.push_back( boost::python::extract<double>(temp_sim_params[j]));
        }
        all_params.push_back(params_temp);
    }
    return all_params;
}

/*
 * Unpacks the list of model references. Each element refers to the index of the model that
 * this particle should simulate.
 */
std::vector<int> Population::unpack_model_references(boost::python::list model_ref_list) {
	std::vector<int> model_ref_vec;
	for (int i = 0; i < _n_sims; ++i) {
		int ref = boost::python::extract<int>(model_ref_list[i]);
		model_ref_vec.push_back(ref);
	}

	return model_ref_vec;
}


/*
 * Returns the state vector of a specified particle, in the form of a python list.
 * List needs to be reshaped (#timepoints, #species)
 */
boost::python::list Population::get_particle_state_list(int particle_ref) {
	return(_particle_vector[particle_ref].get_state_pylist());
}

boost::python::list Population::get_timepoints_list() {
	boost::python::list tp_list;
	for (int i=0; i < _time_array.size(); ++i){
		double tp = _time_array[i];
		tp_list.append(tp);
	}
	return tp_list;
}


BOOST_PYTHON_MODULE(population_modules)
{
	class_<PopDistances>("pop_dist_vec")
		.def(boost::python::vector_indexing_suite<PopDistances>());

    class_<Population>("Population", init<const int, const int, 
    const int, const float, boost::python::list, boost::python::list, boost::python::list>())
    	.def("generate_particles", &Population::generate_particles)
    	.def("simulate_particles", &Population::simulate_particles)
    	.def("calculate_particle_distances", &Population::calculate_particle_distances)
    	.def("accumulate_distances", &Population::accumulate_distances)
    	.def("get_population_distances", &Population::get_population_distances)
    	.def("get_flattened_distances_list", &Population::get_flattened_distances_list)
    	.def("get_particle_state_list", &Population::get_particle_state_list)
    	.def("get_timepoints_list", &Population::get_timepoints_list)
        ;
}