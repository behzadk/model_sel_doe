#include <iostream>
#include "model.h"
#define BOOST_UBLAS_TYPE_CHECK 0

#include <boost/numeric/odeint.hpp>
#include <boost/python.hpp>
#include <boost/python/args.hpp>
#include <boost/python/numpy/ndarray.hpp>
#include <omp.h>
#include <boost/numeric/odeint/external/openmp/openmp.hpp>
#include "particle_sim_opemp.h"
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/extract.hpp>
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
    boost::python::list params_list, boost::python::list model_ref_list, boost::python::list py_fit_species)
{

	_n_sims = n_sims;
	_t_0 = t_0;
	_t_end = t_end;
	_dt = dt;
    // Forwards in time simulation
    if (_dt > 0) {
        for(double i=_t_0; i <_t_end; i+=_dt){
            _time_array.push_back(i);
        }
    }

    // Backwards in time simulation
    if (_dt < 0) {
        for(double i=_t_0; i >_t_end; i+=_dt) {
            _time_array.push_back(i);
        }
    }

	_all_params = unpack_parameters(params_list);
	_all_state_init = unpack_parameters_to_ublas(state_init_list);
	_model_refs = unpack_model_references(model_ref_list);

    for (int j = 0; j < len(py_fit_species); ++j) {
            fit_species.push_back( boost::python::extract<double>(py_fit_species[j]));
    }
}


/*
 * Generates vector of particle objects with their parameters and reference
 * to the model which should be simulated.
*/
void Population::generate_particles()
{
	Models m = Models();

	std::vector<Particle> particle_vector;
    for (int i=0; i < _n_sims; ++i) {
        _particle_vector.push_back(Particle(_all_state_init[i], _all_params[i], m, _model_refs[i]));
    }
}

/*
 * Performs simulation of all particles in the population
*/
void Population::simulate_particles()
{

	#pragma omp parallel for schedule(runtime)
	for (int i=0; i < _n_sims; ++i) {
		try { 
            _particle_vector[i].simulate_particle_rosenbrock(_time_array);

    	} catch (boost::exception_detail::clone_impl<boost::exception_detail::error_info_injector<boost::numeric::odeint::no_progress_error> >) {
            std::string error_string = "no_progress_error";

            // std::cout <<"integration_failed: no_progress_error" << std::endl;
            _particle_vector[i].integration_error = error_string;
    		_particle_vector[i].integration_failed = true;
    	} catch (boost::exception_detail::clone_impl<boost::exception_detail::error_info_injector<boost::numeric::odeint::step_adjustment_error> >){        
            std::string error_string = "step_adjustment_error";

            // std::cout <<"integration_failed: step_adjustment_error" << std::endl;
            _particle_vector[i].integration_error = error_string;
            _particle_vector[i].integration_failed = true;
        } catch (boost::numeric::ublas::internal_logic){
            std::string error_string = "ublas_internal_logic_error";

            // std::cout <<"integration_failed: ublas internal_logic" << std::endl;
            _particle_vector[i].integration_error = error_string;
            _particle_vector[i].integration_failed = true;
        } catch (std::runtime_error& e ){
            std::string error_string = e.what();
            _particle_vector[i].integration_error = error_string;
            _particle_vector[i].integration_failed = true;
        }
	}
}


/*
 * Calculates the distances of all particles in the population.
*/
void Population::calculate_particle_distances()
{
	DistanceFunctions dist = DistanceFunctions();

	#pragma omp parallel for schedule(runtime)
    for (int i=0; i < _n_sims; ++i) {
    	_particle_vector[i].get_state_vec();
    	_particle_vector[i].set_distance_vector(dist.stable_dist( _particle_vector[i].get_state_vec(), fit_species, _particle_vector[i].integration_failed));
    }
}


/*
 * Extracts the distances from all particles into a member vector.
 */
void Population::accumulate_distances()
{
	for (int i=0; i < _n_sims; ++i) {
		_all_distances.push_back( _particle_vector[i].get_sim_distances() );
	}
}


/*
 * Flattens all the particle distances into a list for output to python.
 */
boost::python::list Population::get_flattened_distances_list() 
{
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
std::vector< std::vector<double> > Population::unpack_parameters(boost::python::list nested_parameters) 
{
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
 * Unpacks the nested pylist of parameters to vector of ublas vectors
 * 
 */
std::vector< ublas_vec_t > Population::unpack_parameters_to_ublas(boost::python::list nested_parameters)
{
    std::vector< ublas_vec_t > all_params;
    for (int i = 0; i < _n_sims; ++i){
    	boost::python::list temp_sim_params = boost::python::extract<boost::python::list>(nested_parameters[i]);

    	int num_species = boost::python::len(temp_sim_params);
    	ublas_vec_t params_temp(num_species);

        for (int j = 0; j < len(temp_sim_params); ++j){
            params_temp(j) = ( boost::python::extract<double>(temp_sim_params[j]));
        }
        all_params.push_back(params_temp);
    }
    return all_params;
}


/*
 * Unpacks the list of model references. Each element refers to the index of the model that
 * this particle should simulate.
 */
std::vector<int> Population::unpack_model_references(boost::python::list model_ref_list) 
{
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
boost::python::list Population::get_particle_state_list(int particle_ref) 
{
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

boost::python::list Population::get_particle_eigenvalues(int particle_ref)
{
    return(_particle_vector[particle_ref].get_eigenvalues_eigen());
}

double Population::get_particle_trace(int particle_ref)
{
    return(_particle_vector[particle_ref].get_trace());
}

boost::python::list Population::get_particle_init_state_jacobian(int particle_ref)
{
    return(_particle_vector[particle_ref].get_init_state_jacobian());
}

boost::python::list Population::get_particle_jacobian(boost::python::list py_state, int particle_ref)
{
    return(_particle_vector[particle_ref].get_jacobian(py_state));
}


boost::python::list Population::get_particle_end_state_jacobian(int particle_ref)
{
    return(_particle_vector[particle_ref].get_end_state_jacobian());
}

boost::python::list Population::get_particle_final_species_values(int particle_ref)
{
    return(_particle_vector[particle_ref].get_final_species_values());
}

double Population::get_particle_det(int particle_ref)
{
    return(_particle_vector[particle_ref].get_determinant());
}

void Population::get_particle_laplace_expansion(int particle_ref)
{
    return(_particle_vector[particle_ref].laplace_expansion());
}


bool Population::check_integration_failure(int particle_ref) 
{
    return _particle_vector[particle_ref].integration_failed;
}

/*
* Returns string containing all the integration errrs
*
*/
boost::python::list Population::get_all_particle_integration_errors()
{
    boost::python::list integ_errors;

    for (int i = 0; i < _n_sims; i++)
    {
        integ_errors.append(_particle_vector[i].integration_error);

    }

    return integ_errors;
}

std::string Population::get_particle_integration_error(int particle_ref)
{
    return _particle_vector[particle_ref].integration_error;
}

boost::python::list Population::py_model_func(boost::python::list input_y, int particle_ref)
{
    return _particle_vector[particle_ref].py_model_func(input_y);
}


double Population::get_particle_sum_stdev(int particle_ref, int from_time_point)
{
    return _particle_vector[particle_ref].get_sum_stdev(from_time_point);
}

long double Population::get_particle_sum_grad(int particle_ref)
{
    return _particle_vector[particle_ref].get_sum_grad();
}

boost::python::list Population::get_particle_grads(int particle_ref)
{
    return _particle_vector[particle_ref].get_all_grads();
}

BOOST_PYTHON_MODULE(population_modules)
{
	class_<PopDistances>("pop_dist_vec")
		.def(boost::python::vector_indexing_suite<PopDistances>());

    class_<Population>("Population", init<const int, const int, 
    const int, const float, boost::python::list, boost::python::list, boost::python::list, boost::python::list>())
    	.def("generate_particles", &Population::generate_particles)
    	.def("simulate_particles", &Population::simulate_particles)
    	.def("calculate_particle_distances", &Population::calculate_particle_distances)
    	.def("accumulate_distances", &Population::accumulate_distances)
    	.def("get_population_distances", &Population::get_population_distances)
    	.def("get_flattened_distances_list", &Population::get_flattened_distances_list)
    	.def("get_particle_state_list", &Population::get_particle_state_list)
    	.def("get_timepoints_list", &Population::get_timepoints_list)
        .def("check_integration_failure", &Population::check_integration_failure)
        .def("get_particle_integration_error", &Population::get_particle_integration_error)
        .def("get_all_particle_integration_errors", &Population::get_all_particle_integration_errors)
        .def("get_particle_eigenvalues", &Population::get_particle_eigenvalues)
        .def("get_particle_trace", &Population::get_particle_trace)
        .def("get_particle_init_state_jacobian", &Population::get_particle_init_state_jacobian)
        .def("get_particle_end_state_jacobian", &Population::get_particle_end_state_jacobian)
        .def("get_particle_det", &Population::get_particle_det)
        .def("get_particle_laplace_expansion", &Population::get_particle_laplace_expansion)
        .def("py_model_func", &Population::py_model_func)
        .def("get_particle_final_species_values", &Population::get_particle_final_species_values)
        .def("get_particle_jacobian", &Population::get_particle_jacobian)
        .def("get_particle_sum_stdev", &Population::get_particle_sum_stdev)
        .def("get_particle_sum_grad", &Population::get_particle_sum_grad)
        .def("get_particle_grads", &Population::get_particle_grads)
        ;
}
