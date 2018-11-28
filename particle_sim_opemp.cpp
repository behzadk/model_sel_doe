#include <iostream>
#include "model.h"

#include <boost/numeric/odeint.hpp>
#include <boost/python.hpp>
#include <boost/python/args.hpp>
#include <boost/thread/thread.hpp>
#include <omp.h>
#include <boost/numeric/odeint/external/openmp/openmp.hpp>
#include "particle_sim_opemp.h"
#include <openacc.h>
using namespace std;
using namespace boost::numeric::odeint;
using namespace boost::python;
#include <boost/python/numpy.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <typeinfo>
#include "distances.h"

class ScopedGILRelease
{
public:
    inline ScopedGILRelease(){
        m_thread_state = PyEval_SaveThread();
    }
    inline ~ScopedGILRelease() {
        PyEval_RestoreThread(m_thread_state);
        m_thread_state = NULL;
    }
private:
    PyThreadState * m_thread_state;
};


struct simulation_observer 
{
    vector< state_type >& m_states;
    time_type * m_times;
    int pos = 0;

    simulation_observer( std::vector< state_type > &states , time_type &times )
    : m_states( states ) , m_times( &times ) { }

    void operator()( const state_type &x , double t )
    {
        m_states.push_back( x );
        (*m_times)[pos] = t;
        pos = pos + 1;
    }
};


// struct res_struct
// {
//     boost::python::list res_list;

//     void add_res_pylist(boost::python::list sim_res)
//     {
//         res_list.append(sim_res);
//     }
//     void add_res_vector(boost::python::list sim_res)
//     {
//         res_list.append(sim_res);
//     }

// };


Particle::Particle(std::vector <double> params, Models model_obj, int model_ref)
{
	// Place holder
	Models m = model_obj;
	part_params = params;
    model_ref = model_ref;
}

void Particle::hello_world(){
	std::cout << "hello_world" << std::endl;
}


void Particle::operator() ( const state_type &y , state_type &dxdt , double t)
{
    int model_ref = model_ref;
    std::vector<model_t> m_vec = m.models_vec;
    m.run_model(y, dxdt, t, part_params);
}


void Particle::simulate_particle(double t0, double t_end, double dt) {
    state_type x = {0, 0, 0, 1.0, 2.0, 5.0};
    // make_controlled( abs err tol , relative err tol , maximum step, stepper_type() )
    // Stepper, System, init_state, start time, end time, time
    integrate_const( make_controlled( 1E-8 , 1E-8, 0.01, stepper_type() ) , 
    	boost::ref( *this ) , x , t0, 
    	t_end, dt, simulation_observer(state_vec, times_array) );
}

boost::python::list Particle::get_state_pylist() {
    boost::python::list temp_reslist;
    {
        ScopedGILRelease noGil = ScopedGILRelease();
        for (auto sim_iter = state_vec.begin(); sim_iter != state_vec.end(); ++sim_iter) {
            auto sim_vec = *sim_iter;
            BOOST_FOREACH(uint64_t n, sim_vec) {
                temp_reslist.append(n);
            }
        }

    }
    return temp_reslist;
}

boost::python::list Particle::get_times_pylist() {
    boost::python::list temp_reslist;

    {
        ScopedGILRelease noGil = ScopedGILRelease();
        const std::size_t arr_length = sizeof(times_array) / sizeof(double);

        for (int i =0; i < arr_length; i++) {
            double temp_t  = times_array[i];
            temp_reslist.append(temp_t);
        }
    }
    return temp_reslist;
}

std::vector<state_type>& Particle::get_state_vec() {
    return state_vec;
}

void Particle::set_distance_vector(std::vector<std::vector<long>> sim_dist) {
    this->sim_distances = sim_dist;
}




// std::vector< std::vector<double> > unpack_parameters(int n_sims, boost::python::list nested_parameters) {
//     std::vector< std::vector<double> > all_params;
//     for (int i = 0; i < n_sims; ++i){
//         std::vector<double> params_temp;
//         boost::python::list temp_sim_params = boost::python::extract<boost::python::list>(nested_parameters[i]);
//         for (int j = 0; j < len(temp_sim_params); ++j){
//             params_temp.push_back( boost::python::extract<double>(temp_sim_params[j]));
//         }
//         all_params.push_back(params_temp);
//     }
//     return all_params;
// }



// boost::python::tuple test_openmp(const int n_sims, const int t_0, 
//     const int t_end, const float dt, 
//     boost::python::list params_list)
// {
//     // Species to fit,
//     std::vector<int> fit_species = {3, 4, 5};

//     // Unpack parameters
//     std::vector<std::vector<double>> all_params = unpack_parameters(n_sims, params_list);
//     std::cout << all_params.size() << std::endl;

//     // Initiate model class
//     Models m = Models();

//     // Generate particles
//     std::vector<Particle> particle_vector;
//     int model_ref = 0;
//     for (int i=0; i<n_sims; ++i) {
//         std::cout << i << std::endl;
//         particle_vector.push_back(Particle(all_params[i], m, model_ref));
//     }


//     // Initialise distance class
//     DistanceFunctions dist = DistanceFunctions();

//     // Simulate particles and calculate distance
//     #pragma omp parallel for schedule(runtime)
//     for (int i=0; i < n_sims; ++i) {
//         particle_vector[i].simulate_particle(t_0, t_end, dt);
//         particle_vector[i].set_distance_vector(dist.stable_dist( particle_vector[i].get_state_vec(), fit_species ));
//     }
    

//     boost::python::list state_res_list;
//     boost::python::list time_res_list;

//     for (int i=0; i < n_sims; ++i) {
//         state_res_list.append(particle_vector[i].get_state_pylist());
//         time_res_list.append(particle_vector[i].get_times_pylist());
//     }

//     return boost::python::make_tuple(time_res_list, state_res_list);
// }

// BOOST_PYTHON_MODULE(particle_sim)
// {
//     boost::python::def("p_sim", &test_openmp, (boost::python::arg("n_sims"), 
//         boost::python::arg("t_0"), boost::python::arg("t_end"),
//         boost::python::arg("dt"), boost::python::arg("params_list")) );
// }

// https://stackoverflow.com/questions/6157409/stdvector-to-boostpythonlist