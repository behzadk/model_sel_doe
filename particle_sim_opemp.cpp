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
#include <boost/range/irange.hpp>
#include <boost/numeric/odeint/integrate/max_step_checker.hpp>
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

    simulation_observer( std::vector< state_type > &states )
    : m_states( states )  { }

    void operator()( const state_type &x , double t )
    {
        m_states.push_back( x );
    }
};



Particle::Particle(state_type init_state, std::vector <double> params, Models model_obj, int model_idx)
{
	Models m = model_obj;
	part_params = params;
    model_ref = model_idx;
    state_init = init_state;
}

void Particle::hello_world(){
	std::cout << "hello_world" << std::endl;
}


void Particle::operator() ( const state_type &y , state_type &dxdt , double t)
{
    // int model_ref = model_ref;
    std::vector<model_t> m_vec = m.models_vec;
    m.run_model(y, dxdt, t, part_params, model_ref);
}


void Particle::simulate_particle(double dt, std::vector<double> time_points) {

    // std::vector<double> time_array;

    // for(double i=t0; i <=t_end; i = i+dt){
    //     std::cout<<i <<std::endl;
    //     time_array.push_back(i);
    // }
    // make_controlled( abs err tol , relative err tol , maximum step, stepper_type() )
    // Stepper, System, init_state, start time, end time, time

    // integrate_n_steps( make_controlled( 1E-12 , 1E-12, dt, stepper_type() ) , 
    // 	boost::ref( *this ) , x , t0, 
    // 	dt, num_steps, simulation_observer(state_vec, times_array) );

    // auto range = boost::irange((int) t0, (int) t_end, dt);
    
    boost::numeric::odeint::max_step_checker mx_check =  boost::numeric::odeint::max_step_checker(1E6);
    

    integrate_times( make_controlled( 1E-3 , 1E-3, dt, stepper_type() ) , 
    boost::ref( *this ) , state_init , time_points.begin(), time_points.end(),
    dt, simulation_observer(state_vec), mx_check );

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


std::vector<state_type>& Particle::get_state_vec() {
    return state_vec;
}

void Particle::set_distance_vector(std::vector<std::vector<long>> sim_dist) {
    this->sim_distances = sim_dist;
}