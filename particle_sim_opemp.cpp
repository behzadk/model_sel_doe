#include <iostream>
#include "model.h"

#include <boost/numeric/odeint.hpp>
#include <omp.h>
#include <boost/numeric/odeint/external/openmp/openmp.hpp>
#include "particle_sim_opemp.h"
using namespace std;
using namespace boost::numeric::odeint;
using namespace boost::python;
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/numeric/odeint/integrate/max_step_checker.hpp>
#include "distances.h"

using namespace std;
using namespace boost::numeric::odeint;



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

struct simulation_observer_ublas
{
    std::vector< state_type >& m_states;

    simulation_observer_ublas( std::vector< state_type > &states )
    : m_states( states )  { }

    void operator()( const ublas_vec_t x , double t )
    {
        std::vector<double> new_state;
        for(int i = 0; i < x.size(); i++){
            double val = x(i);
            new_state.push_back(val);
        }

        // std::cout << t << std::endl;
        m_states.push_back( new_state );
    }
};


Particle::Particle(ublas_vec_t init_state, std::vector<double> params, Models model_obj, int model_idx)
{
	Models m = model_obj;
	part_params = params;
    model_ref = model_idx;
    state_init = init_state;
}



void Particle::operator() (const ublas_vec_t & y , ublas_vec_t &dxdt , double t ) // run_model_func
{
    // int model_ref = model_ref;
    std::vector<model_t> m_vec = m.models_vec;
    m.run_model_ublas(y, dxdt, t, part_params, model_ref);
}

void Particle::operator() (const ublas_vec_t & x , ublas_mat_t &J , const double & t , ublas_vec_t &dfdt ) // run_jac
{
    m.run_jac(x, J, t, dfdt, part_params, model_ref);
}


void Particle::simulate_particle_rosenbrock(double dt, std::vector<double> time_points)
{
    // int num_species = state_init.size();
    // ublas_vec_t x(num_species);

    // for(int i = 0; i < state_init.size(); i++) {
    //     x(i) = state_init[i];
    // }

    integrate_times( make_dense_output< rosenbrock4< double > >( 1.0e-6 , 1.0e-6 ) , 
        make_pair(boost::ref( *this ), boost::ref( *this )) , 
        state_init , time_points.begin(), time_points.end() , 1e-6, simulation_observer_ublas(state_vec));

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

void Particle::set_distance_vector(std::vector<std::vector<double>> sim_dist) {
    this->sim_distances = sim_dist;
}