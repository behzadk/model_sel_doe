#include <iostream>
#define BOOST_UBLAS_TYPE_CHECK 0

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
    std::vector< state_type >& m_states;
    size_t m_steps;
    rosenbrock4_controller<rosenbrock4<double> > m_stepper_controller;

    simulation_observer( std::vector< state_type > &states, rosenbrock4_controller<rosenbrock4<double> > &stepper_controller )
    : m_states( states ), m_stepper_controller( stepper_controller )  { }

    void operator()( const ublas_vec_t x , double t )
    {
        m_steps +=1;
        // std::cout << m_steps << std::endl;
        std::vector<double> new_state;

        for(int i = 0; i < x.size(); i++){
            double val = x(i);
            new_state.push_back(val);

            if(val < 0) throw runtime_error("Negative species value");

        }

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
    std::vector<model_t> m_vec = m.models_vec;
    m.run_model_ublas(y, dxdt, t, part_params, model_ref);
}

void Particle::operator() (const ublas_vec_t & x , ublas_mat_t &J , const double & t , ublas_vec_t &dfdt ) // run_jac
{
    m.run_jac(x, J, t, dfdt, part_params, model_ref);
}

/*
* Simulates particle for a given vector of time points. Currently uses default error tolerances and initial step of 
* 1e-6
*/
void Particle::simulate_particle_rosenbrock(std::vector<double> time_points)
{
    auto rosen_stepper = rosenbrock4_controller< rosenbrock4< double > >( 1.0e-6 , 1.0e-6 );

    // auto rosen_stepper = make_controlled< rosenbrock4< double > >( 1.0e-6 , 1.0e-6 );
    // PROBLEM: Solver is getting stuck trying to find smaller and smaller dt. Eventually reaching
    // 0 (maybe) and no longer calling observer at all. 
    // SOLUTION: Write custom stepper that checks dt and throws exception if dt reaches a very small number!
    // https://stackoverflow.com/questions/14465725/limit-number-of-steps-in-boostodeint-integration?noredirect=1&lq=1

    max_step_checker mx_step = max_step_checker(1e4);

    integrate_times(  rosen_stepper, 
        make_pair(boost::ref( *this ), boost::ref( *this )) , 
        state_init , time_points.begin(), time_points.end() , 1e-6, simulation_observer(state_vec, rosen_stepper), mx_step);

}

/*
 * Iterates the state vector, returning a flattened boost::python::list of the results.
 * requires reshaping  (timepoints, species)
 * 
 */
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