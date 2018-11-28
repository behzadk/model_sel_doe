// Header guard
#ifndef __PARTICLE_SIM_OPEMP_H_INCLUDED__
#define __PARTICLE_SIM_OPEMP_H_INCLUDED__

// Forward declared dependencies (none)
#include <iostream>
#include <boost/numeric/odeint.hpp>
#include <boost/python.hpp>
#include <boost/python/args.hpp>
#include <boost/thread/thread.hpp>
#include <omp.h>
#include <boost/numeric/odeint/external/openmp/openmp.hpp>
#include "model.h"

// Include dependencies
using namespace std;
using namespace boost::numeric::odeint;
using namespace boost::python;

const int t_0 = 0;
const int t_end = 1000;
const float dt = 0.01;
const int t_length = (t_end / dt) + dt;

// The class
typedef std::vector<double> state_type;
typedef std::array<double, t_length> time_type;
typedef runge_kutta_dopri5<
          state_type , double ,
          state_type , double ,
          openmp_range_algebra
        > stepper_type;
typedef void (Models::*model_t)(const std::vector<double> &, std::vector<double> &, double, std::vector<double>&);


class Particle
{
	private:
		Models m;
		std::vector<std::vector<long>> sim_distances;

	public:
		// Constructor
		Particle(std::vector<double>, Models, int);

		model_t particle_model;
		int particle_model_int;

		int model_ref;

		std::vector <double> part_params;
		vector <state_type> state_vec;
		time_type times_array;

		void simulate_particle(double t0, double t_end, double dt);
		void set_distance_vector(std::vector<std::vector<long>>);
		std::vector<std::vector<long>>  get_sim_distances() {return sim_distances;};

		// changes the operator () to instead call
		// the object as if it were a function
		void operator() ( const state_type &, state_type &, double);
		void hello_world();

		std::vector<state_type>& get_state_vec();
		boost::python::list get_state_pylist();
		boost::python::list get_times_pylist();


		
};


// Converts a vector to boost::python::list
template <class T>
void vec_to_pylist(std::vector<T> vector) {
    typename std::vector<T>::iterator iter;
    boost::python::list list;
    cout << "here" <<endl;
    for (iter = vector.begin(); iter != vector.end(); ++iter) {
        list.append(iter);
        cout << *iter <<endl;
    }
    cout << "end" <<endl;
}


#endif
