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

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/lu.hpp>

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
    std::vector<int> m_fit_species;

    simulation_observer( std::vector< state_type > &states, rosenbrock4_controller<rosenbrock4<double> > &stepper_controller, std::vector<int> &fit_species )
    : m_states( states ), m_stepper_controller( stepper_controller ), m_fit_species(fit_species)  { }

    void operator()( ublas_vec_t x , double t )
    {
        m_steps +=1;
        std::vector<double> new_state;
        bool species_decayed = false;
        bool negative_species = false;

        for (auto it = m_fit_species.begin(); it != m_fit_species.end(); it++) {
            if(x(*it) < 1e-10) {
                species_decayed = true;
            }
        }

        // //Test
        // if(x(0) < 1e-100) {
        //     species_decayed = true;
        // }
        
        // //Test
        // if(x(1) < 1e-100) {
        //     species_decayed = true;
        // }


        for(int i = 0; i < x.size(); i++){
            double val = x(i);

            // if (val < 1e-200) {
            //     species_decayed = true;
            // }

            if(val < 0) {
                negative_species = true;
            };

            new_state.push_back(val);

        }
        m_states.push_back( new_state );

        if (species_decayed) throw runtime_error("species_decayed");
        if (negative_species) throw runtime_error("negative_species");

    }
};


Particle::Particle(ublas_vec_t init_state, std::vector<double> params, Models model_obj, int model_idx)
{
	Models m = model_obj;
	part_params = params;
    model_ref = model_idx;
    state_init = init_state;
}


/*
* Overloaded functions to either run jacobian or model, depending upon the inputs.
* 
*/
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
void Particle::simulate_particle_rosenbrock(std::vector<double> time_points, double abs_tol, double rel_tol, std::vector<int> fit_species)
{
    auto rosen_stepper = rosenbrock4_controller< rosenbrock4< double > >( abs_tol , rel_tol);

    // auto rosen_stepper = make_controlled< rosenbrock4< double > >( 1.0e-6 , 1.0e-6 );
    // PROBLEM: Solver is getting stuck trying to find smaller and smaller dt. Eventually reaching
    // 0 (maybe) and no longer calling observer at all. 
    // SOLUTION: Write custom stepper that checks dt and throws exception if dt reaches a very small number!
    // https://stackoverflow.com/questions/14465725/limit-number-of-steps-in-boostodeint-integration?noredirect=1&lq=1

    max_step_checker mx_step = max_step_checker(1e5);

    double dt = time_points[1] - time_points[0];

    // // Check if step size is negative
    if ( ( time_points[1] - time_points[0] ) < 0 ) {
        dt = dt * -1;
    }

    integrate_times(  rosen_stepper, 
        make_pair(boost::ref( *this ), boost::ref( *this )) , 
        state_init , time_points.begin(), time_points.end() , dt, simulation_observer(state_vec, rosen_stepper, fit_species), mx_step);
}

/*
* Exposes the model function to python. Takes an input of species values as a list and returns the dydt values as
* a python list
*/
boost::python::list Particle::py_model_func(boost::python::list input_y)
{
    
    int n_species = state_init.size();
    ublas_vec_t y(n_species);

    // Load python data into ublas vector
    for(int i = 0; i < n_species; i++)
    {
        double val = boost::python::extract<double>(input_y[i]);
        y(i) = val;
    }

    ublas_vec_t dydt(n_species);
    double t;

    // Simulate model
    std::vector<model_t> m_vec = m.models_vec;
    m.run_model_ublas(y, dydt, t, part_params, model_ref);

    // Convert dydt to boost python list
    boost::python::list output;
    for(int i = 0; i < n_species; i++)
    {
        output.append(dydt[i]);
    }

    return output;
}


// /*
// *   Not working
// *
// *
// */
// boost::python::list Particle::get_eigenvalues()
// {
//     int n_species = state_init.size();
//     std::cout << "n_species: " << n_species << std::endl;
//     ublas_vec_t y(n_species);
//     for (int i=0; i < n_species; i++) {
//         y(i) = state_vec.back()[i];
//     }
    
//     // Init matrix n_species x n_species
//     ublas_mat_t J (n_species, n_species);

//     // Not sure why this is necessary
//     ublas_vec_t dfdt(n_species);

//     // Dummy values
//     const double t = 0;

//     // Fill jacobian matrix
//     m.run_jac(y, J, t, dfdt, part_params, model_ref);

//     // Init array 
//     double data[n_species][n_species];

//     // Unpack ublas jac into standard array
//     for (int i = 0; i < n_species; i++) {
//         for (int j = 0; j < n_species; j++) {
//             double val = J(i, j);
//             // int pos = (i * n_species) + j;
//             data[i][j] = val;
//         }
//     }

//     boost::python::list output;

//     // Find eigenvalues and eigenvectors. Copy of example for non-symmmetric complex problem
//     // https://www.gnu.org/software/gsl/doc/html/eigen.html#examples
//     // gsl_matrix_view mat_view = gsl_matrix_view_array (data, n_species, n_species);
//     gsl_matrix *gsl_J = gsl_matrix_alloc(n_species, n_species);
//     gsl_vector_complex *eval = gsl_vector_complex_alloc (n_species);
    
//     gsl_eigen_nonsymm_workspace * w = gsl_eigen_nonsymm_alloc(n_species);
//     gsl_eigen_nonsymm_params(0, 0, w);
//     for(int i = 0; i < n_species; i++) {
//         for(int j = 0; j < n_species; j++) {
//             gsl_matrix_set(gsl_J, i, j, data[i][j]);
//         }
//     }

//     gsl_eigen_nonsymm(gsl_J, eval, w); /*diagonalize E which is M at t fixed*/

//     boost::python::list eigen_values;

//     for (int i = 0; i < n_species; i++)
//     {
//         boost::python::list eig;

//         gsl_complex eval_i = gsl_vector_complex_get(eval, i);

//         double real_val = GSL_REAL(eval_i);
//         double img_val = GSL_IMAG(eval_i);

//         eig.append(real_val);
//         eig.append(img_val);

//         eigen_values.append(eig);
//     }

//     output.append(eigen_values);
//     // output.append(eig_real_product);

//     return output;
// }

double Particle::get_trace()
{
    int n_species = state_init.size();

    ublas_vec_t y(n_species);
    for (int i=0; i < n_species; i++) {
        y(i) = state_vec.back()[i];
    }
    
    // Init matrix n_species x n_species
    ublas_mat_t J (n_species, n_species);

    // Dummy parameters
    const double t = 0;

    ublas_vec_t dfdt(n_species);

    // Fill jacobian matrix
    m.run_jac(y, J, t, dfdt, part_params, model_ref);

    double trace = 0;

    for (int i=0; i < n_species; i++) {
        trace = trace + J(i, i);
    }

    return trace;
}


boost::python::list Particle::get_jacobian(boost::python::list input_y)
{
    int n_species = state_init.size();
    
    ublas_vec_t y(n_species);
    for (int i=0; i < n_species; i++) {
        y(i) = boost::python::extract<double>(input_y[i]);
    }

    // Init matrix n_species x n_species
    ublas_mat_t J (n_species, n_species);

    // Not sure why this is necessary
    ublas_vec_t dfdt(n_species);

    // Dummy values
    const double t = 0;

    // Fill jacobian matrix
    m.run_jac(y, J, t, dfdt, part_params, model_ref);

    // Unpack ublas jac into python list
    boost::python::list py_J;
    for (int i = 0; i < n_species; i++) {
        for (int j = 0; j < n_species; j++) {
            double val = J(i, j);
            py_J.append(val);
        }
    }

    return py_J;

}

/*
* Returns the jacobian using the end state
*/
boost::python::list Particle::get_end_state_jacobian()
{
    int n_species = state_init.size();
    
    ublas_vec_t y(n_species);
    for (int i=0; i < n_species; i++) {
        y(i) = state_vec.back()[i];
    }
    

    // Init matrix n_species x n_species
    ublas_mat_t J (n_species, n_species);

    // Not sure why this is necessary
    ublas_vec_t dfdt(n_species);

    // Dummy values
    const double t = 0;

    // Fill jacobian matrix
    m.run_jac(y, J, t, dfdt, part_params, model_ref);


    // Unpack ublas jac into python list
    boost::python::list py_J;
    for (int i = 0; i < n_species; i++) {
        for (int j = 0; j < n_species; j++) {
            double val = J(i, j);
            py_J.append(val);
        }
    }

    return py_J;
}

/*
* Returns the jacobian using the initial state
*
*
*
*/
boost::python::list Particle::get_init_state_jacobian()
{
    int n_species = state_init.size();
    
    ublas_vec_t y(n_species);
    for (int i=0; i < n_species; i++) {
        y(i) = state_init[i];
    }
    

    // Init matrix n_species x n_species
    ublas_mat_t J (n_species, n_species);

    // Not sure why this is necessary
    ublas_vec_t dfdt(n_species);

    // Dummy values
    const double t = 0;

    // Fill jacobian matrix
    m.run_jac(y, J, t, dfdt, part_params, model_ref);


    // Unpack ublas jac into python list
    boost::python::list py_J;
    for (int i = 0; i < n_species; i++) {
        for (int j = 0; j < n_species; j++) {
            double val = J(i, j);
            py_J.append(val);
        }
    }

    return py_J;
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
           for(int i=0; i < sim_vec.size(); i++) {
                double val = sim_vec[i];
                temp_reslist.append(val);
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



/*
Code from here http://programmingexamples.net/wiki/CPP/Boost/Math/uBLAS/determinant
*/
int Particle::determinant_sign(const boost::numeric::ublas::permutation_matrix<std::size_t>& pm)
{
    int pm_sign=1;
    std::size_t size = pm.size();
    for (std::size_t i = 0; i < size; ++i)
        if (i != pm(i))
            pm_sign *= -1.0; // swap_rows would swap a pair of rows here, so we change sign
    return pm_sign;
}

// double Particle::get_determinant() {
//     int n_species = state_init.size();
    
//     ublas_vec_t y(n_species);
//     for (int i=0; i < n_species; i++) {
//         y(i) = state_vec.back()[i];
//         std::cout << (y(i)) << std::endl;
//     }
    
//     // Init matrix n_species x n_species
//     ublas_mat_t J (n_species, n_species);

//     // Not sure why this is necessary
//     ublas_vec_t dfdt(n_species);

//     // Dummy values
//     const double t = 0;

//     // Fill jacobian matrix
//     m.run_jac(y, J, t, dfdt, part_params, model_ref);

//     boost::numeric::ublas::permutation_matrix<std::size_t> pm(J.size1());
//     long double det = 1.0;
//     if( boost::numeric::ublas::lu_factorize(J, pm) ) {
//         det = 0.0;
//     } else {
//         for(int i = 0; i < J.size1(); i++){
//             det *= J(i,i); // multiply by elements on diagonal
//         }

//         det = det * determinant_sign( pm );
//     }

//     return det;
// }

boost::python::list Particle::get_final_species_values()
{
    int n_species = state_init.size();

    boost::python::list end_state;

    ublas_vec_t y(n_species);
    for (int i=0; i < n_species; i++) {
        end_state.append(state_vec.back()[i]);
    }

    return end_state;
}


// void Particle::laplace_expansion()
// {
//     int n_species = state_init.size();
    
//     ublas_vec_t y(n_species);
//     for (int i=0; i < n_species; i++) {
//         y(i) = state_vec.back()[i];
//     }
    
//     // Init matrix n_species x n_species
//     ublas_mat_t J (n_species, n_species);

//     // Not sure why this is necessary
//     ublas_vec_t dfdt(n_species);

//     // Dummy values
//     const double t = 0;

//     // Fill jacobian matrix
//     m.run_jac(y, J, t, dfdt, part_params, model_ref);


//     int i, j;

//     long double sol = 0;
//     // Iterate columns
// }

double Particle::get_sum_stdev(int from_time_point)
{
    DistanceFunctions dist = DistanceFunctions();

    int n_species = state_init.size();
    return dist.get_sum_stdev(state_vec, n_species, from_time_point);
}

// long double Particle::get_sum_grad()
// {
//     DistanceFunctions dist = DistanceFunctions();

//     int n_species = state_init.size();
//     return dist.get_sum_grad(state_vec, n_species);

// }

boost::python::list Particle::get_all_grads()
{

    DistanceFunctions dist = DistanceFunctions();
    int n_species = state_init.size();

    return dist.get_all_species_grads(state_vec, n_species);

}