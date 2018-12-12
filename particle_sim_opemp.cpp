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


Particle::Particle(state_type init_state, std::vector<double> params, Models model_obj, int model_idx)
{
	Models m = model_obj;
	part_params = params;
    model_ref = model_idx;
    state_init = init_state;
}

void Particle::hello_world(){
	std::cout << "hello_world" << std::endl;
}


void Particle::operator() ( const state_type &y , state_type &dxdt , double t) //Original
{
    // int model_ref = model_ref;
    std::vector<model_t> m_vec = m.models_vec;
    // m.run_model(y, dxdt, t, part_params, model_ref);
}

void Particle::operator() (const ublas_vec_t & y , ublas_vec_t &dxdt , double t ) // run_model_func
{
    // int model_ref = model_ref;
    std::vector<model_t> m_vec = m.models_vec;
    m.run_model_ublas(y, dxdt, t, part_params, model_ref);
}

void Particle::operator() (const ublas_vec_t & x , ublas_mat_t &J , const double & t , ublas_vec_t &dfdt ) // run_jac
{
    // int model_ref = model_ref;
    std::vector<model_t> m_vec = m.models_vec;
    spock_jac(x, J, t, dfdt);
    // rpr_jac(x, J, t, dfdt);
}


// New for ublas rosenbrock
void Particle::run_model_func(const ublas_vec_t & y , ublas_vec_t &dxdt , double t )
{
    std::vector<model_t> m_vec = m.models_vec;
    m.run_model_ublas(y, dxdt, t, part_params, model_ref);
}

void Particle::rpr_jac(const ublas_vec_t & y , ublas_mat_t &J , const double &/* t*/ , ublas_vec_t &dfdt)
{

    const double alpha0 = part_params[0];
    const double alpha_param = part_params[1];
    const double coeff = part_params[2];
    const double beta_param = part_params[3];

    J( 0 , 0 ) = -1;
    J( 0 , 1 ) = 0;
    J( 0 , 2 ) = 0;
    J( 0 , 3 ) = 0;
    J( 0 , 4 ) = 0;
    J( 0 , 5 ) = -alpha_param*coeff*std::pow(y[5], coeff)/(y[5]*std::pow(std::pow(y[5], coeff) + 1, 2));
    J( 1 , 0 ) = 0;
    J( 1 , 1 ) = -1;
    J( 1 , 2 ) = 0;
    J( 1 , 3 ) = -alpha_param*coeff*std::pow(y[3], coeff)/(y[3]*std::pow(std::pow(y[3], coeff) + 1, 2));
    J( 1 , 4 ) = 0;
    J( 1 , 5 ) = 0;
    J( 2 , 0 ) = 0;
    J( 2 , 1 ) = 0;
    J( 2 , 2 ) = -1;
    J( 2 , 3 ) = 0;
    J( 2 , 4 ) = -alpha_param*coeff*std::pow(y[4], coeff)/(y[4]*std::pow(std::pow(y[4], coeff) + 1, 2));
    J( 2 , 5 ) = 0;
    J( 3 , 0 ) = beta_param;
    J( 3 , 1 ) = 0;
    J( 3 , 2 ) = 0;
    J( 3 , 3 ) = -beta_param;
    J( 3 , 4 ) = 0;
    J( 3 , 5 ) = 0;
    J( 4 , 0 ) = 0;
    J( 4 , 1 ) = beta_param;
    J( 4 , 2 ) = 0;
    J( 4 , 3 ) = 0;
    J( 4 , 4 ) = -beta_param;
    J( 4 , 5 ) = 0;
    J( 5 , 0 ) = 0;
    J( 5 , 1 ) = 0;
    J( 5 , 2 ) = beta_param;
    J( 5 , 3 ) = 0;
    J( 5 , 4 ) = 0;
    J( 5 , 5 ) = -beta_param;

    dfdt[0] = 0.0;
    dfdt[1] = 0.0;
    dfdt[2] = 0.0;
    dfdt[3] = 0.0;
    dfdt[4] = 0.0;
    dfdt[5] = 0.0;

}

void Particle::spock_jac(const ublas_vec_t & y , ublas_mat_t &J , const double &/* t*/ , ublas_vec_t &dfdt)
{
    const double D = part_params[0];
    const double mux_m = part_params[1];
    const double muc_m = part_params[2];
    const double Kx = part_params[3];
    const double Kc = part_params[4];

    const double omega_c_max = part_params[5];
    const double K_omega = part_params[6];
    const double n_omega = part_params[7];
    const double S0 = part_params[8];

    const double gx = part_params[9];
    const double gc = part_params[10];
    const double C0L = part_params[11];
    const double KDL = part_params[12];

    const double nL = part_params[13];
    const double K1L = part_params[14]; 
    const double K2L = part_params[15];

    const double ymaxL = part_params[16];
    const double K1T = part_params[17];
    const double K2T = part_params[18];

    const double ymaxT = part_params[19];
    const double C0B = part_params[20];
    const double LB = part_params[21];
    const double NB = part_params[22];

    const double KDB = part_params[23];
    const double K1B = part_params[24];
    const double K2B = part_params[25];
    const double K3B = part_params[26];

    const double ymaxB = part_params[27];
    const double cgt = part_params[28];
    const double k_alpha_max = part_params[29];
    const double k_beta_max = part_params[30];

    //Death rate given by a hill function
    double omega_c = omega_c_max * pow(y[3], n_omega) / (pow(K_omega, n_omega) + pow(y[3], n_omega));

    //Growth rates of killer (x) and competitor (c)
    double mux = mux_m * y[2] / (Kx + y[2]);
    double muc = muc_m * y[2] / (Kc + y[2]);

    // Concentration of ligand bound to LuxR
    double CL = C0L * pow(y[4], nL) / (pow(KDL, nL) + pow(y[4], nL));

    // Probability of expression of TetR from the Plux promoter
    double pL = (K1L + K2L * CL) / (1 + K1L + K2L * CL);

    // Concentration of ligand free TetR
    double CFT = pL * ymaxL * cgt;

    // Probability of expression of bacteriocin from Ptet promoter
    double P_T = K1T/(1 + K1T + 2 * K2T * CFT + K2T * K2T * CFT * CFT);

    // Concentration of arabinose bound to AraC
    double CB = C0B*pow(LB, NB)/( pow(KDB,NB) + pow(LB,NB) );

    // Concentration of free AraC
    double CFB = C0B - CB;

    // Expression rate of AHL from araBAD promoter
    double P_B = (K1B + K2B*CB)/(1 + K1B + K2B*CB + K3B*CFB);

    // Rate of bacteriocin expression
    double k_beta = P_T * k_beta_max;

    // Rate of AHL expression
    double k_alpha = P_B * k_alpha_max;


    J( 0 , 0 ) = -D + y[2]*mux_m/(Kx + y[2]);
    J( 0 , 1 ) = 0;
    J( 0 , 2 ) = y[0]*(-y[2]*mux_m/std::pow(Kx + y[2], 2) + mux_m/(Kx + y[2]));
    J( 0 , 3 ) = 0;
    J( 0 , 4 ) = 0;
    J( 1 , 0 ) = 0;
    J( 1 , 1 ) = -std::pow(y[3], n_omega)*omega_c_max/(std::pow(y[3], n_omega) + std::pow(K_omega, n_omega)) - D + y[2]*muc_m/(Kc + y[2]);
    J( 1 , 2 ) = y[1]*(-y[2]*muc_m/std::pow(Kc + y[2], 2) + muc_m/(Kc + y[2]));
    J( 1 , 3 ) = y[1]*(std::pow(y[3], 2*n_omega)*n_omega*omega_c_max/(y[3]*std::pow(std::pow(y[3], n_omega) + std::pow(K_omega, n_omega), 2)) - std::pow(y[3], n_omega)*n_omega*omega_c_max/(y[3]*(std::pow(y[3], n_omega) + std::pow(K_omega, n_omega))));
    J( 1 , 4 ) = 0;
    J( 2 , 0 ) = -y[2]*mux_m/(gx*(Kx + y[2]));
    J( 2 , 1 ) = -y[2]*muc_m/(gc*(Kc + y[2]));
    J( 2 , 2 ) = -D + y[1]*y[2]*muc_m/(gc*std::pow(Kc + y[2], 2)) - y[1]*muc_m/(gc*(Kc + y[2])) + y[0]*y[2]*mux_m/(gx*std::pow(Kx + y[2], 2)) - y[0]*mux_m/(gx*(Kx + y[2]));
    J( 2 , 3 ) = 0;
    J( 2 , 4 ) = 0;
    J( 3 , 0 ) = P_T*k_beta_max;
    J( 3 , 1 ) = 0;
    J( 3 , 2 ) = 0;
    J( 3 , 3 ) = -D;
    J( 3 , 4 ) = 0;
    J( 4 , 0 ) = P_B*k_alpha_max;
    J( 4 , 1 ) = 0;
    J( 4 , 2 ) = 0;
    J( 4 , 3 ) = 0;
    J( 4 , 4 ) = -D;

    // dfdt[0] = 0.0;
    // dfdt[1] = 0.0;
    // dfdt[2] = 0.0;
    // dfdt[3] = 0.0;
    // dfdt[4] = 0.0;
}

void Particle::simulate_particle_rosenbrock(double dt, std::vector<double> time_points)
{
    int num_species = state_init.size();
    ublas_vec_t x(num_species);

    for(int i = 0; i < state_init.size(); i++) {
        x(i) = state_init[i];
    }
    boost::numeric::odeint::max_step_checker mx_check =  boost::numeric::odeint::max_step_checker(1000);

    std::cout << "doing rosenbrock" << std::endl;
    integrate_times( make_dense_output< rosenbrock4< double > >( 1.0e-6 , 1.0e-6 ) , 
        make_pair(boost::ref( *this ), boost::ref( *this )) , 
        x , time_points.begin(), time_points.end() , 1e-6, simulation_observer_ublas(state_vec), mx_check);

}
// New for ublas rosenbrock



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
    


    boost::numeric::odeint::max_step_checker mx_check = boost::numeric::odeint::max_step_checker(1000);

    integrate_times( make_controlled( 1E-6, 1E-6, dt, dopri5_stepper_type() ) , 
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

void Particle::set_distance_vector(std::vector<std::vector<double>> sim_dist) {
    this->sim_distances = sim_dist;
}