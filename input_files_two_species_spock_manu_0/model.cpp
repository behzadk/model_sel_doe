#include <array>
#include <iostream>
#include <vector>
#include <math.h>
#include "model.h"

Models::Models() {
	 models_ublas_vec = {&Models::model_125};
	 models_jac_vec = {&Models::jac_125};
};

void Models::run_model_ublas(const ublas_vec_t &y , ublas_vec_t &dxdt , double t, std::vector <double> &part_params, int &model_ref)
{
	(this->*models_ublas_vec[model_ref])(y, dxdt, t, part_params);
}

void Models::run_jac(const ublas_vec_t & x , ublas_mat_t &J , const double & t , ublas_vec_t &dfdt, std::vector <double> &part_params, int &model_ref)
{
	(this->*models_jac_vec[model_ref])(x, J, t, dfdt, part_params);
}

void Models::model_125(const ublas_vec_t  &y , ublas_vec_t &dydt , double t, std::vector <double> &part_params)
{
	//Unpack parameters
	const double C = part_params[0];
	const double D = part_params[1];
	const double KB_1 = part_params[2];
	const double K_V_1 = part_params[3];
	const double K_mu_glu = part_params[4];
	const double K_omega_1 = part_params[5];
	const double S0_glu = part_params[6];
	const double g_1 = part_params[7];
	const double g_2 = part_params[8];
	const double kA_1 = part_params[9];
	const double kA_2 = part_params[10];
	const double kBmax_1 = part_params[11];
	const double kV_1 = part_params[12];
	const double kV_max_1 = part_params[13];
	const double mu_max_1 = part_params[14];
	const double mu_max_2 = part_params[15];
	const double nB_1 = part_params[16];
	const double nV_1 = part_params[17];
	const double n_omega_1 = part_params[18];
	const double omega_max_1 = part_params[19];

	//Species order is: N_1 N_2 S_glu B_1 A_2 A_1 V_1 
	dydt[0] = -D*y[0] + y[0]*y[2]*mu_max_1/(K_mu_glu + y[2]);
	dydt[1] = -std::pow(y[3], n_omega_1)*K_V_1*y[1]*omega_max_1/((std::pow(y[3], n_omega_1) + std::pow(K_omega_1, n_omega_1))*(K_V_1 + y[6])) - D*y[1] + y[1]*y[2]*mu_max_2/(K_mu_glu + y[2]);
	dydt[2] = -C*y[0]*y[2]*mu_max_1/(g_1*(K_mu_glu + y[2])) - C*y[1]*y[2]*mu_max_2/(g_2*(K_mu_glu + y[2])) + D*(S0_glu - y[2]);
	dydt[3] = -y[3]*D + C*std::pow(KB_1, nB_1)*y[0]*kBmax_1/(std::pow(y[4], nB_1) + std::pow(KB_1, nB_1));
	dydt[4] = -y[4]*D + C*y[0]*kA_2;
	dydt[5] = -y[5]*D + C*y[0]*kA_1;
	dydt[6] = C*y[0]*std::pow(kV_1, nV_1)*kV_max_1/(std::pow(y[5], nV_1) + std::pow(kV_1, nV_1)) - D*y[6];

}
void Models::jac_125(const ublas_vec_t & y , ublas_mat_t &J , const double &/* t*/ , ublas_vec_t &dfdt, std::vector <double> &part_params)
{
	//Unpack parameters
	const double C = part_params[0];
	const double D = part_params[1];
	const double KB_1 = part_params[2];
	const double K_V_1 = part_params[3];
	const double K_mu_glu = part_params[4];
	const double K_omega_1 = part_params[5];
	const double S0_glu = part_params[6];
	const double g_1 = part_params[7];
	const double g_2 = part_params[8];
	const double kA_1 = part_params[9];
	const double kA_2 = part_params[10];
	const double kBmax_1 = part_params[11];
	const double kV_1 = part_params[12];
	const double kV_max_1 = part_params[13];
	const double mu_max_1 = part_params[14];
	const double mu_max_2 = part_params[15];
	const double nB_1 = part_params[16];
	const double nV_1 = part_params[17];
	const double n_omega_1 = part_params[18];
	const double omega_max_1 = part_params[19];

	J( 0 , 0 ) = -D + y[2]*mu_max_1/(K_mu_glu + y[2]);
	J( 0 , 1 ) = 0;
	J( 0 , 2 ) = -y[0]*y[2]*mu_max_1/std::pow(K_mu_glu + y[2], 2) + y[0]*mu_max_1/(K_mu_glu + y[2]);
	J( 0 , 3 ) = 0;
	J( 0 , 4 ) = 0;
	J( 0 , 5 ) = 0;
	J( 0 , 6 ) = 0;
	J( 1 , 0 ) = 0;
	J( 1 , 1 ) = -std::pow(y[3], n_omega_1)*K_V_1*omega_max_1/((std::pow(y[3], n_omega_1) + std::pow(K_omega_1, n_omega_1))*(K_V_1 + y[6])) - D + y[2]*mu_max_2/(K_mu_glu + y[2]);
	J( 1 , 2 ) = -y[1]*y[2]*mu_max_2/std::pow(K_mu_glu + y[2], 2) + y[1]*mu_max_2/(K_mu_glu + y[2]);
	J( 1 , 3 ) = std::pow(y[3], 2*n_omega_1)*K_V_1*y[1]*n_omega_1*omega_max_1/(y[3]*std::pow(std::pow(y[3], n_omega_1) + std::pow(K_omega_1, n_omega_1), 2)*(K_V_1 + y[6])) - std::pow(y[3], n_omega_1)*K_V_1*y[1]*n_omega_1*omega_max_1/(y[3]*(std::pow(y[3], n_omega_1) + std::pow(K_omega_1, n_omega_1))*(K_V_1 + y[6]));
	J( 1 , 4 ) = 0;
	J( 1 , 5 ) = 0;
	J( 1 , 6 ) = std::pow(y[3], n_omega_1)*K_V_1*y[1]*omega_max_1/((std::pow(y[3], n_omega_1) + std::pow(K_omega_1, n_omega_1))*std::pow(K_V_1 + y[6], 2));
	J( 2 , 0 ) = -C*y[2]*mu_max_1/(g_1*(K_mu_glu + y[2]));
	J( 2 , 1 ) = -C*y[2]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	J( 2 , 2 ) = C*y[0]*y[2]*mu_max_1/(g_1*std::pow(K_mu_glu + y[2], 2)) - C*y[0]*mu_max_1/(g_1*(K_mu_glu + y[2])) + C*y[1]*y[2]*mu_max_2/(g_2*std::pow(K_mu_glu + y[2], 2)) - C*y[1]*mu_max_2/(g_2*(K_mu_glu + y[2])) - D;
	J( 2 , 3 ) = 0;
	J( 2 , 4 ) = 0;
	J( 2 , 5 ) = 0;
	J( 2 , 6 ) = 0;
	J( 3 , 0 ) = C*std::pow(KB_1, nB_1)*kBmax_1/(std::pow(y[4], nB_1) + std::pow(KB_1, nB_1));
	J( 3 , 1 ) = 0;
	J( 3 , 2 ) = 0;
	J( 3 , 3 ) = -D;
	J( 3 , 4 ) = -std::pow(y[4], nB_1)*C*std::pow(KB_1, nB_1)*y[0]*kBmax_1*nB_1/(y[4]*std::pow(std::pow(y[4], nB_1) + std::pow(KB_1, nB_1), 2));
	J( 3 , 5 ) = 0;
	J( 3 , 6 ) = 0;
	J( 4 , 0 ) = C*kA_2;
	J( 4 , 1 ) = 0;
	J( 4 , 2 ) = 0;
	J( 4 , 3 ) = 0;
	J( 4 , 4 ) = -D;
	J( 4 , 5 ) = 0;
	J( 4 , 6 ) = 0;
	J( 5 , 0 ) = C*kA_1;
	J( 5 , 1 ) = 0;
	J( 5 , 2 ) = 0;
	J( 5 , 3 ) = 0;
	J( 5 , 4 ) = 0;
	J( 5 , 5 ) = -D;
	J( 5 , 6 ) = 0;
	J( 6 , 0 ) = C*std::pow(kV_1, nV_1)*kV_max_1/(std::pow(y[5], nV_1) + std::pow(kV_1, nV_1));
	J( 6 , 1 ) = 0;
	J( 6 , 2 ) = 0;
	J( 6 , 3 ) = 0;
	J( 6 , 4 ) = 0;
	J( 6 , 5 ) = -std::pow(y[5], nV_1)*C*y[0]*std::pow(kV_1, nV_1)*kV_max_1*nV_1/(y[5]*std::pow(std::pow(y[5], nV_1) + std::pow(kV_1, nV_1), 2));
	J( 6 , 6 ) = -D;

	dfdt[0] = 0.0;
	dfdt[1] = 0.0;
	dfdt[2] = 0.0;
	dfdt[3] = 0.0;
	dfdt[4] = 0.0;
	dfdt[5] = 0.0;
	dfdt[6] = 0.0;

}
