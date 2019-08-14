#include <array>
#include <iostream>
#include <vector>
#include <math.h>
#include "model.h"

Models::Models() {
	 models_ublas_vec = {&Models::model_0, &Models::model_1, &Models::model_2};
	 models_jac_vec = {&Models::jac_0, &Models::jac_1, &Models::jac_2};
};

void Models::run_model_ublas(const ublas_vec_t &y , ublas_vec_t &dxdt , double t, std::vector <double> &part_params, int &model_ref)
{
	(this->*models_ublas_vec[model_ref])(y, dxdt, t, part_params);
}

void Models::run_jac(const ublas_vec_t & x , ublas_mat_t &J , const double & t , ublas_vec_t &dfdt, std::vector <double> &part_params, int &model_ref)
{
	(this->*models_jac_vec[model_ref])(x, J, t, dfdt, part_params);
}

void Models::model_0(const ublas_vec_t  &y , ublas_vec_t &dydt , double t, std::vector <double> &part_params)
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
	const double kA_1 = part_params[8];
	const double kBmax_1 = part_params[9];
	const double kV_1 = part_params[10];
	const double kV_max_1 = part_params[11];
	const double mu_max_1 = part_params[12];
	const double nB_1 = part_params[13];
	const double nV_1 = part_params[14];
	const double n_omega_1 = part_params[15];
	const double omega_max_1 = part_params[16];

	//Species order is: N_1 S_glu B_1 A_1 V_1 
	dydt[0] = -std::pow(y[2], n_omega_1)*K_V_1*y[0]*omega_max_1/((std::pow(y[2], n_omega_1) + std::pow(K_omega_1, n_omega_1))*(K_V_1 + y[4])) - D*y[0] + y[0]*y[1]*mu_max_1/(K_mu_glu + y[1]);
	dydt[1] = -C*y[0]*y[1]*mu_max_1/(g_1*(K_mu_glu + y[1])) + D*(S0_glu - y[1]);
	dydt[2] = std::pow(y[3], nB_1)*C*y[0]*kBmax_1/(std::pow(y[3], nB_1) + std::pow(KB_1, nB_1)) - y[2]*D;
	dydt[3] = -y[3]*D + C*y[0]*kA_1;
	dydt[4] = C*y[0]*std::pow(kV_1, nV_1)*kV_max_1/(std::pow(y[3], nV_1) + std::pow(kV_1, nV_1)) - D*y[4];

}
void Models::jac_0(const ublas_vec_t & y , ublas_mat_t &J , const double &/* t*/ , ublas_vec_t &dfdt, std::vector <double> &part_params)
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
	const double kA_1 = part_params[8];
	const double kBmax_1 = part_params[9];
	const double kV_1 = part_params[10];
	const double kV_max_1 = part_params[11];
	const double mu_max_1 = part_params[12];
	const double nB_1 = part_params[13];
	const double nV_1 = part_params[14];
	const double n_omega_1 = part_params[15];
	const double omega_max_1 = part_params[16];

	J( 0 , 0 ) = -std::pow(y[2], n_omega_1)*K_V_1*omega_max_1/((std::pow(y[2], n_omega_1) + std::pow(K_omega_1, n_omega_1))*(K_V_1 + y[4])) - D + y[1]*mu_max_1/(K_mu_glu + y[1]);
	J( 0 , 1 ) = -y[0]*y[1]*mu_max_1/std::pow(K_mu_glu + y[1], 2) + y[0]*mu_max_1/(K_mu_glu + y[1]);
	J( 0 , 2 ) = std::pow(y[2], 2*n_omega_1)*K_V_1*y[0]*n_omega_1*omega_max_1/(y[2]*std::pow(std::pow(y[2], n_omega_1) + std::pow(K_omega_1, n_omega_1), 2)*(K_V_1 + y[4])) - std::pow(y[2], n_omega_1)*K_V_1*y[0]*n_omega_1*omega_max_1/(y[2]*(std::pow(y[2], n_omega_1) + std::pow(K_omega_1, n_omega_1))*(K_V_1 + y[4]));
	J( 0 , 3 ) = 0;
	J( 0 , 4 ) = std::pow(y[2], n_omega_1)*K_V_1*y[0]*omega_max_1/((std::pow(y[2], n_omega_1) + std::pow(K_omega_1, n_omega_1))*std::pow(K_V_1 + y[4], 2));
	J( 1 , 0 ) = -C*y[1]*mu_max_1/(g_1*(K_mu_glu + y[1]));
	J( 1 , 1 ) = C*y[0]*y[1]*mu_max_1/(g_1*std::pow(K_mu_glu + y[1], 2)) - C*y[0]*mu_max_1/(g_1*(K_mu_glu + y[1])) - D;
	J( 1 , 2 ) = 0;
	J( 1 , 3 ) = 0;
	J( 1 , 4 ) = 0;
	J( 2 , 0 ) = std::pow(y[3], nB_1)*C*kBmax_1/(std::pow(y[3], nB_1) + std::pow(KB_1, nB_1));
	J( 2 , 1 ) = 0;
	J( 2 , 2 ) = -D;
	J( 2 , 3 ) = -std::pow(y[3], 2*nB_1)*C*y[0]*kBmax_1*nB_1/(y[3]*std::pow(std::pow(y[3], nB_1) + std::pow(KB_1, nB_1), 2)) + std::pow(y[3], nB_1)*C*y[0]*kBmax_1*nB_1/(y[3]*(std::pow(y[3], nB_1) + std::pow(KB_1, nB_1)));
	J( 2 , 4 ) = 0;
	J( 3 , 0 ) = C*kA_1;
	J( 3 , 1 ) = 0;
	J( 3 , 2 ) = 0;
	J( 3 , 3 ) = -D;
	J( 3 , 4 ) = 0;
	J( 4 , 0 ) = C*std::pow(kV_1, nV_1)*kV_max_1/(std::pow(y[3], nV_1) + std::pow(kV_1, nV_1));
	J( 4 , 1 ) = 0;
	J( 4 , 2 ) = 0;
	J( 4 , 3 ) = -std::pow(y[3], nV_1)*C*y[0]*std::pow(kV_1, nV_1)*kV_max_1*nV_1/(y[3]*std::pow(std::pow(y[3], nV_1) + std::pow(kV_1, nV_1), 2));
	J( 4 , 4 ) = -D;

}
void Models::model_1(const ublas_vec_t  &y , ublas_vec_t &dydt , double t, std::vector <double> &part_params)
{
	//Unpack parameters
	const double C = part_params[0];
	const double D = part_params[1];
	const double KB_1 = part_params[2];
	const double K_mu_glu = part_params[3];
	const double K_omega_1 = part_params[4];
	const double S0_glu = part_params[5];
	const double g_1 = part_params[6];
	const double kA_1 = part_params[7];
	const double kBmax_1 = part_params[8];
	const double mu_max_1 = part_params[9];
	const double nB_1 = part_params[10];
	const double n_omega_1 = part_params[11];
	const double omega_max_1 = part_params[12];

	//Species order is: N_1 S_glu B_1 A_1 
	dydt[0] = -std::pow(y[2], n_omega_1)*y[0]*omega_max_1/(std::pow(y[2], n_omega_1) + std::pow(K_omega_1, n_omega_1)) - D*y[0] + y[0]*y[1]*mu_max_1/(K_mu_glu + y[1]);
	dydt[1] = -C*y[0]*y[1]*mu_max_1/(g_1*(K_mu_glu + y[1])) + D*(S0_glu - y[1]);
	dydt[2] = std::pow(y[3], nB_1)*C*y[0]*kBmax_1/(std::pow(y[3], nB_1) + std::pow(KB_1, nB_1)) - y[2]*D;
	dydt[3] = -y[3]*D + C*y[0]*kA_1;

}
void Models::jac_1(const ublas_vec_t & y , ublas_mat_t &J , const double &/* t*/ , ublas_vec_t &dfdt, std::vector <double> &part_params)
{
	//Unpack parameters
	const double C = part_params[0];
	const double D = part_params[1];
	const double KB_1 = part_params[2];
	const double K_mu_glu = part_params[3];
	const double K_omega_1 = part_params[4];
	const double S0_glu = part_params[5];
	const double g_1 = part_params[6];
	const double kA_1 = part_params[7];
	const double kBmax_1 = part_params[8];
	const double mu_max_1 = part_params[9];
	const double nB_1 = part_params[10];
	const double n_omega_1 = part_params[11];
	const double omega_max_1 = part_params[12];

	J( 0 , 0 ) = -std::pow(y[2], n_omega_1)*omega_max_1/(std::pow(y[2], n_omega_1) + std::pow(K_omega_1, n_omega_1)) - D + y[1]*mu_max_1/(K_mu_glu + y[1]);
	J( 0 , 1 ) = -y[0]*y[1]*mu_max_1/std::pow(K_mu_glu + y[1], 2) + y[0]*mu_max_1/(K_mu_glu + y[1]);
	J( 0 , 2 ) = std::pow(y[2], 2*n_omega_1)*y[0]*n_omega_1*omega_max_1/(y[2]*std::pow(std::pow(y[2], n_omega_1) + std::pow(K_omega_1, n_omega_1), 2)) - std::pow(y[2], n_omega_1)*y[0]*n_omega_1*omega_max_1/(y[2]*(std::pow(y[2], n_omega_1) + std::pow(K_omega_1, n_omega_1)));
	J( 0 , 3 ) = 0;
	J( 1 , 0 ) = -C*y[1]*mu_max_1/(g_1*(K_mu_glu + y[1]));
	J( 1 , 1 ) = C*y[0]*y[1]*mu_max_1/(g_1*std::pow(K_mu_glu + y[1], 2)) - C*y[0]*mu_max_1/(g_1*(K_mu_glu + y[1])) - D;
	J( 1 , 2 ) = 0;
	J( 1 , 3 ) = 0;
	J( 2 , 0 ) = std::pow(y[3], nB_1)*C*kBmax_1/(std::pow(y[3], nB_1) + std::pow(KB_1, nB_1));
	J( 2 , 1 ) = 0;
	J( 2 , 2 ) = -D;
	J( 2 , 3 ) = -std::pow(y[3], 2*nB_1)*C*y[0]*kBmax_1*nB_1/(y[3]*std::pow(std::pow(y[3], nB_1) + std::pow(KB_1, nB_1), 2)) + std::pow(y[3], nB_1)*C*y[0]*kBmax_1*nB_1/(y[3]*(std::pow(y[3], nB_1) + std::pow(KB_1, nB_1)));
	J( 3 , 0 ) = C*kA_1;
	J( 3 , 1 ) = 0;
	J( 3 , 2 ) = 0;
	J( 3 , 3 ) = -D;

}
void Models::model_2(const ublas_vec_t  &y , ublas_vec_t &dydt , double t, std::vector <double> &part_params)
{
	//Unpack parameters
	const double C = part_params[0];
	const double D = part_params[1];
	const double K_mu_glu = part_params[2];
	const double S0_glu = part_params[3];
	const double g_1 = part_params[4];
	const double mu_max_1 = part_params[5];

	//Species order is: N_1 S_glu 
	dydt[0] = -D*y[0] + y[0]*y[1]*mu_max_1/(K_mu_glu + y[1]);
	dydt[1] = -C*y[0]*y[1]*mu_max_1/(g_1*(K_mu_glu + y[1])) + D*(S0_glu - y[1]);

}
void Models::jac_2(const ublas_vec_t & y , ublas_mat_t &J , const double &/* t*/ , ublas_vec_t &dfdt, std::vector <double> &part_params)
{
	//Unpack parameters
	const double C = part_params[0];
	const double D = part_params[1];
	const double K_mu_glu = part_params[2];
	const double S0_glu = part_params[3];
	const double g_1 = part_params[4];
	const double mu_max_1 = part_params[5];

	J( 0 , 0 ) = -D + y[1]*mu_max_1/(K_mu_glu + y[1]);
	J( 0 , 1 ) = -y[0]*y[1]*mu_max_1/std::pow(K_mu_glu + y[1], 2) + y[0]*mu_max_1/(K_mu_glu + y[1]);
	J( 1 , 0 ) = -C*y[1]*mu_max_1/(g_1*(K_mu_glu + y[1]));
	J( 1 , 1 ) = C*y[0]*y[1]*mu_max_1/(g_1*std::pow(K_mu_glu + y[1], 2)) - C*y[0]*mu_max_1/(g_1*(K_mu_glu + y[1])) - D;

}
