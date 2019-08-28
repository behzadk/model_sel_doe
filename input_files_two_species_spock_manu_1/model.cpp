#include <array>
#include <iostream>
#include <vector>
#include <math.h>
#include "model.h"

Models::Models() {
	 models_ublas_vec = {&Models::model_0, &Models::model_1, &Models::model_2, &Models::model_3, &Models::model_4, &Models::model_5, &Models::model_6, &Models::model_7, &Models::model_8, &Models::model_9, &Models::model_10, &Models::model_11, &Models::model_12, &Models::model_13, &Models::model_14, &Models::model_15, &Models::model_16, &Models::model_17, &Models::model_18, &Models::model_19, &Models::model_20, &Models::model_21, &Models::model_22, &Models::model_23, &Models::model_24, &Models::model_25, &Models::model_26, &Models::model_27, &Models::model_28, &Models::model_29, &Models::model_30, &Models::model_31, &Models::model_32, &Models::model_33, &Models::model_34, &Models::model_35, &Models::model_36, &Models::model_37, &Models::model_38};
	 models_jac_vec = {&Models::jac_0, &Models::jac_1, &Models::jac_2, &Models::jac_3, &Models::jac_4, &Models::jac_5, &Models::jac_6, &Models::jac_7, &Models::jac_8, &Models::jac_9, &Models::jac_10, &Models::jac_11, &Models::jac_12, &Models::jac_13, &Models::jac_14, &Models::jac_15, &Models::jac_16, &Models::jac_17, &Models::jac_18, &Models::jac_19, &Models::jac_20, &Models::jac_21, &Models::jac_22, &Models::jac_23, &Models::jac_24, &Models::jac_25, &Models::jac_26, &Models::jac_27, &Models::jac_28, &Models::jac_29, &Models::jac_30, &Models::jac_31, &Models::jac_32, &Models::jac_33, &Models::jac_34, &Models::jac_35, &Models::jac_36, &Models::jac_37, &Models::jac_38};
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
	const double D = part_params[0];
	const double g_1 = part_params[1];
	const double g_2 = part_params[2];
	const double K_A_B_1 = part_params[3];
	const double K_A_I_1 = part_params[4];
	const double k_I_1 = part_params[5];
	const double K_mu_glu = part_params[6];
	const double k_omega_B_1 = part_params[7];
	const double kA_1 = part_params[8];
	const double kB_max_1 = part_params[9];
	const double kI_max_1 = part_params[10];
	const double mu_max_1 = part_params[11];
	const double mu_max_2 = part_params[12];
	const double n_A_B_1 = part_params[13];
	const double n_A_I_1 = part_params[14];
	const double nI_1 = part_params[15];
	const double S0_glu = part_params[16];

	//Species order is: N_1 N_2 S_glu B_1 A_1 I_1 
	dydt[0] = -y[3]*y[0]*std::pow(k_I_1, nI_1)*k_omega_B_1/(std::pow(y[5], nI_1) + std::pow(k_I_1, nI_1)) - D*y[0] + y[0]*y[2]*mu_max_1/(K_mu_glu + y[2]);
	dydt[1] = -y[3]*y[1]*k_omega_B_1 - D*y[1] + y[1]*y[2]*mu_max_2/(K_mu_glu + y[2]);
	dydt[2] = D*(S0_glu - y[2]) - y[0]*y[2]*mu_max_1/(g_1*(K_mu_glu + y[2])) - y[1]*y[2]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	dydt[3] = std::pow(y[4], n_A_B_1)*y[0]*kB_max_1/(std::pow(y[4], n_A_B_1) + std::pow(K_A_B_1, n_A_B_1)) - y[3]*D;
	dydt[4] = -y[4]*D + y[0]*kA_1;
	dydt[5] = std::pow(y[4], n_A_I_1)*kI_max_1/(std::pow(y[4], n_A_I_1) + std::pow(K_A_I_1, n_A_I_1)) - 1.0/2.0*y[5]*y[2]*mu_max_1/(K_mu_glu + y[2]);

}
void Models::jac_0(const ublas_vec_t & y , ublas_mat_t &J , const double &/* t*/ , ublas_vec_t &dfdt, std::vector <double> &part_params)
{
	//Unpack parameters
	const double D = part_params[0];
	const double g_1 = part_params[1];
	const double g_2 = part_params[2];
	const double K_A_B_1 = part_params[3];
	const double K_A_I_1 = part_params[4];
	const double k_I_1 = part_params[5];
	const double K_mu_glu = part_params[6];
	const double k_omega_B_1 = part_params[7];
	const double kA_1 = part_params[8];
	const double kB_max_1 = part_params[9];
	const double kI_max_1 = part_params[10];
	const double mu_max_1 = part_params[11];
	const double mu_max_2 = part_params[12];
	const double n_A_B_1 = part_params[13];
	const double n_A_I_1 = part_params[14];
	const double nI_1 = part_params[15];
	const double S0_glu = part_params[16];

	J( 0 , 0 ) = -y[3]*std::pow(k_I_1, nI_1)*k_omega_B_1/(std::pow(y[5], nI_1) + std::pow(k_I_1, nI_1)) - D + y[2]*mu_max_1/(K_mu_glu + y[2]);
	J( 0 , 1 ) = 0;
	J( 0 , 2 ) = -y[0]*y[2]*mu_max_1/std::pow(K_mu_glu + y[2], 2) + y[0]*mu_max_1/(K_mu_glu + y[2]);
	J( 0 , 3 ) = -y[0]*std::pow(k_I_1, nI_1)*k_omega_B_1/(std::pow(y[5], nI_1) + std::pow(k_I_1, nI_1));
	J( 0 , 4 ) = 0;
	J( 0 , 5 ) = y[3]*std::pow(y[5], nI_1)*y[0]*std::pow(k_I_1, nI_1)*k_omega_B_1*nI_1/(y[5]*std::pow(std::pow(y[5], nI_1) + std::pow(k_I_1, nI_1), 2));
	J( 1 , 0 ) = 0;
	J( 1 , 1 ) = -y[3]*k_omega_B_1 - D + y[2]*mu_max_2/(K_mu_glu + y[2]);
	J( 1 , 2 ) = -y[1]*y[2]*mu_max_2/std::pow(K_mu_glu + y[2], 2) + y[1]*mu_max_2/(K_mu_glu + y[2]);
	J( 1 , 3 ) = -y[1]*k_omega_B_1;
	J( 1 , 4 ) = 0;
	J( 1 , 5 ) = 0;
	J( 2 , 0 ) = -y[2]*mu_max_1/(g_1*(K_mu_glu + y[2]));
	J( 2 , 1 ) = -y[2]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	J( 2 , 2 ) = -D + y[0]*y[2]*mu_max_1/(g_1*std::pow(K_mu_glu + y[2], 2)) - y[0]*mu_max_1/(g_1*(K_mu_glu + y[2])) + y[1]*y[2]*mu_max_2/(g_2*std::pow(K_mu_glu + y[2], 2)) - y[1]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	J( 2 , 3 ) = 0;
	J( 2 , 4 ) = 0;
	J( 2 , 5 ) = 0;
	J( 3 , 0 ) = std::pow(y[4], n_A_B_1)*kB_max_1/(std::pow(y[4], n_A_B_1) + std::pow(K_A_B_1, n_A_B_1));
	J( 3 , 1 ) = 0;
	J( 3 , 2 ) = 0;
	J( 3 , 3 ) = -D;
	J( 3 , 4 ) = -std::pow(y[4], 2*n_A_B_1)*y[0]*kB_max_1*n_A_B_1/(y[4]*std::pow(std::pow(y[4], n_A_B_1) + std::pow(K_A_B_1, n_A_B_1), 2)) + std::pow(y[4], n_A_B_1)*y[0]*kB_max_1*n_A_B_1/(y[4]*(std::pow(y[4], n_A_B_1) + std::pow(K_A_B_1, n_A_B_1)));
	J( 3 , 5 ) = 0;
	J( 4 , 0 ) = kA_1;
	J( 4 , 1 ) = 0;
	J( 4 , 2 ) = 0;
	J( 4 , 3 ) = 0;
	J( 4 , 4 ) = -D;
	J( 4 , 5 ) = 0;
	J( 5 , 0 ) = 0;
	J( 5 , 1 ) = 0;
	J( 5 , 2 ) = (1.0/2.0)*y[5]*y[2]*mu_max_1/std::pow(K_mu_glu + y[2], 2) - 1.0/2.0*y[5]*mu_max_1/(K_mu_glu + y[2]);
	J( 5 , 3 ) = 0;
	J( 5 , 4 ) = -std::pow(y[4], 2*n_A_I_1)*kI_max_1*n_A_I_1/(y[4]*std::pow(std::pow(y[4], n_A_I_1) + std::pow(K_A_I_1, n_A_I_1), 2)) + std::pow(y[4], n_A_I_1)*kI_max_1*n_A_I_1/(y[4]*(std::pow(y[4], n_A_I_1) + std::pow(K_A_I_1, n_A_I_1)));
	J( 5 , 5 ) = -1.0/2.0*y[2]*mu_max_1/(K_mu_glu + y[2]);

	dfdt[0] = 0.0;
	dfdt[1] = 0.0;
	dfdt[2] = 0.0;
	dfdt[3] = 0.0;
	dfdt[4] = 0.0;
	dfdt[5] = 0.0;

}
void Models::model_1(const ublas_vec_t  &y , ublas_vec_t &dydt , double t, std::vector <double> &part_params)
{
	//Unpack parameters
	const double D = part_params[0];
	const double g_1 = part_params[1];
	const double g_2 = part_params[2];
	const double K_A_B_1 = part_params[3];
	const double K_A_I_1 = part_params[4];
	const double k_I_1 = part_params[5];
	const double K_mu_glu = part_params[6];
	const double k_omega_B_1 = part_params[7];
	const double kA_1 = part_params[8];
	const double kB_max_1 = part_params[9];
	const double kI_max_1 = part_params[10];
	const double mu_max_1 = part_params[11];
	const double mu_max_2 = part_params[12];
	const double n_A_B_1 = part_params[13];
	const double n_A_I_1 = part_params[14];
	const double nI_1 = part_params[15];
	const double S0_glu = part_params[16];

	//Species order is: N_1 N_2 S_glu B_1 A_1 I_1 
	dydt[0] = -y[3]*y[0]*std::pow(k_I_1, nI_1)*k_omega_B_1/(std::pow(y[5], nI_1) + std::pow(k_I_1, nI_1)) - D*y[0] + y[0]*y[2]*mu_max_1/(K_mu_glu + y[2]);
	dydt[1] = -D*y[1] + y[1]*y[2]*mu_max_2/(K_mu_glu + y[2]);
	dydt[2] = D*(S0_glu - y[2]) - y[0]*y[2]*mu_max_1/(g_1*(K_mu_glu + y[2])) - y[1]*y[2]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	dydt[3] = std::pow(y[4], n_A_B_1)*y[0]*kB_max_1/(std::pow(y[4], n_A_B_1) + std::pow(K_A_B_1, n_A_B_1)) - y[3]*D;
	dydt[4] = -y[4]*D + y[0]*kA_1;
	dydt[5] = std::pow(y[4], n_A_I_1)*kI_max_1/(std::pow(y[4], n_A_I_1) + std::pow(K_A_I_1, n_A_I_1)) - 1.0/2.0*y[5]*y[2]*mu_max_1/(K_mu_glu + y[2]);

}
void Models::jac_1(const ublas_vec_t & y , ublas_mat_t &J , const double &/* t*/ , ublas_vec_t &dfdt, std::vector <double> &part_params)
{
	//Unpack parameters
	const double D = part_params[0];
	const double g_1 = part_params[1];
	const double g_2 = part_params[2];
	const double K_A_B_1 = part_params[3];
	const double K_A_I_1 = part_params[4];
	const double k_I_1 = part_params[5];
	const double K_mu_glu = part_params[6];
	const double k_omega_B_1 = part_params[7];
	const double kA_1 = part_params[8];
	const double kB_max_1 = part_params[9];
	const double kI_max_1 = part_params[10];
	const double mu_max_1 = part_params[11];
	const double mu_max_2 = part_params[12];
	const double n_A_B_1 = part_params[13];
	const double n_A_I_1 = part_params[14];
	const double nI_1 = part_params[15];
	const double S0_glu = part_params[16];

	J( 0 , 0 ) = -y[3]*std::pow(k_I_1, nI_1)*k_omega_B_1/(std::pow(y[5], nI_1) + std::pow(k_I_1, nI_1)) - D + y[2]*mu_max_1/(K_mu_glu + y[2]);
	J( 0 , 1 ) = 0;
	J( 0 , 2 ) = -y[0]*y[2]*mu_max_1/std::pow(K_mu_glu + y[2], 2) + y[0]*mu_max_1/(K_mu_glu + y[2]);
	J( 0 , 3 ) = -y[0]*std::pow(k_I_1, nI_1)*k_omega_B_1/(std::pow(y[5], nI_1) + std::pow(k_I_1, nI_1));
	J( 0 , 4 ) = 0;
	J( 0 , 5 ) = y[3]*std::pow(y[5], nI_1)*y[0]*std::pow(k_I_1, nI_1)*k_omega_B_1*nI_1/(y[5]*std::pow(std::pow(y[5], nI_1) + std::pow(k_I_1, nI_1), 2));
	J( 1 , 0 ) = 0;
	J( 1 , 1 ) = -D + y[2]*mu_max_2/(K_mu_glu + y[2]);
	J( 1 , 2 ) = -y[1]*y[2]*mu_max_2/std::pow(K_mu_glu + y[2], 2) + y[1]*mu_max_2/(K_mu_glu + y[2]);
	J( 1 , 3 ) = 0;
	J( 1 , 4 ) = 0;
	J( 1 , 5 ) = 0;
	J( 2 , 0 ) = -y[2]*mu_max_1/(g_1*(K_mu_glu + y[2]));
	J( 2 , 1 ) = -y[2]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	J( 2 , 2 ) = -D + y[0]*y[2]*mu_max_1/(g_1*std::pow(K_mu_glu + y[2], 2)) - y[0]*mu_max_1/(g_1*(K_mu_glu + y[2])) + y[1]*y[2]*mu_max_2/(g_2*std::pow(K_mu_glu + y[2], 2)) - y[1]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	J( 2 , 3 ) = 0;
	J( 2 , 4 ) = 0;
	J( 2 , 5 ) = 0;
	J( 3 , 0 ) = std::pow(y[4], n_A_B_1)*kB_max_1/(std::pow(y[4], n_A_B_1) + std::pow(K_A_B_1, n_A_B_1));
	J( 3 , 1 ) = 0;
	J( 3 , 2 ) = 0;
	J( 3 , 3 ) = -D;
	J( 3 , 4 ) = -std::pow(y[4], 2*n_A_B_1)*y[0]*kB_max_1*n_A_B_1/(y[4]*std::pow(std::pow(y[4], n_A_B_1) + std::pow(K_A_B_1, n_A_B_1), 2)) + std::pow(y[4], n_A_B_1)*y[0]*kB_max_1*n_A_B_1/(y[4]*(std::pow(y[4], n_A_B_1) + std::pow(K_A_B_1, n_A_B_1)));
	J( 3 , 5 ) = 0;
	J( 4 , 0 ) = kA_1;
	J( 4 , 1 ) = 0;
	J( 4 , 2 ) = 0;
	J( 4 , 3 ) = 0;
	J( 4 , 4 ) = -D;
	J( 4 , 5 ) = 0;
	J( 5 , 0 ) = 0;
	J( 5 , 1 ) = 0;
	J( 5 , 2 ) = (1.0/2.0)*y[5]*y[2]*mu_max_1/std::pow(K_mu_glu + y[2], 2) - 1.0/2.0*y[5]*mu_max_1/(K_mu_glu + y[2]);
	J( 5 , 3 ) = 0;
	J( 5 , 4 ) = -std::pow(y[4], 2*n_A_I_1)*kI_max_1*n_A_I_1/(y[4]*std::pow(std::pow(y[4], n_A_I_1) + std::pow(K_A_I_1, n_A_I_1), 2)) + std::pow(y[4], n_A_I_1)*kI_max_1*n_A_I_1/(y[4]*(std::pow(y[4], n_A_I_1) + std::pow(K_A_I_1, n_A_I_1)));
	J( 5 , 5 ) = -1.0/2.0*y[2]*mu_max_1/(K_mu_glu + y[2]);

	dfdt[0] = 0.0;
	dfdt[1] = 0.0;
	dfdt[2] = 0.0;
	dfdt[3] = 0.0;
	dfdt[4] = 0.0;
	dfdt[5] = 0.0;

}
void Models::model_2(const ublas_vec_t  &y , ublas_vec_t &dydt , double t, std::vector <double> &part_params)
{
	//Unpack parameters
	const double D = part_params[0];
	const double g_1 = part_params[1];
	const double g_2 = part_params[2];
	const double K_A_B_1 = part_params[3];
	const double K_A_I_1 = part_params[4];
	const double k_I_1 = part_params[5];
	const double K_mu_glu = part_params[6];
	const double k_omega_B_1 = part_params[7];
	const double kA_1 = part_params[8];
	const double kB_max_1 = part_params[9];
	const double kI_max_1 = part_params[10];
	const double mu_max_1 = part_params[11];
	const double mu_max_2 = part_params[12];
	const double n_A_B_1 = part_params[13];
	const double n_A_I_1 = part_params[14];
	const double nI_1 = part_params[15];
	const double S0_glu = part_params[16];

	//Species order is: N_1 N_2 S_glu B_1 A_1 I_1 
	dydt[0] = -y[3]*y[0]*std::pow(k_I_1, nI_1)*k_omega_B_1/(std::pow(y[5], nI_1) + std::pow(k_I_1, nI_1)) - D*y[0] + y[0]*y[2]*mu_max_1/(K_mu_glu + y[2]);
	dydt[1] = -y[3]*y[1]*k_omega_B_1 - D*y[1] + y[1]*y[2]*mu_max_2/(K_mu_glu + y[2]);
	dydt[2] = D*(S0_glu - y[2]) - y[0]*y[2]*mu_max_1/(g_1*(K_mu_glu + y[2])) - y[1]*y[2]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	dydt[3] = std::pow(y[4], n_A_B_1)*y[0]*kB_max_1/(std::pow(y[4], n_A_B_1) + std::pow(K_A_B_1, n_A_B_1)) - y[3]*D;
	dydt[4] = -y[4]*D + y[0]*kA_1;
	dydt[5] = -1.0/2.0*y[5]*y[2]*mu_max_1/(K_mu_glu + y[2]) + std::pow(K_A_I_1, n_A_I_1)*kI_max_1/(std::pow(y[4], n_A_I_1) + std::pow(K_A_I_1, n_A_I_1));

}
void Models::jac_2(const ublas_vec_t & y , ublas_mat_t &J , const double &/* t*/ , ublas_vec_t &dfdt, std::vector <double> &part_params)
{
	//Unpack parameters
	const double D = part_params[0];
	const double g_1 = part_params[1];
	const double g_2 = part_params[2];
	const double K_A_B_1 = part_params[3];
	const double K_A_I_1 = part_params[4];
	const double k_I_1 = part_params[5];
	const double K_mu_glu = part_params[6];
	const double k_omega_B_1 = part_params[7];
	const double kA_1 = part_params[8];
	const double kB_max_1 = part_params[9];
	const double kI_max_1 = part_params[10];
	const double mu_max_1 = part_params[11];
	const double mu_max_2 = part_params[12];
	const double n_A_B_1 = part_params[13];
	const double n_A_I_1 = part_params[14];
	const double nI_1 = part_params[15];
	const double S0_glu = part_params[16];

	J( 0 , 0 ) = -y[3]*std::pow(k_I_1, nI_1)*k_omega_B_1/(std::pow(y[5], nI_1) + std::pow(k_I_1, nI_1)) - D + y[2]*mu_max_1/(K_mu_glu + y[2]);
	J( 0 , 1 ) = 0;
	J( 0 , 2 ) = -y[0]*y[2]*mu_max_1/std::pow(K_mu_glu + y[2], 2) + y[0]*mu_max_1/(K_mu_glu + y[2]);
	J( 0 , 3 ) = -y[0]*std::pow(k_I_1, nI_1)*k_omega_B_1/(std::pow(y[5], nI_1) + std::pow(k_I_1, nI_1));
	J( 0 , 4 ) = 0;
	J( 0 , 5 ) = y[3]*std::pow(y[5], nI_1)*y[0]*std::pow(k_I_1, nI_1)*k_omega_B_1*nI_1/(y[5]*std::pow(std::pow(y[5], nI_1) + std::pow(k_I_1, nI_1), 2));
	J( 1 , 0 ) = 0;
	J( 1 , 1 ) = -y[3]*k_omega_B_1 - D + y[2]*mu_max_2/(K_mu_glu + y[2]);
	J( 1 , 2 ) = -y[1]*y[2]*mu_max_2/std::pow(K_mu_glu + y[2], 2) + y[1]*mu_max_2/(K_mu_glu + y[2]);
	J( 1 , 3 ) = -y[1]*k_omega_B_1;
	J( 1 , 4 ) = 0;
	J( 1 , 5 ) = 0;
	J( 2 , 0 ) = -y[2]*mu_max_1/(g_1*(K_mu_glu + y[2]));
	J( 2 , 1 ) = -y[2]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	J( 2 , 2 ) = -D + y[0]*y[2]*mu_max_1/(g_1*std::pow(K_mu_glu + y[2], 2)) - y[0]*mu_max_1/(g_1*(K_mu_glu + y[2])) + y[1]*y[2]*mu_max_2/(g_2*std::pow(K_mu_glu + y[2], 2)) - y[1]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	J( 2 , 3 ) = 0;
	J( 2 , 4 ) = 0;
	J( 2 , 5 ) = 0;
	J( 3 , 0 ) = std::pow(y[4], n_A_B_1)*kB_max_1/(std::pow(y[4], n_A_B_1) + std::pow(K_A_B_1, n_A_B_1));
	J( 3 , 1 ) = 0;
	J( 3 , 2 ) = 0;
	J( 3 , 3 ) = -D;
	J( 3 , 4 ) = -std::pow(y[4], 2*n_A_B_1)*y[0]*kB_max_1*n_A_B_1/(y[4]*std::pow(std::pow(y[4], n_A_B_1) + std::pow(K_A_B_1, n_A_B_1), 2)) + std::pow(y[4], n_A_B_1)*y[0]*kB_max_1*n_A_B_1/(y[4]*(std::pow(y[4], n_A_B_1) + std::pow(K_A_B_1, n_A_B_1)));
	J( 3 , 5 ) = 0;
	J( 4 , 0 ) = kA_1;
	J( 4 , 1 ) = 0;
	J( 4 , 2 ) = 0;
	J( 4 , 3 ) = 0;
	J( 4 , 4 ) = -D;
	J( 4 , 5 ) = 0;
	J( 5 , 0 ) = 0;
	J( 5 , 1 ) = 0;
	J( 5 , 2 ) = (1.0/2.0)*y[5]*y[2]*mu_max_1/std::pow(K_mu_glu + y[2], 2) - 1.0/2.0*y[5]*mu_max_1/(K_mu_glu + y[2]);
	J( 5 , 3 ) = 0;
	J( 5 , 4 ) = -std::pow(y[4], n_A_I_1)*std::pow(K_A_I_1, n_A_I_1)*kI_max_1*n_A_I_1/(y[4]*std::pow(std::pow(y[4], n_A_I_1) + std::pow(K_A_I_1, n_A_I_1), 2));
	J( 5 , 5 ) = -1.0/2.0*y[2]*mu_max_1/(K_mu_glu + y[2]);

	dfdt[0] = 0.0;
	dfdt[1] = 0.0;
	dfdt[2] = 0.0;
	dfdt[3] = 0.0;
	dfdt[4] = 0.0;
	dfdt[5] = 0.0;

}
void Models::model_3(const ublas_vec_t  &y , ublas_vec_t &dydt , double t, std::vector <double> &part_params)
{
	//Unpack parameters
	const double D = part_params[0];
	const double g_1 = part_params[1];
	const double g_2 = part_params[2];
	const double K_A_B_1 = part_params[3];
	const double K_A_I_1 = part_params[4];
	const double k_I_1 = part_params[5];
	const double K_mu_glu = part_params[6];
	const double k_omega_B_1 = part_params[7];
	const double kA_1 = part_params[8];
	const double kB_max_1 = part_params[9];
	const double kI_max_1 = part_params[10];
	const double mu_max_1 = part_params[11];
	const double mu_max_2 = part_params[12];
	const double n_A_B_1 = part_params[13];
	const double n_A_I_1 = part_params[14];
	const double nI_1 = part_params[15];
	const double S0_glu = part_params[16];

	//Species order is: N_1 N_2 S_glu B_1 A_1 I_1 
	dydt[0] = -y[3]*y[0]*std::pow(k_I_1, nI_1)*k_omega_B_1/(std::pow(y[5], nI_1) + std::pow(k_I_1, nI_1)) - D*y[0] + y[0]*y[2]*mu_max_1/(K_mu_glu + y[2]);
	dydt[1] = -D*y[1] + y[1]*y[2]*mu_max_2/(K_mu_glu + y[2]);
	dydt[2] = D*(S0_glu - y[2]) - y[0]*y[2]*mu_max_1/(g_1*(K_mu_glu + y[2])) - y[1]*y[2]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	dydt[3] = std::pow(y[4], n_A_B_1)*y[0]*kB_max_1/(std::pow(y[4], n_A_B_1) + std::pow(K_A_B_1, n_A_B_1)) - y[3]*D;
	dydt[4] = -y[4]*D + y[0]*kA_1;
	dydt[5] = -1.0/2.0*y[5]*y[2]*mu_max_1/(K_mu_glu + y[2]) + std::pow(K_A_I_1, n_A_I_1)*kI_max_1/(std::pow(y[4], n_A_I_1) + std::pow(K_A_I_1, n_A_I_1));

}
void Models::jac_3(const ublas_vec_t & y , ublas_mat_t &J , const double &/* t*/ , ublas_vec_t &dfdt, std::vector <double> &part_params)
{
	//Unpack parameters
	const double D = part_params[0];
	const double g_1 = part_params[1];
	const double g_2 = part_params[2];
	const double K_A_B_1 = part_params[3];
	const double K_A_I_1 = part_params[4];
	const double k_I_1 = part_params[5];
	const double K_mu_glu = part_params[6];
	const double k_omega_B_1 = part_params[7];
	const double kA_1 = part_params[8];
	const double kB_max_1 = part_params[9];
	const double kI_max_1 = part_params[10];
	const double mu_max_1 = part_params[11];
	const double mu_max_2 = part_params[12];
	const double n_A_B_1 = part_params[13];
	const double n_A_I_1 = part_params[14];
	const double nI_1 = part_params[15];
	const double S0_glu = part_params[16];

	J( 0 , 0 ) = -y[3]*std::pow(k_I_1, nI_1)*k_omega_B_1/(std::pow(y[5], nI_1) + std::pow(k_I_1, nI_1)) - D + y[2]*mu_max_1/(K_mu_glu + y[2]);
	J( 0 , 1 ) = 0;
	J( 0 , 2 ) = -y[0]*y[2]*mu_max_1/std::pow(K_mu_glu + y[2], 2) + y[0]*mu_max_1/(K_mu_glu + y[2]);
	J( 0 , 3 ) = -y[0]*std::pow(k_I_1, nI_1)*k_omega_B_1/(std::pow(y[5], nI_1) + std::pow(k_I_1, nI_1));
	J( 0 , 4 ) = 0;
	J( 0 , 5 ) = y[3]*std::pow(y[5], nI_1)*y[0]*std::pow(k_I_1, nI_1)*k_omega_B_1*nI_1/(y[5]*std::pow(std::pow(y[5], nI_1) + std::pow(k_I_1, nI_1), 2));
	J( 1 , 0 ) = 0;
	J( 1 , 1 ) = -D + y[2]*mu_max_2/(K_mu_glu + y[2]);
	J( 1 , 2 ) = -y[1]*y[2]*mu_max_2/std::pow(K_mu_glu + y[2], 2) + y[1]*mu_max_2/(K_mu_glu + y[2]);
	J( 1 , 3 ) = 0;
	J( 1 , 4 ) = 0;
	J( 1 , 5 ) = 0;
	J( 2 , 0 ) = -y[2]*mu_max_1/(g_1*(K_mu_glu + y[2]));
	J( 2 , 1 ) = -y[2]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	J( 2 , 2 ) = -D + y[0]*y[2]*mu_max_1/(g_1*std::pow(K_mu_glu + y[2], 2)) - y[0]*mu_max_1/(g_1*(K_mu_glu + y[2])) + y[1]*y[2]*mu_max_2/(g_2*std::pow(K_mu_glu + y[2], 2)) - y[1]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	J( 2 , 3 ) = 0;
	J( 2 , 4 ) = 0;
	J( 2 , 5 ) = 0;
	J( 3 , 0 ) = std::pow(y[4], n_A_B_1)*kB_max_1/(std::pow(y[4], n_A_B_1) + std::pow(K_A_B_1, n_A_B_1));
	J( 3 , 1 ) = 0;
	J( 3 , 2 ) = 0;
	J( 3 , 3 ) = -D;
	J( 3 , 4 ) = -std::pow(y[4], 2*n_A_B_1)*y[0]*kB_max_1*n_A_B_1/(y[4]*std::pow(std::pow(y[4], n_A_B_1) + std::pow(K_A_B_1, n_A_B_1), 2)) + std::pow(y[4], n_A_B_1)*y[0]*kB_max_1*n_A_B_1/(y[4]*(std::pow(y[4], n_A_B_1) + std::pow(K_A_B_1, n_A_B_1)));
	J( 3 , 5 ) = 0;
	J( 4 , 0 ) = kA_1;
	J( 4 , 1 ) = 0;
	J( 4 , 2 ) = 0;
	J( 4 , 3 ) = 0;
	J( 4 , 4 ) = -D;
	J( 4 , 5 ) = 0;
	J( 5 , 0 ) = 0;
	J( 5 , 1 ) = 0;
	J( 5 , 2 ) = (1.0/2.0)*y[5]*y[2]*mu_max_1/std::pow(K_mu_glu + y[2], 2) - 1.0/2.0*y[5]*mu_max_1/(K_mu_glu + y[2]);
	J( 5 , 3 ) = 0;
	J( 5 , 4 ) = -std::pow(y[4], n_A_I_1)*std::pow(K_A_I_1, n_A_I_1)*kI_max_1*n_A_I_1/(y[4]*std::pow(std::pow(y[4], n_A_I_1) + std::pow(K_A_I_1, n_A_I_1), 2));
	J( 5 , 5 ) = -1.0/2.0*y[2]*mu_max_1/(K_mu_glu + y[2]);

	dfdt[0] = 0.0;
	dfdt[1] = 0.0;
	dfdt[2] = 0.0;
	dfdt[3] = 0.0;
	dfdt[4] = 0.0;
	dfdt[5] = 0.0;

}
void Models::model_4(const ublas_vec_t  &y , ublas_vec_t &dydt , double t, std::vector <double> &part_params)
{
	//Unpack parameters
	const double D = part_params[0];
	const double g_1 = part_params[1];
	const double g_2 = part_params[2];
	const double K_A_B_1 = part_params[3];
	const double k_I_1 = part_params[4];
	const double K_mu_glu = part_params[5];
	const double k_omega_B_1 = part_params[6];
	const double kA_1 = part_params[7];
	const double kB_max_1 = part_params[8];
	const double kI_max_1 = part_params[9];
	const double mu_max_1 = part_params[10];
	const double mu_max_2 = part_params[11];
	const double n_A_B_1 = part_params[12];
	const double nI_1 = part_params[13];
	const double S0_glu = part_params[14];

	//Species order is: N_1 N_2 S_glu B_1 A_1 I_1 
	dydt[0] = -y[3]*y[0]*std::pow(k_I_1, nI_1)*k_omega_B_1/(std::pow(y[5], nI_1) + std::pow(k_I_1, nI_1)) - D*y[0] + y[0]*y[2]*mu_max_1/(K_mu_glu + y[2]);
	dydt[1] = -y[3]*y[1]*k_omega_B_1 - D*y[1] + y[1]*y[2]*mu_max_2/(K_mu_glu + y[2]);
	dydt[2] = D*(S0_glu - y[2]) - y[0]*y[2]*mu_max_1/(g_1*(K_mu_glu + y[2])) - y[1]*y[2]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	dydt[3] = std::pow(y[4], n_A_B_1)*y[0]*kB_max_1/(std::pow(y[4], n_A_B_1) + std::pow(K_A_B_1, n_A_B_1)) - y[3]*D;
	dydt[4] = -y[4]*D + y[0]*kA_1;
	dydt[5] = -1.0/2.0*y[5]*y[2]*mu_max_1/(K_mu_glu + y[2]) + kI_max_1;

}
void Models::jac_4(const ublas_vec_t & y , ublas_mat_t &J , const double &/* t*/ , ublas_vec_t &dfdt, std::vector <double> &part_params)
{
	//Unpack parameters
	const double D = part_params[0];
	const double g_1 = part_params[1];
	const double g_2 = part_params[2];
	const double K_A_B_1 = part_params[3];
	const double k_I_1 = part_params[4];
	const double K_mu_glu = part_params[5];
	const double k_omega_B_1 = part_params[6];
	const double kA_1 = part_params[7];
	const double kB_max_1 = part_params[8];
	const double kI_max_1 = part_params[9];
	const double mu_max_1 = part_params[10];
	const double mu_max_2 = part_params[11];
	const double n_A_B_1 = part_params[12];
	const double nI_1 = part_params[13];
	const double S0_glu = part_params[14];

	J( 0 , 0 ) = -y[3]*std::pow(k_I_1, nI_1)*k_omega_B_1/(std::pow(y[5], nI_1) + std::pow(k_I_1, nI_1)) - D + y[2]*mu_max_1/(K_mu_glu + y[2]);
	J( 0 , 1 ) = 0;
	J( 0 , 2 ) = -y[0]*y[2]*mu_max_1/std::pow(K_mu_glu + y[2], 2) + y[0]*mu_max_1/(K_mu_glu + y[2]);
	J( 0 , 3 ) = -y[0]*std::pow(k_I_1, nI_1)*k_omega_B_1/(std::pow(y[5], nI_1) + std::pow(k_I_1, nI_1));
	J( 0 , 4 ) = 0;
	J( 0 , 5 ) = y[3]*std::pow(y[5], nI_1)*y[0]*std::pow(k_I_1, nI_1)*k_omega_B_1*nI_1/(y[5]*std::pow(std::pow(y[5], nI_1) + std::pow(k_I_1, nI_1), 2));
	J( 1 , 0 ) = 0;
	J( 1 , 1 ) = -y[3]*k_omega_B_1 - D + y[2]*mu_max_2/(K_mu_glu + y[2]);
	J( 1 , 2 ) = -y[1]*y[2]*mu_max_2/std::pow(K_mu_glu + y[2], 2) + y[1]*mu_max_2/(K_mu_glu + y[2]);
	J( 1 , 3 ) = -y[1]*k_omega_B_1;
	J( 1 , 4 ) = 0;
	J( 1 , 5 ) = 0;
	J( 2 , 0 ) = -y[2]*mu_max_1/(g_1*(K_mu_glu + y[2]));
	J( 2 , 1 ) = -y[2]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	J( 2 , 2 ) = -D + y[0]*y[2]*mu_max_1/(g_1*std::pow(K_mu_glu + y[2], 2)) - y[0]*mu_max_1/(g_1*(K_mu_glu + y[2])) + y[1]*y[2]*mu_max_2/(g_2*std::pow(K_mu_glu + y[2], 2)) - y[1]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	J( 2 , 3 ) = 0;
	J( 2 , 4 ) = 0;
	J( 2 , 5 ) = 0;
	J( 3 , 0 ) = std::pow(y[4], n_A_B_1)*kB_max_1/(std::pow(y[4], n_A_B_1) + std::pow(K_A_B_1, n_A_B_1));
	J( 3 , 1 ) = 0;
	J( 3 , 2 ) = 0;
	J( 3 , 3 ) = -D;
	J( 3 , 4 ) = -std::pow(y[4], 2*n_A_B_1)*y[0]*kB_max_1*n_A_B_1/(y[4]*std::pow(std::pow(y[4], n_A_B_1) + std::pow(K_A_B_1, n_A_B_1), 2)) + std::pow(y[4], n_A_B_1)*y[0]*kB_max_1*n_A_B_1/(y[4]*(std::pow(y[4], n_A_B_1) + std::pow(K_A_B_1, n_A_B_1)));
	J( 3 , 5 ) = 0;
	J( 4 , 0 ) = kA_1;
	J( 4 , 1 ) = 0;
	J( 4 , 2 ) = 0;
	J( 4 , 3 ) = 0;
	J( 4 , 4 ) = -D;
	J( 4 , 5 ) = 0;
	J( 5 , 0 ) = 0;
	J( 5 , 1 ) = 0;
	J( 5 , 2 ) = (1.0/2.0)*y[5]*y[2]*mu_max_1/std::pow(K_mu_glu + y[2], 2) - 1.0/2.0*y[5]*mu_max_1/(K_mu_glu + y[2]);
	J( 5 , 3 ) = 0;
	J( 5 , 4 ) = 0;
	J( 5 , 5 ) = -1.0/2.0*y[2]*mu_max_1/(K_mu_glu + y[2]);

	dfdt[0] = 0.0;
	dfdt[1] = 0.0;
	dfdt[2] = 0.0;
	dfdt[3] = 0.0;
	dfdt[4] = 0.0;
	dfdt[5] = 0.0;

}
void Models::model_5(const ublas_vec_t  &y , ublas_vec_t &dydt , double t, std::vector <double> &part_params)
{
	//Unpack parameters
	const double D = part_params[0];
	const double g_1 = part_params[1];
	const double g_2 = part_params[2];
	const double K_A_B_1 = part_params[3];
	const double k_I_1 = part_params[4];
	const double K_mu_glu = part_params[5];
	const double k_omega_B_1 = part_params[6];
	const double kA_1 = part_params[7];
	const double kB_max_1 = part_params[8];
	const double kI_max_1 = part_params[9];
	const double mu_max_1 = part_params[10];
	const double mu_max_2 = part_params[11];
	const double n_A_B_1 = part_params[12];
	const double nI_1 = part_params[13];
	const double S0_glu = part_params[14];

	//Species order is: N_1 N_2 S_glu B_1 A_1 I_1 
	dydt[0] = -y[3]*y[0]*std::pow(k_I_1, nI_1)*k_omega_B_1/(std::pow(y[5], nI_1) + std::pow(k_I_1, nI_1)) - D*y[0] + y[0]*y[2]*mu_max_1/(K_mu_glu + y[2]);
	dydt[1] = -D*y[1] + y[1]*y[2]*mu_max_2/(K_mu_glu + y[2]);
	dydt[2] = D*(S0_glu - y[2]) - y[0]*y[2]*mu_max_1/(g_1*(K_mu_glu + y[2])) - y[1]*y[2]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	dydt[3] = std::pow(y[4], n_A_B_1)*y[0]*kB_max_1/(std::pow(y[4], n_A_B_1) + std::pow(K_A_B_1, n_A_B_1)) - y[3]*D;
	dydt[4] = -y[4]*D + y[0]*kA_1;
	dydt[5] = -1.0/2.0*y[5]*y[2]*mu_max_1/(K_mu_glu + y[2]) + kI_max_1;

}
void Models::jac_5(const ublas_vec_t & y , ublas_mat_t &J , const double &/* t*/ , ublas_vec_t &dfdt, std::vector <double> &part_params)
{
	//Unpack parameters
	const double D = part_params[0];
	const double g_1 = part_params[1];
	const double g_2 = part_params[2];
	const double K_A_B_1 = part_params[3];
	const double k_I_1 = part_params[4];
	const double K_mu_glu = part_params[5];
	const double k_omega_B_1 = part_params[6];
	const double kA_1 = part_params[7];
	const double kB_max_1 = part_params[8];
	const double kI_max_1 = part_params[9];
	const double mu_max_1 = part_params[10];
	const double mu_max_2 = part_params[11];
	const double n_A_B_1 = part_params[12];
	const double nI_1 = part_params[13];
	const double S0_glu = part_params[14];

	J( 0 , 0 ) = -y[3]*std::pow(k_I_1, nI_1)*k_omega_B_1/(std::pow(y[5], nI_1) + std::pow(k_I_1, nI_1)) - D + y[2]*mu_max_1/(K_mu_glu + y[2]);
	J( 0 , 1 ) = 0;
	J( 0 , 2 ) = -y[0]*y[2]*mu_max_1/std::pow(K_mu_glu + y[2], 2) + y[0]*mu_max_1/(K_mu_glu + y[2]);
	J( 0 , 3 ) = -y[0]*std::pow(k_I_1, nI_1)*k_omega_B_1/(std::pow(y[5], nI_1) + std::pow(k_I_1, nI_1));
	J( 0 , 4 ) = 0;
	J( 0 , 5 ) = y[3]*std::pow(y[5], nI_1)*y[0]*std::pow(k_I_1, nI_1)*k_omega_B_1*nI_1/(y[5]*std::pow(std::pow(y[5], nI_1) + std::pow(k_I_1, nI_1), 2));
	J( 1 , 0 ) = 0;
	J( 1 , 1 ) = -D + y[2]*mu_max_2/(K_mu_glu + y[2]);
	J( 1 , 2 ) = -y[1]*y[2]*mu_max_2/std::pow(K_mu_glu + y[2], 2) + y[1]*mu_max_2/(K_mu_glu + y[2]);
	J( 1 , 3 ) = 0;
	J( 1 , 4 ) = 0;
	J( 1 , 5 ) = 0;
	J( 2 , 0 ) = -y[2]*mu_max_1/(g_1*(K_mu_glu + y[2]));
	J( 2 , 1 ) = -y[2]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	J( 2 , 2 ) = -D + y[0]*y[2]*mu_max_1/(g_1*std::pow(K_mu_glu + y[2], 2)) - y[0]*mu_max_1/(g_1*(K_mu_glu + y[2])) + y[1]*y[2]*mu_max_2/(g_2*std::pow(K_mu_glu + y[2], 2)) - y[1]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	J( 2 , 3 ) = 0;
	J( 2 , 4 ) = 0;
	J( 2 , 5 ) = 0;
	J( 3 , 0 ) = std::pow(y[4], n_A_B_1)*kB_max_1/(std::pow(y[4], n_A_B_1) + std::pow(K_A_B_1, n_A_B_1));
	J( 3 , 1 ) = 0;
	J( 3 , 2 ) = 0;
	J( 3 , 3 ) = -D;
	J( 3 , 4 ) = -std::pow(y[4], 2*n_A_B_1)*y[0]*kB_max_1*n_A_B_1/(y[4]*std::pow(std::pow(y[4], n_A_B_1) + std::pow(K_A_B_1, n_A_B_1), 2)) + std::pow(y[4], n_A_B_1)*y[0]*kB_max_1*n_A_B_1/(y[4]*(std::pow(y[4], n_A_B_1) + std::pow(K_A_B_1, n_A_B_1)));
	J( 3 , 5 ) = 0;
	J( 4 , 0 ) = kA_1;
	J( 4 , 1 ) = 0;
	J( 4 , 2 ) = 0;
	J( 4 , 3 ) = 0;
	J( 4 , 4 ) = -D;
	J( 4 , 5 ) = 0;
	J( 5 , 0 ) = 0;
	J( 5 , 1 ) = 0;
	J( 5 , 2 ) = (1.0/2.0)*y[5]*y[2]*mu_max_1/std::pow(K_mu_glu + y[2], 2) - 1.0/2.0*y[5]*mu_max_1/(K_mu_glu + y[2]);
	J( 5 , 3 ) = 0;
	J( 5 , 4 ) = 0;
	J( 5 , 5 ) = -1.0/2.0*y[2]*mu_max_1/(K_mu_glu + y[2]);

	dfdt[0] = 0.0;
	dfdt[1] = 0.0;
	dfdt[2] = 0.0;
	dfdt[3] = 0.0;
	dfdt[4] = 0.0;
	dfdt[5] = 0.0;

}
void Models::model_6(const ublas_vec_t  &y , ublas_vec_t &dydt , double t, std::vector <double> &part_params)
{
	//Unpack parameters
	const double D = part_params[0];
	const double g_1 = part_params[1];
	const double g_2 = part_params[2];
	const double K_A_B_1 = part_params[3];
	const double K_mu_glu = part_params[4];
	const double k_omega_B_1 = part_params[5];
	const double kA_1 = part_params[6];
	const double kB_max_1 = part_params[7];
	const double mu_max_1 = part_params[8];
	const double mu_max_2 = part_params[9];
	const double n_A_B_1 = part_params[10];
	const double S0_glu = part_params[11];

	//Species order is: N_1 N_2 S_glu B_1 A_1 
	dydt[0] = -y[3]*y[0]*k_omega_B_1 - D*y[0] + y[0]*y[2]*mu_max_1/(K_mu_glu + y[2]);
	dydt[1] = -y[3]*y[1]*k_omega_B_1 - D*y[1] + y[1]*y[2]*mu_max_2/(K_mu_glu + y[2]);
	dydt[2] = D*(S0_glu - y[2]) - y[0]*y[2]*mu_max_1/(g_1*(K_mu_glu + y[2])) - y[1]*y[2]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	dydt[3] = std::pow(y[4], n_A_B_1)*y[0]*kB_max_1/(std::pow(y[4], n_A_B_1) + std::pow(K_A_B_1, n_A_B_1)) - y[3]*D;
	dydt[4] = -y[4]*D + y[0]*kA_1;

}
void Models::jac_6(const ublas_vec_t & y , ublas_mat_t &J , const double &/* t*/ , ublas_vec_t &dfdt, std::vector <double> &part_params)
{
	//Unpack parameters
	const double D = part_params[0];
	const double g_1 = part_params[1];
	const double g_2 = part_params[2];
	const double K_A_B_1 = part_params[3];
	const double K_mu_glu = part_params[4];
	const double k_omega_B_1 = part_params[5];
	const double kA_1 = part_params[6];
	const double kB_max_1 = part_params[7];
	const double mu_max_1 = part_params[8];
	const double mu_max_2 = part_params[9];
	const double n_A_B_1 = part_params[10];
	const double S0_glu = part_params[11];

	J( 0 , 0 ) = -y[3]*k_omega_B_1 - D + y[2]*mu_max_1/(K_mu_glu + y[2]);
	J( 0 , 1 ) = 0;
	J( 0 , 2 ) = -y[0]*y[2]*mu_max_1/std::pow(K_mu_glu + y[2], 2) + y[0]*mu_max_1/(K_mu_glu + y[2]);
	J( 0 , 3 ) = -y[0]*k_omega_B_1;
	J( 0 , 4 ) = 0;
	J( 1 , 0 ) = 0;
	J( 1 , 1 ) = -y[3]*k_omega_B_1 - D + y[2]*mu_max_2/(K_mu_glu + y[2]);
	J( 1 , 2 ) = -y[1]*y[2]*mu_max_2/std::pow(K_mu_glu + y[2], 2) + y[1]*mu_max_2/(K_mu_glu + y[2]);
	J( 1 , 3 ) = -y[1]*k_omega_B_1;
	J( 1 , 4 ) = 0;
	J( 2 , 0 ) = -y[2]*mu_max_1/(g_1*(K_mu_glu + y[2]));
	J( 2 , 1 ) = -y[2]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	J( 2 , 2 ) = -D + y[0]*y[2]*mu_max_1/(g_1*std::pow(K_mu_glu + y[2], 2)) - y[0]*mu_max_1/(g_1*(K_mu_glu + y[2])) + y[1]*y[2]*mu_max_2/(g_2*std::pow(K_mu_glu + y[2], 2)) - y[1]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	J( 2 , 3 ) = 0;
	J( 2 , 4 ) = 0;
	J( 3 , 0 ) = std::pow(y[4], n_A_B_1)*kB_max_1/(std::pow(y[4], n_A_B_1) + std::pow(K_A_B_1, n_A_B_1));
	J( 3 , 1 ) = 0;
	J( 3 , 2 ) = 0;
	J( 3 , 3 ) = -D;
	J( 3 , 4 ) = -std::pow(y[4], 2*n_A_B_1)*y[0]*kB_max_1*n_A_B_1/(y[4]*std::pow(std::pow(y[4], n_A_B_1) + std::pow(K_A_B_1, n_A_B_1), 2)) + std::pow(y[4], n_A_B_1)*y[0]*kB_max_1*n_A_B_1/(y[4]*(std::pow(y[4], n_A_B_1) + std::pow(K_A_B_1, n_A_B_1)));
	J( 4 , 0 ) = kA_1;
	J( 4 , 1 ) = 0;
	J( 4 , 2 ) = 0;
	J( 4 , 3 ) = 0;
	J( 4 , 4 ) = -D;

	dfdt[0] = 0.0;
	dfdt[1] = 0.0;
	dfdt[2] = 0.0;
	dfdt[3] = 0.0;
	dfdt[4] = 0.0;

}
void Models::model_7(const ublas_vec_t  &y , ublas_vec_t &dydt , double t, std::vector <double> &part_params)
{
	//Unpack parameters
	const double D = part_params[0];
	const double g_1 = part_params[1];
	const double g_2 = part_params[2];
	const double K_A_B_1 = part_params[3];
	const double K_mu_glu = part_params[4];
	const double k_omega_B_1 = part_params[5];
	const double kA_1 = part_params[6];
	const double kB_max_1 = part_params[7];
	const double mu_max_1 = part_params[8];
	const double mu_max_2 = part_params[9];
	const double n_A_B_1 = part_params[10];
	const double S0_glu = part_params[11];

	//Species order is: N_1 N_2 S_glu B_1 A_1 
	dydt[0] = -y[3]*y[0]*k_omega_B_1 - D*y[0] + y[0]*y[2]*mu_max_1/(K_mu_glu + y[2]);
	dydt[1] = -D*y[1] + y[1]*y[2]*mu_max_2/(K_mu_glu + y[2]);
	dydt[2] = D*(S0_glu - y[2]) - y[0]*y[2]*mu_max_1/(g_1*(K_mu_glu + y[2])) - y[1]*y[2]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	dydt[3] = std::pow(y[4], n_A_B_1)*y[0]*kB_max_1/(std::pow(y[4], n_A_B_1) + std::pow(K_A_B_1, n_A_B_1)) - y[3]*D;
	dydt[4] = -y[4]*D + y[0]*kA_1;

}
void Models::jac_7(const ublas_vec_t & y , ublas_mat_t &J , const double &/* t*/ , ublas_vec_t &dfdt, std::vector <double> &part_params)
{
	//Unpack parameters
	const double D = part_params[0];
	const double g_1 = part_params[1];
	const double g_2 = part_params[2];
	const double K_A_B_1 = part_params[3];
	const double K_mu_glu = part_params[4];
	const double k_omega_B_1 = part_params[5];
	const double kA_1 = part_params[6];
	const double kB_max_1 = part_params[7];
	const double mu_max_1 = part_params[8];
	const double mu_max_2 = part_params[9];
	const double n_A_B_1 = part_params[10];
	const double S0_glu = part_params[11];

	J( 0 , 0 ) = -y[3]*k_omega_B_1 - D + y[2]*mu_max_1/(K_mu_glu + y[2]);
	J( 0 , 1 ) = 0;
	J( 0 , 2 ) = -y[0]*y[2]*mu_max_1/std::pow(K_mu_glu + y[2], 2) + y[0]*mu_max_1/(K_mu_glu + y[2]);
	J( 0 , 3 ) = -y[0]*k_omega_B_1;
	J( 0 , 4 ) = 0;
	J( 1 , 0 ) = 0;
	J( 1 , 1 ) = -D + y[2]*mu_max_2/(K_mu_glu + y[2]);
	J( 1 , 2 ) = -y[1]*y[2]*mu_max_2/std::pow(K_mu_glu + y[2], 2) + y[1]*mu_max_2/(K_mu_glu + y[2]);
	J( 1 , 3 ) = 0;
	J( 1 , 4 ) = 0;
	J( 2 , 0 ) = -y[2]*mu_max_1/(g_1*(K_mu_glu + y[2]));
	J( 2 , 1 ) = -y[2]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	J( 2 , 2 ) = -D + y[0]*y[2]*mu_max_1/(g_1*std::pow(K_mu_glu + y[2], 2)) - y[0]*mu_max_1/(g_1*(K_mu_glu + y[2])) + y[1]*y[2]*mu_max_2/(g_2*std::pow(K_mu_glu + y[2], 2)) - y[1]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	J( 2 , 3 ) = 0;
	J( 2 , 4 ) = 0;
	J( 3 , 0 ) = std::pow(y[4], n_A_B_1)*kB_max_1/(std::pow(y[4], n_A_B_1) + std::pow(K_A_B_1, n_A_B_1));
	J( 3 , 1 ) = 0;
	J( 3 , 2 ) = 0;
	J( 3 , 3 ) = -D;
	J( 3 , 4 ) = -std::pow(y[4], 2*n_A_B_1)*y[0]*kB_max_1*n_A_B_1/(y[4]*std::pow(std::pow(y[4], n_A_B_1) + std::pow(K_A_B_1, n_A_B_1), 2)) + std::pow(y[4], n_A_B_1)*y[0]*kB_max_1*n_A_B_1/(y[4]*(std::pow(y[4], n_A_B_1) + std::pow(K_A_B_1, n_A_B_1)));
	J( 4 , 0 ) = kA_1;
	J( 4 , 1 ) = 0;
	J( 4 , 2 ) = 0;
	J( 4 , 3 ) = 0;
	J( 4 , 4 ) = -D;

	dfdt[0] = 0.0;
	dfdt[1] = 0.0;
	dfdt[2] = 0.0;
	dfdt[3] = 0.0;
	dfdt[4] = 0.0;

}
void Models::model_8(const ublas_vec_t  &y , ublas_vec_t &dydt , double t, std::vector <double> &part_params)
{
	//Unpack parameters
	const double D = part_params[0];
	const double g_1 = part_params[1];
	const double g_2 = part_params[2];
	const double K_A_B_1 = part_params[3];
	const double K_A_I_1 = part_params[4];
	const double K_mu_glu = part_params[5];
	const double k_omega_B_1 = part_params[6];
	const double kA_1 = part_params[7];
	const double kB_max_1 = part_params[8];
	const double kI_max_1 = part_params[9];
	const double mu_max_1 = part_params[10];
	const double mu_max_2 = part_params[11];
	const double n_A_B_1 = part_params[12];
	const double n_A_I_1 = part_params[13];
	const double S0_glu = part_params[14];

	//Species order is: N_1 N_2 S_glu B_1 A_1 I_1 
	dydt[0] = -D*y[0] + y[0]*y[2]*mu_max_1/(K_mu_glu + y[2]);
	dydt[1] = -y[3]*y[1]*k_omega_B_1 - D*y[1] + y[1]*y[2]*mu_max_2/(K_mu_glu + y[2]);
	dydt[2] = D*(S0_glu - y[2]) - y[0]*y[2]*mu_max_1/(g_1*(K_mu_glu + y[2])) - y[1]*y[2]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	dydt[3] = std::pow(y[4], n_A_B_1)*y[0]*kB_max_1/(std::pow(y[4], n_A_B_1) + std::pow(K_A_B_1, n_A_B_1)) - y[3]*D;
	dydt[4] = -y[4]*D + y[0]*kA_1;
	dydt[5] = std::pow(y[4], n_A_I_1)*kI_max_1/(std::pow(y[4], n_A_I_1) + std::pow(K_A_I_1, n_A_I_1)) - 1.0/2.0*y[5]*y[2]*mu_max_1/(K_mu_glu + y[2]);

}
void Models::jac_8(const ublas_vec_t & y , ublas_mat_t &J , const double &/* t*/ , ublas_vec_t &dfdt, std::vector <double> &part_params)
{
	//Unpack parameters
	const double D = part_params[0];
	const double g_1 = part_params[1];
	const double g_2 = part_params[2];
	const double K_A_B_1 = part_params[3];
	const double K_A_I_1 = part_params[4];
	const double K_mu_glu = part_params[5];
	const double k_omega_B_1 = part_params[6];
	const double kA_1 = part_params[7];
	const double kB_max_1 = part_params[8];
	const double kI_max_1 = part_params[9];
	const double mu_max_1 = part_params[10];
	const double mu_max_2 = part_params[11];
	const double n_A_B_1 = part_params[12];
	const double n_A_I_1 = part_params[13];
	const double S0_glu = part_params[14];

	J( 0 , 0 ) = -D + y[2]*mu_max_1/(K_mu_glu + y[2]);
	J( 0 , 1 ) = 0;
	J( 0 , 2 ) = -y[0]*y[2]*mu_max_1/std::pow(K_mu_glu + y[2], 2) + y[0]*mu_max_1/(K_mu_glu + y[2]);
	J( 0 , 3 ) = 0;
	J( 0 , 4 ) = 0;
	J( 0 , 5 ) = 0;
	J( 1 , 0 ) = 0;
	J( 1 , 1 ) = -y[3]*k_omega_B_1 - D + y[2]*mu_max_2/(K_mu_glu + y[2]);
	J( 1 , 2 ) = -y[1]*y[2]*mu_max_2/std::pow(K_mu_glu + y[2], 2) + y[1]*mu_max_2/(K_mu_glu + y[2]);
	J( 1 , 3 ) = -y[1]*k_omega_B_1;
	J( 1 , 4 ) = 0;
	J( 1 , 5 ) = 0;
	J( 2 , 0 ) = -y[2]*mu_max_1/(g_1*(K_mu_glu + y[2]));
	J( 2 , 1 ) = -y[2]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	J( 2 , 2 ) = -D + y[0]*y[2]*mu_max_1/(g_1*std::pow(K_mu_glu + y[2], 2)) - y[0]*mu_max_1/(g_1*(K_mu_glu + y[2])) + y[1]*y[2]*mu_max_2/(g_2*std::pow(K_mu_glu + y[2], 2)) - y[1]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	J( 2 , 3 ) = 0;
	J( 2 , 4 ) = 0;
	J( 2 , 5 ) = 0;
	J( 3 , 0 ) = std::pow(y[4], n_A_B_1)*kB_max_1/(std::pow(y[4], n_A_B_1) + std::pow(K_A_B_1, n_A_B_1));
	J( 3 , 1 ) = 0;
	J( 3 , 2 ) = 0;
	J( 3 , 3 ) = -D;
	J( 3 , 4 ) = -std::pow(y[4], 2*n_A_B_1)*y[0]*kB_max_1*n_A_B_1/(y[4]*std::pow(std::pow(y[4], n_A_B_1) + std::pow(K_A_B_1, n_A_B_1), 2)) + std::pow(y[4], n_A_B_1)*y[0]*kB_max_1*n_A_B_1/(y[4]*(std::pow(y[4], n_A_B_1) + std::pow(K_A_B_1, n_A_B_1)));
	J( 3 , 5 ) = 0;
	J( 4 , 0 ) = kA_1;
	J( 4 , 1 ) = 0;
	J( 4 , 2 ) = 0;
	J( 4 , 3 ) = 0;
	J( 4 , 4 ) = -D;
	J( 4 , 5 ) = 0;
	J( 5 , 0 ) = 0;
	J( 5 , 1 ) = 0;
	J( 5 , 2 ) = (1.0/2.0)*y[5]*y[2]*mu_max_1/std::pow(K_mu_glu + y[2], 2) - 1.0/2.0*y[5]*mu_max_1/(K_mu_glu + y[2]);
	J( 5 , 3 ) = 0;
	J( 5 , 4 ) = -std::pow(y[4], 2*n_A_I_1)*kI_max_1*n_A_I_1/(y[4]*std::pow(std::pow(y[4], n_A_I_1) + std::pow(K_A_I_1, n_A_I_1), 2)) + std::pow(y[4], n_A_I_1)*kI_max_1*n_A_I_1/(y[4]*(std::pow(y[4], n_A_I_1) + std::pow(K_A_I_1, n_A_I_1)));
	J( 5 , 5 ) = -1.0/2.0*y[2]*mu_max_1/(K_mu_glu + y[2]);

	dfdt[0] = 0.0;
	dfdt[1] = 0.0;
	dfdt[2] = 0.0;
	dfdt[3] = 0.0;
	dfdt[4] = 0.0;
	dfdt[5] = 0.0;

}
void Models::model_9(const ublas_vec_t  &y , ublas_vec_t &dydt , double t, std::vector <double> &part_params)
{
	//Unpack parameters
	const double D = part_params[0];
	const double g_1 = part_params[1];
	const double g_2 = part_params[2];
	const double K_A_B_1 = part_params[3];
	const double K_A_I_1 = part_params[4];
	const double K_mu_glu = part_params[5];
	const double k_omega_B_1 = part_params[6];
	const double kA_1 = part_params[7];
	const double kB_max_1 = part_params[8];
	const double kI_max_1 = part_params[9];
	const double mu_max_1 = part_params[10];
	const double mu_max_2 = part_params[11];
	const double n_A_B_1 = part_params[12];
	const double n_A_I_1 = part_params[13];
	const double S0_glu = part_params[14];

	//Species order is: N_1 N_2 S_glu B_1 A_1 I_1 
	dydt[0] = -D*y[0] + y[0]*y[2]*mu_max_1/(K_mu_glu + y[2]);
	dydt[1] = -y[3]*y[1]*k_omega_B_1 - D*y[1] + y[1]*y[2]*mu_max_2/(K_mu_glu + y[2]);
	dydt[2] = D*(S0_glu - y[2]) - y[0]*y[2]*mu_max_1/(g_1*(K_mu_glu + y[2])) - y[1]*y[2]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	dydt[3] = std::pow(y[4], n_A_B_1)*y[0]*kB_max_1/(std::pow(y[4], n_A_B_1) + std::pow(K_A_B_1, n_A_B_1)) - y[3]*D;
	dydt[4] = -y[4]*D + y[0]*kA_1;
	dydt[5] = -1.0/2.0*y[5]*y[2]*mu_max_1/(K_mu_glu + y[2]) + std::pow(K_A_I_1, n_A_I_1)*kI_max_1/(std::pow(y[4], n_A_I_1) + std::pow(K_A_I_1, n_A_I_1));

}
void Models::jac_9(const ublas_vec_t & y , ublas_mat_t &J , const double &/* t*/ , ublas_vec_t &dfdt, std::vector <double> &part_params)
{
	//Unpack parameters
	const double D = part_params[0];
	const double g_1 = part_params[1];
	const double g_2 = part_params[2];
	const double K_A_B_1 = part_params[3];
	const double K_A_I_1 = part_params[4];
	const double K_mu_glu = part_params[5];
	const double k_omega_B_1 = part_params[6];
	const double kA_1 = part_params[7];
	const double kB_max_1 = part_params[8];
	const double kI_max_1 = part_params[9];
	const double mu_max_1 = part_params[10];
	const double mu_max_2 = part_params[11];
	const double n_A_B_1 = part_params[12];
	const double n_A_I_1 = part_params[13];
	const double S0_glu = part_params[14];

	J( 0 , 0 ) = -D + y[2]*mu_max_1/(K_mu_glu + y[2]);
	J( 0 , 1 ) = 0;
	J( 0 , 2 ) = -y[0]*y[2]*mu_max_1/std::pow(K_mu_glu + y[2], 2) + y[0]*mu_max_1/(K_mu_glu + y[2]);
	J( 0 , 3 ) = 0;
	J( 0 , 4 ) = 0;
	J( 0 , 5 ) = 0;
	J( 1 , 0 ) = 0;
	J( 1 , 1 ) = -y[3]*k_omega_B_1 - D + y[2]*mu_max_2/(K_mu_glu + y[2]);
	J( 1 , 2 ) = -y[1]*y[2]*mu_max_2/std::pow(K_mu_glu + y[2], 2) + y[1]*mu_max_2/(K_mu_glu + y[2]);
	J( 1 , 3 ) = -y[1]*k_omega_B_1;
	J( 1 , 4 ) = 0;
	J( 1 , 5 ) = 0;
	J( 2 , 0 ) = -y[2]*mu_max_1/(g_1*(K_mu_glu + y[2]));
	J( 2 , 1 ) = -y[2]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	J( 2 , 2 ) = -D + y[0]*y[2]*mu_max_1/(g_1*std::pow(K_mu_glu + y[2], 2)) - y[0]*mu_max_1/(g_1*(K_mu_glu + y[2])) + y[1]*y[2]*mu_max_2/(g_2*std::pow(K_mu_glu + y[2], 2)) - y[1]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	J( 2 , 3 ) = 0;
	J( 2 , 4 ) = 0;
	J( 2 , 5 ) = 0;
	J( 3 , 0 ) = std::pow(y[4], n_A_B_1)*kB_max_1/(std::pow(y[4], n_A_B_1) + std::pow(K_A_B_1, n_A_B_1));
	J( 3 , 1 ) = 0;
	J( 3 , 2 ) = 0;
	J( 3 , 3 ) = -D;
	J( 3 , 4 ) = -std::pow(y[4], 2*n_A_B_1)*y[0]*kB_max_1*n_A_B_1/(y[4]*std::pow(std::pow(y[4], n_A_B_1) + std::pow(K_A_B_1, n_A_B_1), 2)) + std::pow(y[4], n_A_B_1)*y[0]*kB_max_1*n_A_B_1/(y[4]*(std::pow(y[4], n_A_B_1) + std::pow(K_A_B_1, n_A_B_1)));
	J( 3 , 5 ) = 0;
	J( 4 , 0 ) = kA_1;
	J( 4 , 1 ) = 0;
	J( 4 , 2 ) = 0;
	J( 4 , 3 ) = 0;
	J( 4 , 4 ) = -D;
	J( 4 , 5 ) = 0;
	J( 5 , 0 ) = 0;
	J( 5 , 1 ) = 0;
	J( 5 , 2 ) = (1.0/2.0)*y[5]*y[2]*mu_max_1/std::pow(K_mu_glu + y[2], 2) - 1.0/2.0*y[5]*mu_max_1/(K_mu_glu + y[2]);
	J( 5 , 3 ) = 0;
	J( 5 , 4 ) = -std::pow(y[4], n_A_I_1)*std::pow(K_A_I_1, n_A_I_1)*kI_max_1*n_A_I_1/(y[4]*std::pow(std::pow(y[4], n_A_I_1) + std::pow(K_A_I_1, n_A_I_1), 2));
	J( 5 , 5 ) = -1.0/2.0*y[2]*mu_max_1/(K_mu_glu + y[2]);

	dfdt[0] = 0.0;
	dfdt[1] = 0.0;
	dfdt[2] = 0.0;
	dfdt[3] = 0.0;
	dfdt[4] = 0.0;
	dfdt[5] = 0.0;

}
void Models::model_10(const ublas_vec_t  &y , ublas_vec_t &dydt , double t, std::vector <double> &part_params)
{
	//Unpack parameters
	const double D = part_params[0];
	const double g_1 = part_params[1];
	const double g_2 = part_params[2];
	const double K_A_B_1 = part_params[3];
	const double K_mu_glu = part_params[4];
	const double k_omega_B_1 = part_params[5];
	const double kA_1 = part_params[6];
	const double kB_max_1 = part_params[7];
	const double kI_max_1 = part_params[8];
	const double mu_max_1 = part_params[9];
	const double mu_max_2 = part_params[10];
	const double n_A_B_1 = part_params[11];
	const double S0_glu = part_params[12];

	//Species order is: N_1 N_2 S_glu B_1 A_1 I_1 
	dydt[0] = -D*y[0] + y[0]*y[2]*mu_max_1/(K_mu_glu + y[2]);
	dydt[1] = -y[3]*y[1]*k_omega_B_1 - D*y[1] + y[1]*y[2]*mu_max_2/(K_mu_glu + y[2]);
	dydt[2] = D*(S0_glu - y[2]) - y[0]*y[2]*mu_max_1/(g_1*(K_mu_glu + y[2])) - y[1]*y[2]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	dydt[3] = std::pow(y[4], n_A_B_1)*y[0]*kB_max_1/(std::pow(y[4], n_A_B_1) + std::pow(K_A_B_1, n_A_B_1)) - y[3]*D;
	dydt[4] = -y[4]*D + y[0]*kA_1;
	dydt[5] = -1.0/2.0*y[5]*y[2]*mu_max_1/(K_mu_glu + y[2]) + kI_max_1;

}
void Models::jac_10(const ublas_vec_t & y , ublas_mat_t &J , const double &/* t*/ , ublas_vec_t &dfdt, std::vector <double> &part_params)
{
	//Unpack parameters
	const double D = part_params[0];
	const double g_1 = part_params[1];
	const double g_2 = part_params[2];
	const double K_A_B_1 = part_params[3];
	const double K_mu_glu = part_params[4];
	const double k_omega_B_1 = part_params[5];
	const double kA_1 = part_params[6];
	const double kB_max_1 = part_params[7];
	const double kI_max_1 = part_params[8];
	const double mu_max_1 = part_params[9];
	const double mu_max_2 = part_params[10];
	const double n_A_B_1 = part_params[11];
	const double S0_glu = part_params[12];

	J( 0 , 0 ) = -D + y[2]*mu_max_1/(K_mu_glu + y[2]);
	J( 0 , 1 ) = 0;
	J( 0 , 2 ) = -y[0]*y[2]*mu_max_1/std::pow(K_mu_glu + y[2], 2) + y[0]*mu_max_1/(K_mu_glu + y[2]);
	J( 0 , 3 ) = 0;
	J( 0 , 4 ) = 0;
	J( 0 , 5 ) = 0;
	J( 1 , 0 ) = 0;
	J( 1 , 1 ) = -y[3]*k_omega_B_1 - D + y[2]*mu_max_2/(K_mu_glu + y[2]);
	J( 1 , 2 ) = -y[1]*y[2]*mu_max_2/std::pow(K_mu_glu + y[2], 2) + y[1]*mu_max_2/(K_mu_glu + y[2]);
	J( 1 , 3 ) = -y[1]*k_omega_B_1;
	J( 1 , 4 ) = 0;
	J( 1 , 5 ) = 0;
	J( 2 , 0 ) = -y[2]*mu_max_1/(g_1*(K_mu_glu + y[2]));
	J( 2 , 1 ) = -y[2]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	J( 2 , 2 ) = -D + y[0]*y[2]*mu_max_1/(g_1*std::pow(K_mu_glu + y[2], 2)) - y[0]*mu_max_1/(g_1*(K_mu_glu + y[2])) + y[1]*y[2]*mu_max_2/(g_2*std::pow(K_mu_glu + y[2], 2)) - y[1]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	J( 2 , 3 ) = 0;
	J( 2 , 4 ) = 0;
	J( 2 , 5 ) = 0;
	J( 3 , 0 ) = std::pow(y[4], n_A_B_1)*kB_max_1/(std::pow(y[4], n_A_B_1) + std::pow(K_A_B_1, n_A_B_1));
	J( 3 , 1 ) = 0;
	J( 3 , 2 ) = 0;
	J( 3 , 3 ) = -D;
	J( 3 , 4 ) = -std::pow(y[4], 2*n_A_B_1)*y[0]*kB_max_1*n_A_B_1/(y[4]*std::pow(std::pow(y[4], n_A_B_1) + std::pow(K_A_B_1, n_A_B_1), 2)) + std::pow(y[4], n_A_B_1)*y[0]*kB_max_1*n_A_B_1/(y[4]*(std::pow(y[4], n_A_B_1) + std::pow(K_A_B_1, n_A_B_1)));
	J( 3 , 5 ) = 0;
	J( 4 , 0 ) = kA_1;
	J( 4 , 1 ) = 0;
	J( 4 , 2 ) = 0;
	J( 4 , 3 ) = 0;
	J( 4 , 4 ) = -D;
	J( 4 , 5 ) = 0;
	J( 5 , 0 ) = 0;
	J( 5 , 1 ) = 0;
	J( 5 , 2 ) = (1.0/2.0)*y[5]*y[2]*mu_max_1/std::pow(K_mu_glu + y[2], 2) - 1.0/2.0*y[5]*mu_max_1/(K_mu_glu + y[2]);
	J( 5 , 3 ) = 0;
	J( 5 , 4 ) = 0;
	J( 5 , 5 ) = -1.0/2.0*y[2]*mu_max_1/(K_mu_glu + y[2]);

	dfdt[0] = 0.0;
	dfdt[1] = 0.0;
	dfdt[2] = 0.0;
	dfdt[3] = 0.0;
	dfdt[4] = 0.0;
	dfdt[5] = 0.0;

}
void Models::model_11(const ublas_vec_t  &y , ublas_vec_t &dydt , double t, std::vector <double> &part_params)
{
	//Unpack parameters
	const double D = part_params[0];
	const double g_1 = part_params[1];
	const double g_2 = part_params[2];
	const double K_A_B_1 = part_params[3];
	const double K_mu_glu = part_params[4];
	const double k_omega_B_1 = part_params[5];
	const double kA_1 = part_params[6];
	const double kB_max_1 = part_params[7];
	const double mu_max_1 = part_params[8];
	const double mu_max_2 = part_params[9];
	const double n_A_B_1 = part_params[10];
	const double S0_glu = part_params[11];

	//Species order is: N_1 N_2 S_glu B_1 A_1 
	dydt[0] = -D*y[0] + y[0]*y[2]*mu_max_1/(K_mu_glu + y[2]);
	dydt[1] = -y[3]*y[1]*k_omega_B_1 - D*y[1] + y[1]*y[2]*mu_max_2/(K_mu_glu + y[2]);
	dydt[2] = D*(S0_glu - y[2]) - y[0]*y[2]*mu_max_1/(g_1*(K_mu_glu + y[2])) - y[1]*y[2]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	dydt[3] = std::pow(y[4], n_A_B_1)*y[0]*kB_max_1/(std::pow(y[4], n_A_B_1) + std::pow(K_A_B_1, n_A_B_1)) - y[3]*D;
	dydt[4] = -y[4]*D + y[0]*kA_1;

}
void Models::jac_11(const ublas_vec_t & y , ublas_mat_t &J , const double &/* t*/ , ublas_vec_t &dfdt, std::vector <double> &part_params)
{
	//Unpack parameters
	const double D = part_params[0];
	const double g_1 = part_params[1];
	const double g_2 = part_params[2];
	const double K_A_B_1 = part_params[3];
	const double K_mu_glu = part_params[4];
	const double k_omega_B_1 = part_params[5];
	const double kA_1 = part_params[6];
	const double kB_max_1 = part_params[7];
	const double mu_max_1 = part_params[8];
	const double mu_max_2 = part_params[9];
	const double n_A_B_1 = part_params[10];
	const double S0_glu = part_params[11];

	J( 0 , 0 ) = -D + y[2]*mu_max_1/(K_mu_glu + y[2]);
	J( 0 , 1 ) = 0;
	J( 0 , 2 ) = -y[0]*y[2]*mu_max_1/std::pow(K_mu_glu + y[2], 2) + y[0]*mu_max_1/(K_mu_glu + y[2]);
	J( 0 , 3 ) = 0;
	J( 0 , 4 ) = 0;
	J( 1 , 0 ) = 0;
	J( 1 , 1 ) = -y[3]*k_omega_B_1 - D + y[2]*mu_max_2/(K_mu_glu + y[2]);
	J( 1 , 2 ) = -y[1]*y[2]*mu_max_2/std::pow(K_mu_glu + y[2], 2) + y[1]*mu_max_2/(K_mu_glu + y[2]);
	J( 1 , 3 ) = -y[1]*k_omega_B_1;
	J( 1 , 4 ) = 0;
	J( 2 , 0 ) = -y[2]*mu_max_1/(g_1*(K_mu_glu + y[2]));
	J( 2 , 1 ) = -y[2]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	J( 2 , 2 ) = -D + y[0]*y[2]*mu_max_1/(g_1*std::pow(K_mu_glu + y[2], 2)) - y[0]*mu_max_1/(g_1*(K_mu_glu + y[2])) + y[1]*y[2]*mu_max_2/(g_2*std::pow(K_mu_glu + y[2], 2)) - y[1]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	J( 2 , 3 ) = 0;
	J( 2 , 4 ) = 0;
	J( 3 , 0 ) = std::pow(y[4], n_A_B_1)*kB_max_1/(std::pow(y[4], n_A_B_1) + std::pow(K_A_B_1, n_A_B_1));
	J( 3 , 1 ) = 0;
	J( 3 , 2 ) = 0;
	J( 3 , 3 ) = -D;
	J( 3 , 4 ) = -std::pow(y[4], 2*n_A_B_1)*y[0]*kB_max_1*n_A_B_1/(y[4]*std::pow(std::pow(y[4], n_A_B_1) + std::pow(K_A_B_1, n_A_B_1), 2)) + std::pow(y[4], n_A_B_1)*y[0]*kB_max_1*n_A_B_1/(y[4]*(std::pow(y[4], n_A_B_1) + std::pow(K_A_B_1, n_A_B_1)));
	J( 4 , 0 ) = kA_1;
	J( 4 , 1 ) = 0;
	J( 4 , 2 ) = 0;
	J( 4 , 3 ) = 0;
	J( 4 , 4 ) = -D;

	dfdt[0] = 0.0;
	dfdt[1] = 0.0;
	dfdt[2] = 0.0;
	dfdt[3] = 0.0;
	dfdt[4] = 0.0;

}
void Models::model_12(const ublas_vec_t  &y , ublas_vec_t &dydt , double t, std::vector <double> &part_params)
{
	//Unpack parameters
	const double D = part_params[0];
	const double g_1 = part_params[1];
	const double g_2 = part_params[2];
	const double K_A_B_1 = part_params[3];
	const double K_A_I_1 = part_params[4];
	const double k_I_1 = part_params[5];
	const double K_mu_glu = part_params[6];
	const double k_omega_B_1 = part_params[7];
	const double kA_1 = part_params[8];
	const double kB_max_1 = part_params[9];
	const double kI_max_1 = part_params[10];
	const double mu_max_1 = part_params[11];
	const double mu_max_2 = part_params[12];
	const double n_A_B_1 = part_params[13];
	const double n_A_I_1 = part_params[14];
	const double nI_1 = part_params[15];
	const double S0_glu = part_params[16];

	//Species order is: N_1 N_2 S_glu B_1 A_1 I_1 
	dydt[0] = -y[3]*y[0]*std::pow(k_I_1, nI_1)*k_omega_B_1/(std::pow(y[5], nI_1) + std::pow(k_I_1, nI_1)) - D*y[0] + y[0]*y[2]*mu_max_1/(K_mu_glu + y[2]);
	dydt[1] = -y[3]*y[1]*k_omega_B_1 - D*y[1] + y[1]*y[2]*mu_max_2/(K_mu_glu + y[2]);
	dydt[2] = D*(S0_glu - y[2]) - y[0]*y[2]*mu_max_1/(g_1*(K_mu_glu + y[2])) - y[1]*y[2]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	dydt[3] = -y[3]*D + std::pow(K_A_B_1, n_A_B_1)*y[0]*kB_max_1/(std::pow(y[4], n_A_B_1) + std::pow(K_A_B_1, n_A_B_1));
	dydt[4] = -y[4]*D + y[0]*kA_1;
	dydt[5] = std::pow(y[4], n_A_I_1)*kI_max_1/(std::pow(y[4], n_A_I_1) + std::pow(K_A_I_1, n_A_I_1)) - 1.0/2.0*y[5]*y[2]*mu_max_1/(K_mu_glu + y[2]);

}
void Models::jac_12(const ublas_vec_t & y , ublas_mat_t &J , const double &/* t*/ , ublas_vec_t &dfdt, std::vector <double> &part_params)
{
	//Unpack parameters
	const double D = part_params[0];
	const double g_1 = part_params[1];
	const double g_2 = part_params[2];
	const double K_A_B_1 = part_params[3];
	const double K_A_I_1 = part_params[4];
	const double k_I_1 = part_params[5];
	const double K_mu_glu = part_params[6];
	const double k_omega_B_1 = part_params[7];
	const double kA_1 = part_params[8];
	const double kB_max_1 = part_params[9];
	const double kI_max_1 = part_params[10];
	const double mu_max_1 = part_params[11];
	const double mu_max_2 = part_params[12];
	const double n_A_B_1 = part_params[13];
	const double n_A_I_1 = part_params[14];
	const double nI_1 = part_params[15];
	const double S0_glu = part_params[16];

	J( 0 , 0 ) = -y[3]*std::pow(k_I_1, nI_1)*k_omega_B_1/(std::pow(y[5], nI_1) + std::pow(k_I_1, nI_1)) - D + y[2]*mu_max_1/(K_mu_glu + y[2]);
	J( 0 , 1 ) = 0;
	J( 0 , 2 ) = -y[0]*y[2]*mu_max_1/std::pow(K_mu_glu + y[2], 2) + y[0]*mu_max_1/(K_mu_glu + y[2]);
	J( 0 , 3 ) = -y[0]*std::pow(k_I_1, nI_1)*k_omega_B_1/(std::pow(y[5], nI_1) + std::pow(k_I_1, nI_1));
	J( 0 , 4 ) = 0;
	J( 0 , 5 ) = y[3]*std::pow(y[5], nI_1)*y[0]*std::pow(k_I_1, nI_1)*k_omega_B_1*nI_1/(y[5]*std::pow(std::pow(y[5], nI_1) + std::pow(k_I_1, nI_1), 2));
	J( 1 , 0 ) = 0;
	J( 1 , 1 ) = -y[3]*k_omega_B_1 - D + y[2]*mu_max_2/(K_mu_glu + y[2]);
	J( 1 , 2 ) = -y[1]*y[2]*mu_max_2/std::pow(K_mu_glu + y[2], 2) + y[1]*mu_max_2/(K_mu_glu + y[2]);
	J( 1 , 3 ) = -y[1]*k_omega_B_1;
	J( 1 , 4 ) = 0;
	J( 1 , 5 ) = 0;
	J( 2 , 0 ) = -y[2]*mu_max_1/(g_1*(K_mu_glu + y[2]));
	J( 2 , 1 ) = -y[2]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	J( 2 , 2 ) = -D + y[0]*y[2]*mu_max_1/(g_1*std::pow(K_mu_glu + y[2], 2)) - y[0]*mu_max_1/(g_1*(K_mu_glu + y[2])) + y[1]*y[2]*mu_max_2/(g_2*std::pow(K_mu_glu + y[2], 2)) - y[1]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	J( 2 , 3 ) = 0;
	J( 2 , 4 ) = 0;
	J( 2 , 5 ) = 0;
	J( 3 , 0 ) = std::pow(K_A_B_1, n_A_B_1)*kB_max_1/(std::pow(y[4], n_A_B_1) + std::pow(K_A_B_1, n_A_B_1));
	J( 3 , 1 ) = 0;
	J( 3 , 2 ) = 0;
	J( 3 , 3 ) = -D;
	J( 3 , 4 ) = -std::pow(y[4], n_A_B_1)*std::pow(K_A_B_1, n_A_B_1)*y[0]*kB_max_1*n_A_B_1/(y[4]*std::pow(std::pow(y[4], n_A_B_1) + std::pow(K_A_B_1, n_A_B_1), 2));
	J( 3 , 5 ) = 0;
	J( 4 , 0 ) = kA_1;
	J( 4 , 1 ) = 0;
	J( 4 , 2 ) = 0;
	J( 4 , 3 ) = 0;
	J( 4 , 4 ) = -D;
	J( 4 , 5 ) = 0;
	J( 5 , 0 ) = 0;
	J( 5 , 1 ) = 0;
	J( 5 , 2 ) = (1.0/2.0)*y[5]*y[2]*mu_max_1/std::pow(K_mu_glu + y[2], 2) - 1.0/2.0*y[5]*mu_max_1/(K_mu_glu + y[2]);
	J( 5 , 3 ) = 0;
	J( 5 , 4 ) = -std::pow(y[4], 2*n_A_I_1)*kI_max_1*n_A_I_1/(y[4]*std::pow(std::pow(y[4], n_A_I_1) + std::pow(K_A_I_1, n_A_I_1), 2)) + std::pow(y[4], n_A_I_1)*kI_max_1*n_A_I_1/(y[4]*(std::pow(y[4], n_A_I_1) + std::pow(K_A_I_1, n_A_I_1)));
	J( 5 , 5 ) = -1.0/2.0*y[2]*mu_max_1/(K_mu_glu + y[2]);

	dfdt[0] = 0.0;
	dfdt[1] = 0.0;
	dfdt[2] = 0.0;
	dfdt[3] = 0.0;
	dfdt[4] = 0.0;
	dfdt[5] = 0.0;

}
void Models::model_13(const ublas_vec_t  &y , ublas_vec_t &dydt , double t, std::vector <double> &part_params)
{
	//Unpack parameters
	const double D = part_params[0];
	const double g_1 = part_params[1];
	const double g_2 = part_params[2];
	const double K_A_B_1 = part_params[3];
	const double K_A_I_1 = part_params[4];
	const double k_I_1 = part_params[5];
	const double K_mu_glu = part_params[6];
	const double k_omega_B_1 = part_params[7];
	const double kA_1 = part_params[8];
	const double kB_max_1 = part_params[9];
	const double kI_max_1 = part_params[10];
	const double mu_max_1 = part_params[11];
	const double mu_max_2 = part_params[12];
	const double n_A_B_1 = part_params[13];
	const double n_A_I_1 = part_params[14];
	const double nI_1 = part_params[15];
	const double S0_glu = part_params[16];

	//Species order is: N_1 N_2 S_glu B_1 A_1 I_1 
	dydt[0] = -y[3]*y[0]*std::pow(k_I_1, nI_1)*k_omega_B_1/(std::pow(y[5], nI_1) + std::pow(k_I_1, nI_1)) - D*y[0] + y[0]*y[2]*mu_max_1/(K_mu_glu + y[2]);
	dydt[1] = -D*y[1] + y[1]*y[2]*mu_max_2/(K_mu_glu + y[2]);
	dydt[2] = D*(S0_glu - y[2]) - y[0]*y[2]*mu_max_1/(g_1*(K_mu_glu + y[2])) - y[1]*y[2]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	dydt[3] = -y[3]*D + std::pow(K_A_B_1, n_A_B_1)*y[0]*kB_max_1/(std::pow(y[4], n_A_B_1) + std::pow(K_A_B_1, n_A_B_1));
	dydt[4] = -y[4]*D + y[0]*kA_1;
	dydt[5] = std::pow(y[4], n_A_I_1)*kI_max_1/(std::pow(y[4], n_A_I_1) + std::pow(K_A_I_1, n_A_I_1)) - 1.0/2.0*y[5]*y[2]*mu_max_1/(K_mu_glu + y[2]);

}
void Models::jac_13(const ublas_vec_t & y , ublas_mat_t &J , const double &/* t*/ , ublas_vec_t &dfdt, std::vector <double> &part_params)
{
	//Unpack parameters
	const double D = part_params[0];
	const double g_1 = part_params[1];
	const double g_2 = part_params[2];
	const double K_A_B_1 = part_params[3];
	const double K_A_I_1 = part_params[4];
	const double k_I_1 = part_params[5];
	const double K_mu_glu = part_params[6];
	const double k_omega_B_1 = part_params[7];
	const double kA_1 = part_params[8];
	const double kB_max_1 = part_params[9];
	const double kI_max_1 = part_params[10];
	const double mu_max_1 = part_params[11];
	const double mu_max_2 = part_params[12];
	const double n_A_B_1 = part_params[13];
	const double n_A_I_1 = part_params[14];
	const double nI_1 = part_params[15];
	const double S0_glu = part_params[16];

	J( 0 , 0 ) = -y[3]*std::pow(k_I_1, nI_1)*k_omega_B_1/(std::pow(y[5], nI_1) + std::pow(k_I_1, nI_1)) - D + y[2]*mu_max_1/(K_mu_glu + y[2]);
	J( 0 , 1 ) = 0;
	J( 0 , 2 ) = -y[0]*y[2]*mu_max_1/std::pow(K_mu_glu + y[2], 2) + y[0]*mu_max_1/(K_mu_glu + y[2]);
	J( 0 , 3 ) = -y[0]*std::pow(k_I_1, nI_1)*k_omega_B_1/(std::pow(y[5], nI_1) + std::pow(k_I_1, nI_1));
	J( 0 , 4 ) = 0;
	J( 0 , 5 ) = y[3]*std::pow(y[5], nI_1)*y[0]*std::pow(k_I_1, nI_1)*k_omega_B_1*nI_1/(y[5]*std::pow(std::pow(y[5], nI_1) + std::pow(k_I_1, nI_1), 2));
	J( 1 , 0 ) = 0;
	J( 1 , 1 ) = -D + y[2]*mu_max_2/(K_mu_glu + y[2]);
	J( 1 , 2 ) = -y[1]*y[2]*mu_max_2/std::pow(K_mu_glu + y[2], 2) + y[1]*mu_max_2/(K_mu_glu + y[2]);
	J( 1 , 3 ) = 0;
	J( 1 , 4 ) = 0;
	J( 1 , 5 ) = 0;
	J( 2 , 0 ) = -y[2]*mu_max_1/(g_1*(K_mu_glu + y[2]));
	J( 2 , 1 ) = -y[2]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	J( 2 , 2 ) = -D + y[0]*y[2]*mu_max_1/(g_1*std::pow(K_mu_glu + y[2], 2)) - y[0]*mu_max_1/(g_1*(K_mu_glu + y[2])) + y[1]*y[2]*mu_max_2/(g_2*std::pow(K_mu_glu + y[2], 2)) - y[1]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	J( 2 , 3 ) = 0;
	J( 2 , 4 ) = 0;
	J( 2 , 5 ) = 0;
	J( 3 , 0 ) = std::pow(K_A_B_1, n_A_B_1)*kB_max_1/(std::pow(y[4], n_A_B_1) + std::pow(K_A_B_1, n_A_B_1));
	J( 3 , 1 ) = 0;
	J( 3 , 2 ) = 0;
	J( 3 , 3 ) = -D;
	J( 3 , 4 ) = -std::pow(y[4], n_A_B_1)*std::pow(K_A_B_1, n_A_B_1)*y[0]*kB_max_1*n_A_B_1/(y[4]*std::pow(std::pow(y[4], n_A_B_1) + std::pow(K_A_B_1, n_A_B_1), 2));
	J( 3 , 5 ) = 0;
	J( 4 , 0 ) = kA_1;
	J( 4 , 1 ) = 0;
	J( 4 , 2 ) = 0;
	J( 4 , 3 ) = 0;
	J( 4 , 4 ) = -D;
	J( 4 , 5 ) = 0;
	J( 5 , 0 ) = 0;
	J( 5 , 1 ) = 0;
	J( 5 , 2 ) = (1.0/2.0)*y[5]*y[2]*mu_max_1/std::pow(K_mu_glu + y[2], 2) - 1.0/2.0*y[5]*mu_max_1/(K_mu_glu + y[2]);
	J( 5 , 3 ) = 0;
	J( 5 , 4 ) = -std::pow(y[4], 2*n_A_I_1)*kI_max_1*n_A_I_1/(y[4]*std::pow(std::pow(y[4], n_A_I_1) + std::pow(K_A_I_1, n_A_I_1), 2)) + std::pow(y[4], n_A_I_1)*kI_max_1*n_A_I_1/(y[4]*(std::pow(y[4], n_A_I_1) + std::pow(K_A_I_1, n_A_I_1)));
	J( 5 , 5 ) = -1.0/2.0*y[2]*mu_max_1/(K_mu_glu + y[2]);

	dfdt[0] = 0.0;
	dfdt[1] = 0.0;
	dfdt[2] = 0.0;
	dfdt[3] = 0.0;
	dfdt[4] = 0.0;
	dfdt[5] = 0.0;

}
void Models::model_14(const ublas_vec_t  &y , ublas_vec_t &dydt , double t, std::vector <double> &part_params)
{
	//Unpack parameters
	const double D = part_params[0];
	const double g_1 = part_params[1];
	const double g_2 = part_params[2];
	const double K_A_B_1 = part_params[3];
	const double K_A_I_1 = part_params[4];
	const double k_I_1 = part_params[5];
	const double K_mu_glu = part_params[6];
	const double k_omega_B_1 = part_params[7];
	const double kA_1 = part_params[8];
	const double kB_max_1 = part_params[9];
	const double kI_max_1 = part_params[10];
	const double mu_max_1 = part_params[11];
	const double mu_max_2 = part_params[12];
	const double n_A_B_1 = part_params[13];
	const double n_A_I_1 = part_params[14];
	const double nI_1 = part_params[15];
	const double S0_glu = part_params[16];

	//Species order is: N_1 N_2 S_glu B_1 A_1 I_1 
	dydt[0] = -y[3]*y[0]*std::pow(k_I_1, nI_1)*k_omega_B_1/(std::pow(y[5], nI_1) + std::pow(k_I_1, nI_1)) - D*y[0] + y[0]*y[2]*mu_max_1/(K_mu_glu + y[2]);
	dydt[1] = -y[3]*y[1]*k_omega_B_1 - D*y[1] + y[1]*y[2]*mu_max_2/(K_mu_glu + y[2]);
	dydt[2] = D*(S0_glu - y[2]) - y[0]*y[2]*mu_max_1/(g_1*(K_mu_glu + y[2])) - y[1]*y[2]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	dydt[3] = -y[3]*D + std::pow(K_A_B_1, n_A_B_1)*y[0]*kB_max_1/(std::pow(y[4], n_A_B_1) + std::pow(K_A_B_1, n_A_B_1));
	dydt[4] = -y[4]*D + y[0]*kA_1;
	dydt[5] = -1.0/2.0*y[5]*y[2]*mu_max_1/(K_mu_glu + y[2]) + std::pow(K_A_I_1, n_A_I_1)*kI_max_1/(std::pow(y[4], n_A_I_1) + std::pow(K_A_I_1, n_A_I_1));

}
void Models::jac_14(const ublas_vec_t & y , ublas_mat_t &J , const double &/* t*/ , ublas_vec_t &dfdt, std::vector <double> &part_params)
{
	//Unpack parameters
	const double D = part_params[0];
	const double g_1 = part_params[1];
	const double g_2 = part_params[2];
	const double K_A_B_1 = part_params[3];
	const double K_A_I_1 = part_params[4];
	const double k_I_1 = part_params[5];
	const double K_mu_glu = part_params[6];
	const double k_omega_B_1 = part_params[7];
	const double kA_1 = part_params[8];
	const double kB_max_1 = part_params[9];
	const double kI_max_1 = part_params[10];
	const double mu_max_1 = part_params[11];
	const double mu_max_2 = part_params[12];
	const double n_A_B_1 = part_params[13];
	const double n_A_I_1 = part_params[14];
	const double nI_1 = part_params[15];
	const double S0_glu = part_params[16];

	J( 0 , 0 ) = -y[3]*std::pow(k_I_1, nI_1)*k_omega_B_1/(std::pow(y[5], nI_1) + std::pow(k_I_1, nI_1)) - D + y[2]*mu_max_1/(K_mu_glu + y[2]);
	J( 0 , 1 ) = 0;
	J( 0 , 2 ) = -y[0]*y[2]*mu_max_1/std::pow(K_mu_glu + y[2], 2) + y[0]*mu_max_1/(K_mu_glu + y[2]);
	J( 0 , 3 ) = -y[0]*std::pow(k_I_1, nI_1)*k_omega_B_1/(std::pow(y[5], nI_1) + std::pow(k_I_1, nI_1));
	J( 0 , 4 ) = 0;
	J( 0 , 5 ) = y[3]*std::pow(y[5], nI_1)*y[0]*std::pow(k_I_1, nI_1)*k_omega_B_1*nI_1/(y[5]*std::pow(std::pow(y[5], nI_1) + std::pow(k_I_1, nI_1), 2));
	J( 1 , 0 ) = 0;
	J( 1 , 1 ) = -y[3]*k_omega_B_1 - D + y[2]*mu_max_2/(K_mu_glu + y[2]);
	J( 1 , 2 ) = -y[1]*y[2]*mu_max_2/std::pow(K_mu_glu + y[2], 2) + y[1]*mu_max_2/(K_mu_glu + y[2]);
	J( 1 , 3 ) = -y[1]*k_omega_B_1;
	J( 1 , 4 ) = 0;
	J( 1 , 5 ) = 0;
	J( 2 , 0 ) = -y[2]*mu_max_1/(g_1*(K_mu_glu + y[2]));
	J( 2 , 1 ) = -y[2]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	J( 2 , 2 ) = -D + y[0]*y[2]*mu_max_1/(g_1*std::pow(K_mu_glu + y[2], 2)) - y[0]*mu_max_1/(g_1*(K_mu_glu + y[2])) + y[1]*y[2]*mu_max_2/(g_2*std::pow(K_mu_glu + y[2], 2)) - y[1]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	J( 2 , 3 ) = 0;
	J( 2 , 4 ) = 0;
	J( 2 , 5 ) = 0;
	J( 3 , 0 ) = std::pow(K_A_B_1, n_A_B_1)*kB_max_1/(std::pow(y[4], n_A_B_1) + std::pow(K_A_B_1, n_A_B_1));
	J( 3 , 1 ) = 0;
	J( 3 , 2 ) = 0;
	J( 3 , 3 ) = -D;
	J( 3 , 4 ) = -std::pow(y[4], n_A_B_1)*std::pow(K_A_B_1, n_A_B_1)*y[0]*kB_max_1*n_A_B_1/(y[4]*std::pow(std::pow(y[4], n_A_B_1) + std::pow(K_A_B_1, n_A_B_1), 2));
	J( 3 , 5 ) = 0;
	J( 4 , 0 ) = kA_1;
	J( 4 , 1 ) = 0;
	J( 4 , 2 ) = 0;
	J( 4 , 3 ) = 0;
	J( 4 , 4 ) = -D;
	J( 4 , 5 ) = 0;
	J( 5 , 0 ) = 0;
	J( 5 , 1 ) = 0;
	J( 5 , 2 ) = (1.0/2.0)*y[5]*y[2]*mu_max_1/std::pow(K_mu_glu + y[2], 2) - 1.0/2.0*y[5]*mu_max_1/(K_mu_glu + y[2]);
	J( 5 , 3 ) = 0;
	J( 5 , 4 ) = -std::pow(y[4], n_A_I_1)*std::pow(K_A_I_1, n_A_I_1)*kI_max_1*n_A_I_1/(y[4]*std::pow(std::pow(y[4], n_A_I_1) + std::pow(K_A_I_1, n_A_I_1), 2));
	J( 5 , 5 ) = -1.0/2.0*y[2]*mu_max_1/(K_mu_glu + y[2]);

	dfdt[0] = 0.0;
	dfdt[1] = 0.0;
	dfdt[2] = 0.0;
	dfdt[3] = 0.0;
	dfdt[4] = 0.0;
	dfdt[5] = 0.0;

}
void Models::model_15(const ublas_vec_t  &y , ublas_vec_t &dydt , double t, std::vector <double> &part_params)
{
	//Unpack parameters
	const double D = part_params[0];
	const double g_1 = part_params[1];
	const double g_2 = part_params[2];
	const double K_A_B_1 = part_params[3];
	const double K_A_I_1 = part_params[4];
	const double k_I_1 = part_params[5];
	const double K_mu_glu = part_params[6];
	const double k_omega_B_1 = part_params[7];
	const double kA_1 = part_params[8];
	const double kB_max_1 = part_params[9];
	const double kI_max_1 = part_params[10];
	const double mu_max_1 = part_params[11];
	const double mu_max_2 = part_params[12];
	const double n_A_B_1 = part_params[13];
	const double n_A_I_1 = part_params[14];
	const double nI_1 = part_params[15];
	const double S0_glu = part_params[16];

	//Species order is: N_1 N_2 S_glu B_1 A_1 I_1 
	dydt[0] = -y[3]*y[0]*std::pow(k_I_1, nI_1)*k_omega_B_1/(std::pow(y[5], nI_1) + std::pow(k_I_1, nI_1)) - D*y[0] + y[0]*y[2]*mu_max_1/(K_mu_glu + y[2]);
	dydt[1] = -D*y[1] + y[1]*y[2]*mu_max_2/(K_mu_glu + y[2]);
	dydt[2] = D*(S0_glu - y[2]) - y[0]*y[2]*mu_max_1/(g_1*(K_mu_glu + y[2])) - y[1]*y[2]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	dydt[3] = -y[3]*D + std::pow(K_A_B_1, n_A_B_1)*y[0]*kB_max_1/(std::pow(y[4], n_A_B_1) + std::pow(K_A_B_1, n_A_B_1));
	dydt[4] = -y[4]*D + y[0]*kA_1;
	dydt[5] = -1.0/2.0*y[5]*y[2]*mu_max_1/(K_mu_glu + y[2]) + std::pow(K_A_I_1, n_A_I_1)*kI_max_1/(std::pow(y[4], n_A_I_1) + std::pow(K_A_I_1, n_A_I_1));

}
void Models::jac_15(const ublas_vec_t & y , ublas_mat_t &J , const double &/* t*/ , ublas_vec_t &dfdt, std::vector <double> &part_params)
{
	//Unpack parameters
	const double D = part_params[0];
	const double g_1 = part_params[1];
	const double g_2 = part_params[2];
	const double K_A_B_1 = part_params[3];
	const double K_A_I_1 = part_params[4];
	const double k_I_1 = part_params[5];
	const double K_mu_glu = part_params[6];
	const double k_omega_B_1 = part_params[7];
	const double kA_1 = part_params[8];
	const double kB_max_1 = part_params[9];
	const double kI_max_1 = part_params[10];
	const double mu_max_1 = part_params[11];
	const double mu_max_2 = part_params[12];
	const double n_A_B_1 = part_params[13];
	const double n_A_I_1 = part_params[14];
	const double nI_1 = part_params[15];
	const double S0_glu = part_params[16];

	J( 0 , 0 ) = -y[3]*std::pow(k_I_1, nI_1)*k_omega_B_1/(std::pow(y[5], nI_1) + std::pow(k_I_1, nI_1)) - D + y[2]*mu_max_1/(K_mu_glu + y[2]);
	J( 0 , 1 ) = 0;
	J( 0 , 2 ) = -y[0]*y[2]*mu_max_1/std::pow(K_mu_glu + y[2], 2) + y[0]*mu_max_1/(K_mu_glu + y[2]);
	J( 0 , 3 ) = -y[0]*std::pow(k_I_1, nI_1)*k_omega_B_1/(std::pow(y[5], nI_1) + std::pow(k_I_1, nI_1));
	J( 0 , 4 ) = 0;
	J( 0 , 5 ) = y[3]*std::pow(y[5], nI_1)*y[0]*std::pow(k_I_1, nI_1)*k_omega_B_1*nI_1/(y[5]*std::pow(std::pow(y[5], nI_1) + std::pow(k_I_1, nI_1), 2));
	J( 1 , 0 ) = 0;
	J( 1 , 1 ) = -D + y[2]*mu_max_2/(K_mu_glu + y[2]);
	J( 1 , 2 ) = -y[1]*y[2]*mu_max_2/std::pow(K_mu_glu + y[2], 2) + y[1]*mu_max_2/(K_mu_glu + y[2]);
	J( 1 , 3 ) = 0;
	J( 1 , 4 ) = 0;
	J( 1 , 5 ) = 0;
	J( 2 , 0 ) = -y[2]*mu_max_1/(g_1*(K_mu_glu + y[2]));
	J( 2 , 1 ) = -y[2]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	J( 2 , 2 ) = -D + y[0]*y[2]*mu_max_1/(g_1*std::pow(K_mu_glu + y[2], 2)) - y[0]*mu_max_1/(g_1*(K_mu_glu + y[2])) + y[1]*y[2]*mu_max_2/(g_2*std::pow(K_mu_glu + y[2], 2)) - y[1]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	J( 2 , 3 ) = 0;
	J( 2 , 4 ) = 0;
	J( 2 , 5 ) = 0;
	J( 3 , 0 ) = std::pow(K_A_B_1, n_A_B_1)*kB_max_1/(std::pow(y[4], n_A_B_1) + std::pow(K_A_B_1, n_A_B_1));
	J( 3 , 1 ) = 0;
	J( 3 , 2 ) = 0;
	J( 3 , 3 ) = -D;
	J( 3 , 4 ) = -std::pow(y[4], n_A_B_1)*std::pow(K_A_B_1, n_A_B_1)*y[0]*kB_max_1*n_A_B_1/(y[4]*std::pow(std::pow(y[4], n_A_B_1) + std::pow(K_A_B_1, n_A_B_1), 2));
	J( 3 , 5 ) = 0;
	J( 4 , 0 ) = kA_1;
	J( 4 , 1 ) = 0;
	J( 4 , 2 ) = 0;
	J( 4 , 3 ) = 0;
	J( 4 , 4 ) = -D;
	J( 4 , 5 ) = 0;
	J( 5 , 0 ) = 0;
	J( 5 , 1 ) = 0;
	J( 5 , 2 ) = (1.0/2.0)*y[5]*y[2]*mu_max_1/std::pow(K_mu_glu + y[2], 2) - 1.0/2.0*y[5]*mu_max_1/(K_mu_glu + y[2]);
	J( 5 , 3 ) = 0;
	J( 5 , 4 ) = -std::pow(y[4], n_A_I_1)*std::pow(K_A_I_1, n_A_I_1)*kI_max_1*n_A_I_1/(y[4]*std::pow(std::pow(y[4], n_A_I_1) + std::pow(K_A_I_1, n_A_I_1), 2));
	J( 5 , 5 ) = -1.0/2.0*y[2]*mu_max_1/(K_mu_glu + y[2]);

	dfdt[0] = 0.0;
	dfdt[1] = 0.0;
	dfdt[2] = 0.0;
	dfdt[3] = 0.0;
	dfdt[4] = 0.0;
	dfdt[5] = 0.0;

}
void Models::model_16(const ublas_vec_t  &y , ublas_vec_t &dydt , double t, std::vector <double> &part_params)
{
	//Unpack parameters
	const double D = part_params[0];
	const double g_1 = part_params[1];
	const double g_2 = part_params[2];
	const double K_A_B_1 = part_params[3];
	const double k_I_1 = part_params[4];
	const double K_mu_glu = part_params[5];
	const double k_omega_B_1 = part_params[6];
	const double kA_1 = part_params[7];
	const double kB_max_1 = part_params[8];
	const double kI_max_1 = part_params[9];
	const double mu_max_1 = part_params[10];
	const double mu_max_2 = part_params[11];
	const double n_A_B_1 = part_params[12];
	const double nI_1 = part_params[13];
	const double S0_glu = part_params[14];

	//Species order is: N_1 N_2 S_glu B_1 A_1 I_1 
	dydt[0] = -y[3]*y[0]*std::pow(k_I_1, nI_1)*k_omega_B_1/(std::pow(y[5], nI_1) + std::pow(k_I_1, nI_1)) - D*y[0] + y[0]*y[2]*mu_max_1/(K_mu_glu + y[2]);
	dydt[1] = -y[3]*y[1]*k_omega_B_1 - D*y[1] + y[1]*y[2]*mu_max_2/(K_mu_glu + y[2]);
	dydt[2] = D*(S0_glu - y[2]) - y[0]*y[2]*mu_max_1/(g_1*(K_mu_glu + y[2])) - y[1]*y[2]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	dydt[3] = -y[3]*D + std::pow(K_A_B_1, n_A_B_1)*y[0]*kB_max_1/(std::pow(y[4], n_A_B_1) + std::pow(K_A_B_1, n_A_B_1));
	dydt[4] = -y[4]*D + y[0]*kA_1;
	dydt[5] = -1.0/2.0*y[5]*y[2]*mu_max_1/(K_mu_glu + y[2]) + kI_max_1;

}
void Models::jac_16(const ublas_vec_t & y , ublas_mat_t &J , const double &/* t*/ , ublas_vec_t &dfdt, std::vector <double> &part_params)
{
	//Unpack parameters
	const double D = part_params[0];
	const double g_1 = part_params[1];
	const double g_2 = part_params[2];
	const double K_A_B_1 = part_params[3];
	const double k_I_1 = part_params[4];
	const double K_mu_glu = part_params[5];
	const double k_omega_B_1 = part_params[6];
	const double kA_1 = part_params[7];
	const double kB_max_1 = part_params[8];
	const double kI_max_1 = part_params[9];
	const double mu_max_1 = part_params[10];
	const double mu_max_2 = part_params[11];
	const double n_A_B_1 = part_params[12];
	const double nI_1 = part_params[13];
	const double S0_glu = part_params[14];

	J( 0 , 0 ) = -y[3]*std::pow(k_I_1, nI_1)*k_omega_B_1/(std::pow(y[5], nI_1) + std::pow(k_I_1, nI_1)) - D + y[2]*mu_max_1/(K_mu_glu + y[2]);
	J( 0 , 1 ) = 0;
	J( 0 , 2 ) = -y[0]*y[2]*mu_max_1/std::pow(K_mu_glu + y[2], 2) + y[0]*mu_max_1/(K_mu_glu + y[2]);
	J( 0 , 3 ) = -y[0]*std::pow(k_I_1, nI_1)*k_omega_B_1/(std::pow(y[5], nI_1) + std::pow(k_I_1, nI_1));
	J( 0 , 4 ) = 0;
	J( 0 , 5 ) = y[3]*std::pow(y[5], nI_1)*y[0]*std::pow(k_I_1, nI_1)*k_omega_B_1*nI_1/(y[5]*std::pow(std::pow(y[5], nI_1) + std::pow(k_I_1, nI_1), 2));
	J( 1 , 0 ) = 0;
	J( 1 , 1 ) = -y[3]*k_omega_B_1 - D + y[2]*mu_max_2/(K_mu_glu + y[2]);
	J( 1 , 2 ) = -y[1]*y[2]*mu_max_2/std::pow(K_mu_glu + y[2], 2) + y[1]*mu_max_2/(K_mu_glu + y[2]);
	J( 1 , 3 ) = -y[1]*k_omega_B_1;
	J( 1 , 4 ) = 0;
	J( 1 , 5 ) = 0;
	J( 2 , 0 ) = -y[2]*mu_max_1/(g_1*(K_mu_glu + y[2]));
	J( 2 , 1 ) = -y[2]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	J( 2 , 2 ) = -D + y[0]*y[2]*mu_max_1/(g_1*std::pow(K_mu_glu + y[2], 2)) - y[0]*mu_max_1/(g_1*(K_mu_glu + y[2])) + y[1]*y[2]*mu_max_2/(g_2*std::pow(K_mu_glu + y[2], 2)) - y[1]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	J( 2 , 3 ) = 0;
	J( 2 , 4 ) = 0;
	J( 2 , 5 ) = 0;
	J( 3 , 0 ) = std::pow(K_A_B_1, n_A_B_1)*kB_max_1/(std::pow(y[4], n_A_B_1) + std::pow(K_A_B_1, n_A_B_1));
	J( 3 , 1 ) = 0;
	J( 3 , 2 ) = 0;
	J( 3 , 3 ) = -D;
	J( 3 , 4 ) = -std::pow(y[4], n_A_B_1)*std::pow(K_A_B_1, n_A_B_1)*y[0]*kB_max_1*n_A_B_1/(y[4]*std::pow(std::pow(y[4], n_A_B_1) + std::pow(K_A_B_1, n_A_B_1), 2));
	J( 3 , 5 ) = 0;
	J( 4 , 0 ) = kA_1;
	J( 4 , 1 ) = 0;
	J( 4 , 2 ) = 0;
	J( 4 , 3 ) = 0;
	J( 4 , 4 ) = -D;
	J( 4 , 5 ) = 0;
	J( 5 , 0 ) = 0;
	J( 5 , 1 ) = 0;
	J( 5 , 2 ) = (1.0/2.0)*y[5]*y[2]*mu_max_1/std::pow(K_mu_glu + y[2], 2) - 1.0/2.0*y[5]*mu_max_1/(K_mu_glu + y[2]);
	J( 5 , 3 ) = 0;
	J( 5 , 4 ) = 0;
	J( 5 , 5 ) = -1.0/2.0*y[2]*mu_max_1/(K_mu_glu + y[2]);

	dfdt[0] = 0.0;
	dfdt[1] = 0.0;
	dfdt[2] = 0.0;
	dfdt[3] = 0.0;
	dfdt[4] = 0.0;
	dfdt[5] = 0.0;

}
void Models::model_17(const ublas_vec_t  &y , ublas_vec_t &dydt , double t, std::vector <double> &part_params)
{
	//Unpack parameters
	const double D = part_params[0];
	const double g_1 = part_params[1];
	const double g_2 = part_params[2];
	const double K_A_B_1 = part_params[3];
	const double k_I_1 = part_params[4];
	const double K_mu_glu = part_params[5];
	const double k_omega_B_1 = part_params[6];
	const double kA_1 = part_params[7];
	const double kB_max_1 = part_params[8];
	const double kI_max_1 = part_params[9];
	const double mu_max_1 = part_params[10];
	const double mu_max_2 = part_params[11];
	const double n_A_B_1 = part_params[12];
	const double nI_1 = part_params[13];
	const double S0_glu = part_params[14];

	//Species order is: N_1 N_2 S_glu B_1 A_1 I_1 
	dydt[0] = -y[3]*y[0]*std::pow(k_I_1, nI_1)*k_omega_B_1/(std::pow(y[5], nI_1) + std::pow(k_I_1, nI_1)) - D*y[0] + y[0]*y[2]*mu_max_1/(K_mu_glu + y[2]);
	dydt[1] = -D*y[1] + y[1]*y[2]*mu_max_2/(K_mu_glu + y[2]);
	dydt[2] = D*(S0_glu - y[2]) - y[0]*y[2]*mu_max_1/(g_1*(K_mu_glu + y[2])) - y[1]*y[2]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	dydt[3] = -y[3]*D + std::pow(K_A_B_1, n_A_B_1)*y[0]*kB_max_1/(std::pow(y[4], n_A_B_1) + std::pow(K_A_B_1, n_A_B_1));
	dydt[4] = -y[4]*D + y[0]*kA_1;
	dydt[5] = -1.0/2.0*y[5]*y[2]*mu_max_1/(K_mu_glu + y[2]) + kI_max_1;

}
void Models::jac_17(const ublas_vec_t & y , ublas_mat_t &J , const double &/* t*/ , ublas_vec_t &dfdt, std::vector <double> &part_params)
{
	//Unpack parameters
	const double D = part_params[0];
	const double g_1 = part_params[1];
	const double g_2 = part_params[2];
	const double K_A_B_1 = part_params[3];
	const double k_I_1 = part_params[4];
	const double K_mu_glu = part_params[5];
	const double k_omega_B_1 = part_params[6];
	const double kA_1 = part_params[7];
	const double kB_max_1 = part_params[8];
	const double kI_max_1 = part_params[9];
	const double mu_max_1 = part_params[10];
	const double mu_max_2 = part_params[11];
	const double n_A_B_1 = part_params[12];
	const double nI_1 = part_params[13];
	const double S0_glu = part_params[14];

	J( 0 , 0 ) = -y[3]*std::pow(k_I_1, nI_1)*k_omega_B_1/(std::pow(y[5], nI_1) + std::pow(k_I_1, nI_1)) - D + y[2]*mu_max_1/(K_mu_glu + y[2]);
	J( 0 , 1 ) = 0;
	J( 0 , 2 ) = -y[0]*y[2]*mu_max_1/std::pow(K_mu_glu + y[2], 2) + y[0]*mu_max_1/(K_mu_glu + y[2]);
	J( 0 , 3 ) = -y[0]*std::pow(k_I_1, nI_1)*k_omega_B_1/(std::pow(y[5], nI_1) + std::pow(k_I_1, nI_1));
	J( 0 , 4 ) = 0;
	J( 0 , 5 ) = y[3]*std::pow(y[5], nI_1)*y[0]*std::pow(k_I_1, nI_1)*k_omega_B_1*nI_1/(y[5]*std::pow(std::pow(y[5], nI_1) + std::pow(k_I_1, nI_1), 2));
	J( 1 , 0 ) = 0;
	J( 1 , 1 ) = -D + y[2]*mu_max_2/(K_mu_glu + y[2]);
	J( 1 , 2 ) = -y[1]*y[2]*mu_max_2/std::pow(K_mu_glu + y[2], 2) + y[1]*mu_max_2/(K_mu_glu + y[2]);
	J( 1 , 3 ) = 0;
	J( 1 , 4 ) = 0;
	J( 1 , 5 ) = 0;
	J( 2 , 0 ) = -y[2]*mu_max_1/(g_1*(K_mu_glu + y[2]));
	J( 2 , 1 ) = -y[2]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	J( 2 , 2 ) = -D + y[0]*y[2]*mu_max_1/(g_1*std::pow(K_mu_glu + y[2], 2)) - y[0]*mu_max_1/(g_1*(K_mu_glu + y[2])) + y[1]*y[2]*mu_max_2/(g_2*std::pow(K_mu_glu + y[2], 2)) - y[1]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	J( 2 , 3 ) = 0;
	J( 2 , 4 ) = 0;
	J( 2 , 5 ) = 0;
	J( 3 , 0 ) = std::pow(K_A_B_1, n_A_B_1)*kB_max_1/(std::pow(y[4], n_A_B_1) + std::pow(K_A_B_1, n_A_B_1));
	J( 3 , 1 ) = 0;
	J( 3 , 2 ) = 0;
	J( 3 , 3 ) = -D;
	J( 3 , 4 ) = -std::pow(y[4], n_A_B_1)*std::pow(K_A_B_1, n_A_B_1)*y[0]*kB_max_1*n_A_B_1/(y[4]*std::pow(std::pow(y[4], n_A_B_1) + std::pow(K_A_B_1, n_A_B_1), 2));
	J( 3 , 5 ) = 0;
	J( 4 , 0 ) = kA_1;
	J( 4 , 1 ) = 0;
	J( 4 , 2 ) = 0;
	J( 4 , 3 ) = 0;
	J( 4 , 4 ) = -D;
	J( 4 , 5 ) = 0;
	J( 5 , 0 ) = 0;
	J( 5 , 1 ) = 0;
	J( 5 , 2 ) = (1.0/2.0)*y[5]*y[2]*mu_max_1/std::pow(K_mu_glu + y[2], 2) - 1.0/2.0*y[5]*mu_max_1/(K_mu_glu + y[2]);
	J( 5 , 3 ) = 0;
	J( 5 , 4 ) = 0;
	J( 5 , 5 ) = -1.0/2.0*y[2]*mu_max_1/(K_mu_glu + y[2]);

	dfdt[0] = 0.0;
	dfdt[1] = 0.0;
	dfdt[2] = 0.0;
	dfdt[3] = 0.0;
	dfdt[4] = 0.0;
	dfdt[5] = 0.0;

}
void Models::model_18(const ublas_vec_t  &y , ublas_vec_t &dydt , double t, std::vector <double> &part_params)
{
	//Unpack parameters
	const double D = part_params[0];
	const double g_1 = part_params[1];
	const double g_2 = part_params[2];
	const double K_A_B_1 = part_params[3];
	const double K_mu_glu = part_params[4];
	const double k_omega_B_1 = part_params[5];
	const double kA_1 = part_params[6];
	const double kB_max_1 = part_params[7];
	const double mu_max_1 = part_params[8];
	const double mu_max_2 = part_params[9];
	const double n_A_B_1 = part_params[10];
	const double S0_glu = part_params[11];

	//Species order is: N_1 N_2 S_glu B_1 A_1 
	dydt[0] = -y[3]*y[0]*k_omega_B_1 - D*y[0] + y[0]*y[2]*mu_max_1/(K_mu_glu + y[2]);
	dydt[1] = -y[3]*y[1]*k_omega_B_1 - D*y[1] + y[1]*y[2]*mu_max_2/(K_mu_glu + y[2]);
	dydt[2] = D*(S0_glu - y[2]) - y[0]*y[2]*mu_max_1/(g_1*(K_mu_glu + y[2])) - y[1]*y[2]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	dydt[3] = -y[3]*D + std::pow(K_A_B_1, n_A_B_1)*y[0]*kB_max_1/(std::pow(y[4], n_A_B_1) + std::pow(K_A_B_1, n_A_B_1));
	dydt[4] = -y[4]*D + y[0]*kA_1;

}
void Models::jac_18(const ublas_vec_t & y , ublas_mat_t &J , const double &/* t*/ , ublas_vec_t &dfdt, std::vector <double> &part_params)
{
	//Unpack parameters
	const double D = part_params[0];
	const double g_1 = part_params[1];
	const double g_2 = part_params[2];
	const double K_A_B_1 = part_params[3];
	const double K_mu_glu = part_params[4];
	const double k_omega_B_1 = part_params[5];
	const double kA_1 = part_params[6];
	const double kB_max_1 = part_params[7];
	const double mu_max_1 = part_params[8];
	const double mu_max_2 = part_params[9];
	const double n_A_B_1 = part_params[10];
	const double S0_glu = part_params[11];

	J( 0 , 0 ) = -y[3]*k_omega_B_1 - D + y[2]*mu_max_1/(K_mu_glu + y[2]);
	J( 0 , 1 ) = 0;
	J( 0 , 2 ) = -y[0]*y[2]*mu_max_1/std::pow(K_mu_glu + y[2], 2) + y[0]*mu_max_1/(K_mu_glu + y[2]);
	J( 0 , 3 ) = -y[0]*k_omega_B_1;
	J( 0 , 4 ) = 0;
	J( 1 , 0 ) = 0;
	J( 1 , 1 ) = -y[3]*k_omega_B_1 - D + y[2]*mu_max_2/(K_mu_glu + y[2]);
	J( 1 , 2 ) = -y[1]*y[2]*mu_max_2/std::pow(K_mu_glu + y[2], 2) + y[1]*mu_max_2/(K_mu_glu + y[2]);
	J( 1 , 3 ) = -y[1]*k_omega_B_1;
	J( 1 , 4 ) = 0;
	J( 2 , 0 ) = -y[2]*mu_max_1/(g_1*(K_mu_glu + y[2]));
	J( 2 , 1 ) = -y[2]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	J( 2 , 2 ) = -D + y[0]*y[2]*mu_max_1/(g_1*std::pow(K_mu_glu + y[2], 2)) - y[0]*mu_max_1/(g_1*(K_mu_glu + y[2])) + y[1]*y[2]*mu_max_2/(g_2*std::pow(K_mu_glu + y[2], 2)) - y[1]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	J( 2 , 3 ) = 0;
	J( 2 , 4 ) = 0;
	J( 3 , 0 ) = std::pow(K_A_B_1, n_A_B_1)*kB_max_1/(std::pow(y[4], n_A_B_1) + std::pow(K_A_B_1, n_A_B_1));
	J( 3 , 1 ) = 0;
	J( 3 , 2 ) = 0;
	J( 3 , 3 ) = -D;
	J( 3 , 4 ) = -std::pow(y[4], n_A_B_1)*std::pow(K_A_B_1, n_A_B_1)*y[0]*kB_max_1*n_A_B_1/(y[4]*std::pow(std::pow(y[4], n_A_B_1) + std::pow(K_A_B_1, n_A_B_1), 2));
	J( 4 , 0 ) = kA_1;
	J( 4 , 1 ) = 0;
	J( 4 , 2 ) = 0;
	J( 4 , 3 ) = 0;
	J( 4 , 4 ) = -D;

	dfdt[0] = 0.0;
	dfdt[1] = 0.0;
	dfdt[2] = 0.0;
	dfdt[3] = 0.0;
	dfdt[4] = 0.0;

}
void Models::model_19(const ublas_vec_t  &y , ublas_vec_t &dydt , double t, std::vector <double> &part_params)
{
	//Unpack parameters
	const double D = part_params[0];
	const double g_1 = part_params[1];
	const double g_2 = part_params[2];
	const double K_A_B_1 = part_params[3];
	const double K_mu_glu = part_params[4];
	const double k_omega_B_1 = part_params[5];
	const double kA_1 = part_params[6];
	const double kB_max_1 = part_params[7];
	const double mu_max_1 = part_params[8];
	const double mu_max_2 = part_params[9];
	const double n_A_B_1 = part_params[10];
	const double S0_glu = part_params[11];

	//Species order is: N_1 N_2 S_glu B_1 A_1 
	dydt[0] = -y[3]*y[0]*k_omega_B_1 - D*y[0] + y[0]*y[2]*mu_max_1/(K_mu_glu + y[2]);
	dydt[1] = -D*y[1] + y[1]*y[2]*mu_max_2/(K_mu_glu + y[2]);
	dydt[2] = D*(S0_glu - y[2]) - y[0]*y[2]*mu_max_1/(g_1*(K_mu_glu + y[2])) - y[1]*y[2]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	dydt[3] = -y[3]*D + std::pow(K_A_B_1, n_A_B_1)*y[0]*kB_max_1/(std::pow(y[4], n_A_B_1) + std::pow(K_A_B_1, n_A_B_1));
	dydt[4] = -y[4]*D + y[0]*kA_1;

}
void Models::jac_19(const ublas_vec_t & y , ublas_mat_t &J , const double &/* t*/ , ublas_vec_t &dfdt, std::vector <double> &part_params)
{
	//Unpack parameters
	const double D = part_params[0];
	const double g_1 = part_params[1];
	const double g_2 = part_params[2];
	const double K_A_B_1 = part_params[3];
	const double K_mu_glu = part_params[4];
	const double k_omega_B_1 = part_params[5];
	const double kA_1 = part_params[6];
	const double kB_max_1 = part_params[7];
	const double mu_max_1 = part_params[8];
	const double mu_max_2 = part_params[9];
	const double n_A_B_1 = part_params[10];
	const double S0_glu = part_params[11];

	J( 0 , 0 ) = -y[3]*k_omega_B_1 - D + y[2]*mu_max_1/(K_mu_glu + y[2]);
	J( 0 , 1 ) = 0;
	J( 0 , 2 ) = -y[0]*y[2]*mu_max_1/std::pow(K_mu_glu + y[2], 2) + y[0]*mu_max_1/(K_mu_glu + y[2]);
	J( 0 , 3 ) = -y[0]*k_omega_B_1;
	J( 0 , 4 ) = 0;
	J( 1 , 0 ) = 0;
	J( 1 , 1 ) = -D + y[2]*mu_max_2/(K_mu_glu + y[2]);
	J( 1 , 2 ) = -y[1]*y[2]*mu_max_2/std::pow(K_mu_glu + y[2], 2) + y[1]*mu_max_2/(K_mu_glu + y[2]);
	J( 1 , 3 ) = 0;
	J( 1 , 4 ) = 0;
	J( 2 , 0 ) = -y[2]*mu_max_1/(g_1*(K_mu_glu + y[2]));
	J( 2 , 1 ) = -y[2]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	J( 2 , 2 ) = -D + y[0]*y[2]*mu_max_1/(g_1*std::pow(K_mu_glu + y[2], 2)) - y[0]*mu_max_1/(g_1*(K_mu_glu + y[2])) + y[1]*y[2]*mu_max_2/(g_2*std::pow(K_mu_glu + y[2], 2)) - y[1]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	J( 2 , 3 ) = 0;
	J( 2 , 4 ) = 0;
	J( 3 , 0 ) = std::pow(K_A_B_1, n_A_B_1)*kB_max_1/(std::pow(y[4], n_A_B_1) + std::pow(K_A_B_1, n_A_B_1));
	J( 3 , 1 ) = 0;
	J( 3 , 2 ) = 0;
	J( 3 , 3 ) = -D;
	J( 3 , 4 ) = -std::pow(y[4], n_A_B_1)*std::pow(K_A_B_1, n_A_B_1)*y[0]*kB_max_1*n_A_B_1/(y[4]*std::pow(std::pow(y[4], n_A_B_1) + std::pow(K_A_B_1, n_A_B_1), 2));
	J( 4 , 0 ) = kA_1;
	J( 4 , 1 ) = 0;
	J( 4 , 2 ) = 0;
	J( 4 , 3 ) = 0;
	J( 4 , 4 ) = -D;

	dfdt[0] = 0.0;
	dfdt[1] = 0.0;
	dfdt[2] = 0.0;
	dfdt[3] = 0.0;
	dfdt[4] = 0.0;

}
void Models::model_20(const ublas_vec_t  &y , ublas_vec_t &dydt , double t, std::vector <double> &part_params)
{
	//Unpack parameters
	const double D = part_params[0];
	const double g_1 = part_params[1];
	const double g_2 = part_params[2];
	const double K_A_B_1 = part_params[3];
	const double K_A_I_1 = part_params[4];
	const double K_mu_glu = part_params[5];
	const double k_omega_B_1 = part_params[6];
	const double kA_1 = part_params[7];
	const double kB_max_1 = part_params[8];
	const double kI_max_1 = part_params[9];
	const double mu_max_1 = part_params[10];
	const double mu_max_2 = part_params[11];
	const double n_A_B_1 = part_params[12];
	const double n_A_I_1 = part_params[13];
	const double S0_glu = part_params[14];

	//Species order is: N_1 N_2 S_glu B_1 A_1 I_1 
	dydt[0] = -D*y[0] + y[0]*y[2]*mu_max_1/(K_mu_glu + y[2]);
	dydt[1] = -y[3]*y[1]*k_omega_B_1 - D*y[1] + y[1]*y[2]*mu_max_2/(K_mu_glu + y[2]);
	dydt[2] = D*(S0_glu - y[2]) - y[0]*y[2]*mu_max_1/(g_1*(K_mu_glu + y[2])) - y[1]*y[2]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	dydt[3] = -y[3]*D + std::pow(K_A_B_1, n_A_B_1)*y[0]*kB_max_1/(std::pow(y[4], n_A_B_1) + std::pow(K_A_B_1, n_A_B_1));
	dydt[4] = -y[4]*D + y[0]*kA_1;
	dydt[5] = std::pow(y[4], n_A_I_1)*kI_max_1/(std::pow(y[4], n_A_I_1) + std::pow(K_A_I_1, n_A_I_1)) - 1.0/2.0*y[5]*y[2]*mu_max_1/(K_mu_glu + y[2]);

}
void Models::jac_20(const ublas_vec_t & y , ublas_mat_t &J , const double &/* t*/ , ublas_vec_t &dfdt, std::vector <double> &part_params)
{
	//Unpack parameters
	const double D = part_params[0];
	const double g_1 = part_params[1];
	const double g_2 = part_params[2];
	const double K_A_B_1 = part_params[3];
	const double K_A_I_1 = part_params[4];
	const double K_mu_glu = part_params[5];
	const double k_omega_B_1 = part_params[6];
	const double kA_1 = part_params[7];
	const double kB_max_1 = part_params[8];
	const double kI_max_1 = part_params[9];
	const double mu_max_1 = part_params[10];
	const double mu_max_2 = part_params[11];
	const double n_A_B_1 = part_params[12];
	const double n_A_I_1 = part_params[13];
	const double S0_glu = part_params[14];

	J( 0 , 0 ) = -D + y[2]*mu_max_1/(K_mu_glu + y[2]);
	J( 0 , 1 ) = 0;
	J( 0 , 2 ) = -y[0]*y[2]*mu_max_1/std::pow(K_mu_glu + y[2], 2) + y[0]*mu_max_1/(K_mu_glu + y[2]);
	J( 0 , 3 ) = 0;
	J( 0 , 4 ) = 0;
	J( 0 , 5 ) = 0;
	J( 1 , 0 ) = 0;
	J( 1 , 1 ) = -y[3]*k_omega_B_1 - D + y[2]*mu_max_2/(K_mu_glu + y[2]);
	J( 1 , 2 ) = -y[1]*y[2]*mu_max_2/std::pow(K_mu_glu + y[2], 2) + y[1]*mu_max_2/(K_mu_glu + y[2]);
	J( 1 , 3 ) = -y[1]*k_omega_B_1;
	J( 1 , 4 ) = 0;
	J( 1 , 5 ) = 0;
	J( 2 , 0 ) = -y[2]*mu_max_1/(g_1*(K_mu_glu + y[2]));
	J( 2 , 1 ) = -y[2]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	J( 2 , 2 ) = -D + y[0]*y[2]*mu_max_1/(g_1*std::pow(K_mu_glu + y[2], 2)) - y[0]*mu_max_1/(g_1*(K_mu_glu + y[2])) + y[1]*y[2]*mu_max_2/(g_2*std::pow(K_mu_glu + y[2], 2)) - y[1]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	J( 2 , 3 ) = 0;
	J( 2 , 4 ) = 0;
	J( 2 , 5 ) = 0;
	J( 3 , 0 ) = std::pow(K_A_B_1, n_A_B_1)*kB_max_1/(std::pow(y[4], n_A_B_1) + std::pow(K_A_B_1, n_A_B_1));
	J( 3 , 1 ) = 0;
	J( 3 , 2 ) = 0;
	J( 3 , 3 ) = -D;
	J( 3 , 4 ) = -std::pow(y[4], n_A_B_1)*std::pow(K_A_B_1, n_A_B_1)*y[0]*kB_max_1*n_A_B_1/(y[4]*std::pow(std::pow(y[4], n_A_B_1) + std::pow(K_A_B_1, n_A_B_1), 2));
	J( 3 , 5 ) = 0;
	J( 4 , 0 ) = kA_1;
	J( 4 , 1 ) = 0;
	J( 4 , 2 ) = 0;
	J( 4 , 3 ) = 0;
	J( 4 , 4 ) = -D;
	J( 4 , 5 ) = 0;
	J( 5 , 0 ) = 0;
	J( 5 , 1 ) = 0;
	J( 5 , 2 ) = (1.0/2.0)*y[5]*y[2]*mu_max_1/std::pow(K_mu_glu + y[2], 2) - 1.0/2.0*y[5]*mu_max_1/(K_mu_glu + y[2]);
	J( 5 , 3 ) = 0;
	J( 5 , 4 ) = -std::pow(y[4], 2*n_A_I_1)*kI_max_1*n_A_I_1/(y[4]*std::pow(std::pow(y[4], n_A_I_1) + std::pow(K_A_I_1, n_A_I_1), 2)) + std::pow(y[4], n_A_I_1)*kI_max_1*n_A_I_1/(y[4]*(std::pow(y[4], n_A_I_1) + std::pow(K_A_I_1, n_A_I_1)));
	J( 5 , 5 ) = -1.0/2.0*y[2]*mu_max_1/(K_mu_glu + y[2]);

	dfdt[0] = 0.0;
	dfdt[1] = 0.0;
	dfdt[2] = 0.0;
	dfdt[3] = 0.0;
	dfdt[4] = 0.0;
	dfdt[5] = 0.0;

}
void Models::model_21(const ublas_vec_t  &y , ublas_vec_t &dydt , double t, std::vector <double> &part_params)
{
	//Unpack parameters
	const double D = part_params[0];
	const double g_1 = part_params[1];
	const double g_2 = part_params[2];
	const double K_A_B_1 = part_params[3];
	const double K_A_I_1 = part_params[4];
	const double K_mu_glu = part_params[5];
	const double k_omega_B_1 = part_params[6];
	const double kA_1 = part_params[7];
	const double kB_max_1 = part_params[8];
	const double kI_max_1 = part_params[9];
	const double mu_max_1 = part_params[10];
	const double mu_max_2 = part_params[11];
	const double n_A_B_1 = part_params[12];
	const double n_A_I_1 = part_params[13];
	const double S0_glu = part_params[14];

	//Species order is: N_1 N_2 S_glu B_1 A_1 I_1 
	dydt[0] = -D*y[0] + y[0]*y[2]*mu_max_1/(K_mu_glu + y[2]);
	dydt[1] = -y[3]*y[1]*k_omega_B_1 - D*y[1] + y[1]*y[2]*mu_max_2/(K_mu_glu + y[2]);
	dydt[2] = D*(S0_glu - y[2]) - y[0]*y[2]*mu_max_1/(g_1*(K_mu_glu + y[2])) - y[1]*y[2]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	dydt[3] = -y[3]*D + std::pow(K_A_B_1, n_A_B_1)*y[0]*kB_max_1/(std::pow(y[4], n_A_B_1) + std::pow(K_A_B_1, n_A_B_1));
	dydt[4] = -y[4]*D + y[0]*kA_1;
	dydt[5] = -1.0/2.0*y[5]*y[2]*mu_max_1/(K_mu_glu + y[2]) + std::pow(K_A_I_1, n_A_I_1)*kI_max_1/(std::pow(y[4], n_A_I_1) + std::pow(K_A_I_1, n_A_I_1));

}
void Models::jac_21(const ublas_vec_t & y , ublas_mat_t &J , const double &/* t*/ , ublas_vec_t &dfdt, std::vector <double> &part_params)
{
	//Unpack parameters
	const double D = part_params[0];
	const double g_1 = part_params[1];
	const double g_2 = part_params[2];
	const double K_A_B_1 = part_params[3];
	const double K_A_I_1 = part_params[4];
	const double K_mu_glu = part_params[5];
	const double k_omega_B_1 = part_params[6];
	const double kA_1 = part_params[7];
	const double kB_max_1 = part_params[8];
	const double kI_max_1 = part_params[9];
	const double mu_max_1 = part_params[10];
	const double mu_max_2 = part_params[11];
	const double n_A_B_1 = part_params[12];
	const double n_A_I_1 = part_params[13];
	const double S0_glu = part_params[14];

	J( 0 , 0 ) = -D + y[2]*mu_max_1/(K_mu_glu + y[2]);
	J( 0 , 1 ) = 0;
	J( 0 , 2 ) = -y[0]*y[2]*mu_max_1/std::pow(K_mu_glu + y[2], 2) + y[0]*mu_max_1/(K_mu_glu + y[2]);
	J( 0 , 3 ) = 0;
	J( 0 , 4 ) = 0;
	J( 0 , 5 ) = 0;
	J( 1 , 0 ) = 0;
	J( 1 , 1 ) = -y[3]*k_omega_B_1 - D + y[2]*mu_max_2/(K_mu_glu + y[2]);
	J( 1 , 2 ) = -y[1]*y[2]*mu_max_2/std::pow(K_mu_glu + y[2], 2) + y[1]*mu_max_2/(K_mu_glu + y[2]);
	J( 1 , 3 ) = -y[1]*k_omega_B_1;
	J( 1 , 4 ) = 0;
	J( 1 , 5 ) = 0;
	J( 2 , 0 ) = -y[2]*mu_max_1/(g_1*(K_mu_glu + y[2]));
	J( 2 , 1 ) = -y[2]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	J( 2 , 2 ) = -D + y[0]*y[2]*mu_max_1/(g_1*std::pow(K_mu_glu + y[2], 2)) - y[0]*mu_max_1/(g_1*(K_mu_glu + y[2])) + y[1]*y[2]*mu_max_2/(g_2*std::pow(K_mu_glu + y[2], 2)) - y[1]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	J( 2 , 3 ) = 0;
	J( 2 , 4 ) = 0;
	J( 2 , 5 ) = 0;
	J( 3 , 0 ) = std::pow(K_A_B_1, n_A_B_1)*kB_max_1/(std::pow(y[4], n_A_B_1) + std::pow(K_A_B_1, n_A_B_1));
	J( 3 , 1 ) = 0;
	J( 3 , 2 ) = 0;
	J( 3 , 3 ) = -D;
	J( 3 , 4 ) = -std::pow(y[4], n_A_B_1)*std::pow(K_A_B_1, n_A_B_1)*y[0]*kB_max_1*n_A_B_1/(y[4]*std::pow(std::pow(y[4], n_A_B_1) + std::pow(K_A_B_1, n_A_B_1), 2));
	J( 3 , 5 ) = 0;
	J( 4 , 0 ) = kA_1;
	J( 4 , 1 ) = 0;
	J( 4 , 2 ) = 0;
	J( 4 , 3 ) = 0;
	J( 4 , 4 ) = -D;
	J( 4 , 5 ) = 0;
	J( 5 , 0 ) = 0;
	J( 5 , 1 ) = 0;
	J( 5 , 2 ) = (1.0/2.0)*y[5]*y[2]*mu_max_1/std::pow(K_mu_glu + y[2], 2) - 1.0/2.0*y[5]*mu_max_1/(K_mu_glu + y[2]);
	J( 5 , 3 ) = 0;
	J( 5 , 4 ) = -std::pow(y[4], n_A_I_1)*std::pow(K_A_I_1, n_A_I_1)*kI_max_1*n_A_I_1/(y[4]*std::pow(std::pow(y[4], n_A_I_1) + std::pow(K_A_I_1, n_A_I_1), 2));
	J( 5 , 5 ) = -1.0/2.0*y[2]*mu_max_1/(K_mu_glu + y[2]);

	dfdt[0] = 0.0;
	dfdt[1] = 0.0;
	dfdt[2] = 0.0;
	dfdt[3] = 0.0;
	dfdt[4] = 0.0;
	dfdt[5] = 0.0;

}
void Models::model_22(const ublas_vec_t  &y , ublas_vec_t &dydt , double t, std::vector <double> &part_params)
{
	//Unpack parameters
	const double D = part_params[0];
	const double g_1 = part_params[1];
	const double g_2 = part_params[2];
	const double K_A_B_1 = part_params[3];
	const double K_mu_glu = part_params[4];
	const double k_omega_B_1 = part_params[5];
	const double kA_1 = part_params[6];
	const double kB_max_1 = part_params[7];
	const double kI_max_1 = part_params[8];
	const double mu_max_1 = part_params[9];
	const double mu_max_2 = part_params[10];
	const double n_A_B_1 = part_params[11];
	const double S0_glu = part_params[12];

	//Species order is: N_1 N_2 S_glu B_1 A_1 I_1 
	dydt[0] = -D*y[0] + y[0]*y[2]*mu_max_1/(K_mu_glu + y[2]);
	dydt[1] = -y[3]*y[1]*k_omega_B_1 - D*y[1] + y[1]*y[2]*mu_max_2/(K_mu_glu + y[2]);
	dydt[2] = D*(S0_glu - y[2]) - y[0]*y[2]*mu_max_1/(g_1*(K_mu_glu + y[2])) - y[1]*y[2]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	dydt[3] = -y[3]*D + std::pow(K_A_B_1, n_A_B_1)*y[0]*kB_max_1/(std::pow(y[4], n_A_B_1) + std::pow(K_A_B_1, n_A_B_1));
	dydt[4] = -y[4]*D + y[0]*kA_1;
	dydt[5] = -1.0/2.0*y[5]*y[2]*mu_max_1/(K_mu_glu + y[2]) + kI_max_1;

}
void Models::jac_22(const ublas_vec_t & y , ublas_mat_t &J , const double &/* t*/ , ublas_vec_t &dfdt, std::vector <double> &part_params)
{
	//Unpack parameters
	const double D = part_params[0];
	const double g_1 = part_params[1];
	const double g_2 = part_params[2];
	const double K_A_B_1 = part_params[3];
	const double K_mu_glu = part_params[4];
	const double k_omega_B_1 = part_params[5];
	const double kA_1 = part_params[6];
	const double kB_max_1 = part_params[7];
	const double kI_max_1 = part_params[8];
	const double mu_max_1 = part_params[9];
	const double mu_max_2 = part_params[10];
	const double n_A_B_1 = part_params[11];
	const double S0_glu = part_params[12];

	J( 0 , 0 ) = -D + y[2]*mu_max_1/(K_mu_glu + y[2]);
	J( 0 , 1 ) = 0;
	J( 0 , 2 ) = -y[0]*y[2]*mu_max_1/std::pow(K_mu_glu + y[2], 2) + y[0]*mu_max_1/(K_mu_glu + y[2]);
	J( 0 , 3 ) = 0;
	J( 0 , 4 ) = 0;
	J( 0 , 5 ) = 0;
	J( 1 , 0 ) = 0;
	J( 1 , 1 ) = -y[3]*k_omega_B_1 - D + y[2]*mu_max_2/(K_mu_glu + y[2]);
	J( 1 , 2 ) = -y[1]*y[2]*mu_max_2/std::pow(K_mu_glu + y[2], 2) + y[1]*mu_max_2/(K_mu_glu + y[2]);
	J( 1 , 3 ) = -y[1]*k_omega_B_1;
	J( 1 , 4 ) = 0;
	J( 1 , 5 ) = 0;
	J( 2 , 0 ) = -y[2]*mu_max_1/(g_1*(K_mu_glu + y[2]));
	J( 2 , 1 ) = -y[2]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	J( 2 , 2 ) = -D + y[0]*y[2]*mu_max_1/(g_1*std::pow(K_mu_glu + y[2], 2)) - y[0]*mu_max_1/(g_1*(K_mu_glu + y[2])) + y[1]*y[2]*mu_max_2/(g_2*std::pow(K_mu_glu + y[2], 2)) - y[1]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	J( 2 , 3 ) = 0;
	J( 2 , 4 ) = 0;
	J( 2 , 5 ) = 0;
	J( 3 , 0 ) = std::pow(K_A_B_1, n_A_B_1)*kB_max_1/(std::pow(y[4], n_A_B_1) + std::pow(K_A_B_1, n_A_B_1));
	J( 3 , 1 ) = 0;
	J( 3 , 2 ) = 0;
	J( 3 , 3 ) = -D;
	J( 3 , 4 ) = -std::pow(y[4], n_A_B_1)*std::pow(K_A_B_1, n_A_B_1)*y[0]*kB_max_1*n_A_B_1/(y[4]*std::pow(std::pow(y[4], n_A_B_1) + std::pow(K_A_B_1, n_A_B_1), 2));
	J( 3 , 5 ) = 0;
	J( 4 , 0 ) = kA_1;
	J( 4 , 1 ) = 0;
	J( 4 , 2 ) = 0;
	J( 4 , 3 ) = 0;
	J( 4 , 4 ) = -D;
	J( 4 , 5 ) = 0;
	J( 5 , 0 ) = 0;
	J( 5 , 1 ) = 0;
	J( 5 , 2 ) = (1.0/2.0)*y[5]*y[2]*mu_max_1/std::pow(K_mu_glu + y[2], 2) - 1.0/2.0*y[5]*mu_max_1/(K_mu_glu + y[2]);
	J( 5 , 3 ) = 0;
	J( 5 , 4 ) = 0;
	J( 5 , 5 ) = -1.0/2.0*y[2]*mu_max_1/(K_mu_glu + y[2]);

	dfdt[0] = 0.0;
	dfdt[1] = 0.0;
	dfdt[2] = 0.0;
	dfdt[3] = 0.0;
	dfdt[4] = 0.0;
	dfdt[5] = 0.0;

}
void Models::model_23(const ublas_vec_t  &y , ublas_vec_t &dydt , double t, std::vector <double> &part_params)
{
	//Unpack parameters
	const double D = part_params[0];
	const double g_1 = part_params[1];
	const double g_2 = part_params[2];
	const double K_A_B_1 = part_params[3];
	const double K_mu_glu = part_params[4];
	const double k_omega_B_1 = part_params[5];
	const double kA_1 = part_params[6];
	const double kB_max_1 = part_params[7];
	const double mu_max_1 = part_params[8];
	const double mu_max_2 = part_params[9];
	const double n_A_B_1 = part_params[10];
	const double S0_glu = part_params[11];

	//Species order is: N_1 N_2 S_glu B_1 A_1 
	dydt[0] = -D*y[0] + y[0]*y[2]*mu_max_1/(K_mu_glu + y[2]);
	dydt[1] = -y[3]*y[1]*k_omega_B_1 - D*y[1] + y[1]*y[2]*mu_max_2/(K_mu_glu + y[2]);
	dydt[2] = D*(S0_glu - y[2]) - y[0]*y[2]*mu_max_1/(g_1*(K_mu_glu + y[2])) - y[1]*y[2]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	dydt[3] = -y[3]*D + std::pow(K_A_B_1, n_A_B_1)*y[0]*kB_max_1/(std::pow(y[4], n_A_B_1) + std::pow(K_A_B_1, n_A_B_1));
	dydt[4] = -y[4]*D + y[0]*kA_1;

}
void Models::jac_23(const ublas_vec_t & y , ublas_mat_t &J , const double &/* t*/ , ublas_vec_t &dfdt, std::vector <double> &part_params)
{
	//Unpack parameters
	const double D = part_params[0];
	const double g_1 = part_params[1];
	const double g_2 = part_params[2];
	const double K_A_B_1 = part_params[3];
	const double K_mu_glu = part_params[4];
	const double k_omega_B_1 = part_params[5];
	const double kA_1 = part_params[6];
	const double kB_max_1 = part_params[7];
	const double mu_max_1 = part_params[8];
	const double mu_max_2 = part_params[9];
	const double n_A_B_1 = part_params[10];
	const double S0_glu = part_params[11];

	J( 0 , 0 ) = -D + y[2]*mu_max_1/(K_mu_glu + y[2]);
	J( 0 , 1 ) = 0;
	J( 0 , 2 ) = -y[0]*y[2]*mu_max_1/std::pow(K_mu_glu + y[2], 2) + y[0]*mu_max_1/(K_mu_glu + y[2]);
	J( 0 , 3 ) = 0;
	J( 0 , 4 ) = 0;
	J( 1 , 0 ) = 0;
	J( 1 , 1 ) = -y[3]*k_omega_B_1 - D + y[2]*mu_max_2/(K_mu_glu + y[2]);
	J( 1 , 2 ) = -y[1]*y[2]*mu_max_2/std::pow(K_mu_glu + y[2], 2) + y[1]*mu_max_2/(K_mu_glu + y[2]);
	J( 1 , 3 ) = -y[1]*k_omega_B_1;
	J( 1 , 4 ) = 0;
	J( 2 , 0 ) = -y[2]*mu_max_1/(g_1*(K_mu_glu + y[2]));
	J( 2 , 1 ) = -y[2]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	J( 2 , 2 ) = -D + y[0]*y[2]*mu_max_1/(g_1*std::pow(K_mu_glu + y[2], 2)) - y[0]*mu_max_1/(g_1*(K_mu_glu + y[2])) + y[1]*y[2]*mu_max_2/(g_2*std::pow(K_mu_glu + y[2], 2)) - y[1]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	J( 2 , 3 ) = 0;
	J( 2 , 4 ) = 0;
	J( 3 , 0 ) = std::pow(K_A_B_1, n_A_B_1)*kB_max_1/(std::pow(y[4], n_A_B_1) + std::pow(K_A_B_1, n_A_B_1));
	J( 3 , 1 ) = 0;
	J( 3 , 2 ) = 0;
	J( 3 , 3 ) = -D;
	J( 3 , 4 ) = -std::pow(y[4], n_A_B_1)*std::pow(K_A_B_1, n_A_B_1)*y[0]*kB_max_1*n_A_B_1/(y[4]*std::pow(std::pow(y[4], n_A_B_1) + std::pow(K_A_B_1, n_A_B_1), 2));
	J( 4 , 0 ) = kA_1;
	J( 4 , 1 ) = 0;
	J( 4 , 2 ) = 0;
	J( 4 , 3 ) = 0;
	J( 4 , 4 ) = -D;

	dfdt[0] = 0.0;
	dfdt[1] = 0.0;
	dfdt[2] = 0.0;
	dfdt[3] = 0.0;
	dfdt[4] = 0.0;

}
void Models::model_24(const ublas_vec_t  &y , ublas_vec_t &dydt , double t, std::vector <double> &part_params)
{
	//Unpack parameters
	const double D = part_params[0];
	const double g_1 = part_params[1];
	const double g_2 = part_params[2];
	const double K_A_I_1 = part_params[3];
	const double k_I_1 = part_params[4];
	const double K_mu_glu = part_params[5];
	const double k_omega_B_1 = part_params[6];
	const double kA_1 = part_params[7];
	const double kB_max_1 = part_params[8];
	const double kI_max_1 = part_params[9];
	const double mu_max_1 = part_params[10];
	const double mu_max_2 = part_params[11];
	const double n_A_I_1 = part_params[12];
	const double nI_1 = part_params[13];
	const double S0_glu = part_params[14];

	//Species order is: N_1 N_2 S_glu B_1 A_1 I_1 
	dydt[0] = -y[3]*y[0]*std::pow(k_I_1, nI_1)*k_omega_B_1/(std::pow(y[5], nI_1) + std::pow(k_I_1, nI_1)) - D*y[0] + y[0]*y[2]*mu_max_1/(K_mu_glu + y[2]);
	dydt[1] = -y[3]*y[1]*k_omega_B_1 - D*y[1] + y[1]*y[2]*mu_max_2/(K_mu_glu + y[2]);
	dydt[2] = D*(S0_glu - y[2]) - y[0]*y[2]*mu_max_1/(g_1*(K_mu_glu + y[2])) - y[1]*y[2]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	dydt[3] = -y[3]*D + y[0]*kB_max_1;
	dydt[4] = -y[4]*D + y[0]*kA_1;
	dydt[5] = std::pow(y[4], n_A_I_1)*kI_max_1/(std::pow(y[4], n_A_I_1) + std::pow(K_A_I_1, n_A_I_1)) - 1.0/2.0*y[5]*y[2]*mu_max_1/(K_mu_glu + y[2]);

}
void Models::jac_24(const ublas_vec_t & y , ublas_mat_t &J , const double &/* t*/ , ublas_vec_t &dfdt, std::vector <double> &part_params)
{
	//Unpack parameters
	const double D = part_params[0];
	const double g_1 = part_params[1];
	const double g_2 = part_params[2];
	const double K_A_I_1 = part_params[3];
	const double k_I_1 = part_params[4];
	const double K_mu_glu = part_params[5];
	const double k_omega_B_1 = part_params[6];
	const double kA_1 = part_params[7];
	const double kB_max_1 = part_params[8];
	const double kI_max_1 = part_params[9];
	const double mu_max_1 = part_params[10];
	const double mu_max_2 = part_params[11];
	const double n_A_I_1 = part_params[12];
	const double nI_1 = part_params[13];
	const double S0_glu = part_params[14];

	J( 0 , 0 ) = -y[3]*std::pow(k_I_1, nI_1)*k_omega_B_1/(std::pow(y[5], nI_1) + std::pow(k_I_1, nI_1)) - D + y[2]*mu_max_1/(K_mu_glu + y[2]);
	J( 0 , 1 ) = 0;
	J( 0 , 2 ) = -y[0]*y[2]*mu_max_1/std::pow(K_mu_glu + y[2], 2) + y[0]*mu_max_1/(K_mu_glu + y[2]);
	J( 0 , 3 ) = -y[0]*std::pow(k_I_1, nI_1)*k_omega_B_1/(std::pow(y[5], nI_1) + std::pow(k_I_1, nI_1));
	J( 0 , 4 ) = 0;
	J( 0 , 5 ) = y[3]*std::pow(y[5], nI_1)*y[0]*std::pow(k_I_1, nI_1)*k_omega_B_1*nI_1/(y[5]*std::pow(std::pow(y[5], nI_1) + std::pow(k_I_1, nI_1), 2));
	J( 1 , 0 ) = 0;
	J( 1 , 1 ) = -y[3]*k_omega_B_1 - D + y[2]*mu_max_2/(K_mu_glu + y[2]);
	J( 1 , 2 ) = -y[1]*y[2]*mu_max_2/std::pow(K_mu_glu + y[2], 2) + y[1]*mu_max_2/(K_mu_glu + y[2]);
	J( 1 , 3 ) = -y[1]*k_omega_B_1;
	J( 1 , 4 ) = 0;
	J( 1 , 5 ) = 0;
	J( 2 , 0 ) = -y[2]*mu_max_1/(g_1*(K_mu_glu + y[2]));
	J( 2 , 1 ) = -y[2]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	J( 2 , 2 ) = -D + y[0]*y[2]*mu_max_1/(g_1*std::pow(K_mu_glu + y[2], 2)) - y[0]*mu_max_1/(g_1*(K_mu_glu + y[2])) + y[1]*y[2]*mu_max_2/(g_2*std::pow(K_mu_glu + y[2], 2)) - y[1]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	J( 2 , 3 ) = 0;
	J( 2 , 4 ) = 0;
	J( 2 , 5 ) = 0;
	J( 3 , 0 ) = kB_max_1;
	J( 3 , 1 ) = 0;
	J( 3 , 2 ) = 0;
	J( 3 , 3 ) = -D;
	J( 3 , 4 ) = 0;
	J( 3 , 5 ) = 0;
	J( 4 , 0 ) = kA_1;
	J( 4 , 1 ) = 0;
	J( 4 , 2 ) = 0;
	J( 4 , 3 ) = 0;
	J( 4 , 4 ) = -D;
	J( 4 , 5 ) = 0;
	J( 5 , 0 ) = 0;
	J( 5 , 1 ) = 0;
	J( 5 , 2 ) = (1.0/2.0)*y[5]*y[2]*mu_max_1/std::pow(K_mu_glu + y[2], 2) - 1.0/2.0*y[5]*mu_max_1/(K_mu_glu + y[2]);
	J( 5 , 3 ) = 0;
	J( 5 , 4 ) = -std::pow(y[4], 2*n_A_I_1)*kI_max_1*n_A_I_1/(y[4]*std::pow(std::pow(y[4], n_A_I_1) + std::pow(K_A_I_1, n_A_I_1), 2)) + std::pow(y[4], n_A_I_1)*kI_max_1*n_A_I_1/(y[4]*(std::pow(y[4], n_A_I_1) + std::pow(K_A_I_1, n_A_I_1)));
	J( 5 , 5 ) = -1.0/2.0*y[2]*mu_max_1/(K_mu_glu + y[2]);

	dfdt[0] = 0.0;
	dfdt[1] = 0.0;
	dfdt[2] = 0.0;
	dfdt[3] = 0.0;
	dfdt[4] = 0.0;
	dfdt[5] = 0.0;

}
void Models::model_25(const ublas_vec_t  &y , ublas_vec_t &dydt , double t, std::vector <double> &part_params)
{
	//Unpack parameters
	const double D = part_params[0];
	const double g_1 = part_params[1];
	const double g_2 = part_params[2];
	const double K_A_I_1 = part_params[3];
	const double k_I_1 = part_params[4];
	const double K_mu_glu = part_params[5];
	const double k_omega_B_1 = part_params[6];
	const double kA_1 = part_params[7];
	const double kB_max_1 = part_params[8];
	const double kI_max_1 = part_params[9];
	const double mu_max_1 = part_params[10];
	const double mu_max_2 = part_params[11];
	const double n_A_I_1 = part_params[12];
	const double nI_1 = part_params[13];
	const double S0_glu = part_params[14];

	//Species order is: N_1 N_2 S_glu B_1 A_1 I_1 
	dydt[0] = -y[3]*y[0]*std::pow(k_I_1, nI_1)*k_omega_B_1/(std::pow(y[5], nI_1) + std::pow(k_I_1, nI_1)) - D*y[0] + y[0]*y[2]*mu_max_1/(K_mu_glu + y[2]);
	dydt[1] = -D*y[1] + y[1]*y[2]*mu_max_2/(K_mu_glu + y[2]);
	dydt[2] = D*(S0_glu - y[2]) - y[0]*y[2]*mu_max_1/(g_1*(K_mu_glu + y[2])) - y[1]*y[2]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	dydt[3] = -y[3]*D + y[0]*kB_max_1;
	dydt[4] = -y[4]*D + y[0]*kA_1;
	dydt[5] = std::pow(y[4], n_A_I_1)*kI_max_1/(std::pow(y[4], n_A_I_1) + std::pow(K_A_I_1, n_A_I_1)) - 1.0/2.0*y[5]*y[2]*mu_max_1/(K_mu_glu + y[2]);

}
void Models::jac_25(const ublas_vec_t & y , ublas_mat_t &J , const double &/* t*/ , ublas_vec_t &dfdt, std::vector <double> &part_params)
{
	//Unpack parameters
	const double D = part_params[0];
	const double g_1 = part_params[1];
	const double g_2 = part_params[2];
	const double K_A_I_1 = part_params[3];
	const double k_I_1 = part_params[4];
	const double K_mu_glu = part_params[5];
	const double k_omega_B_1 = part_params[6];
	const double kA_1 = part_params[7];
	const double kB_max_1 = part_params[8];
	const double kI_max_1 = part_params[9];
	const double mu_max_1 = part_params[10];
	const double mu_max_2 = part_params[11];
	const double n_A_I_1 = part_params[12];
	const double nI_1 = part_params[13];
	const double S0_glu = part_params[14];

	J( 0 , 0 ) = -y[3]*std::pow(k_I_1, nI_1)*k_omega_B_1/(std::pow(y[5], nI_1) + std::pow(k_I_1, nI_1)) - D + y[2]*mu_max_1/(K_mu_glu + y[2]);
	J( 0 , 1 ) = 0;
	J( 0 , 2 ) = -y[0]*y[2]*mu_max_1/std::pow(K_mu_glu + y[2], 2) + y[0]*mu_max_1/(K_mu_glu + y[2]);
	J( 0 , 3 ) = -y[0]*std::pow(k_I_1, nI_1)*k_omega_B_1/(std::pow(y[5], nI_1) + std::pow(k_I_1, nI_1));
	J( 0 , 4 ) = 0;
	J( 0 , 5 ) = y[3]*std::pow(y[5], nI_1)*y[0]*std::pow(k_I_1, nI_1)*k_omega_B_1*nI_1/(y[5]*std::pow(std::pow(y[5], nI_1) + std::pow(k_I_1, nI_1), 2));
	J( 1 , 0 ) = 0;
	J( 1 , 1 ) = -D + y[2]*mu_max_2/(K_mu_glu + y[2]);
	J( 1 , 2 ) = -y[1]*y[2]*mu_max_2/std::pow(K_mu_glu + y[2], 2) + y[1]*mu_max_2/(K_mu_glu + y[2]);
	J( 1 , 3 ) = 0;
	J( 1 , 4 ) = 0;
	J( 1 , 5 ) = 0;
	J( 2 , 0 ) = -y[2]*mu_max_1/(g_1*(K_mu_glu + y[2]));
	J( 2 , 1 ) = -y[2]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	J( 2 , 2 ) = -D + y[0]*y[2]*mu_max_1/(g_1*std::pow(K_mu_glu + y[2], 2)) - y[0]*mu_max_1/(g_1*(K_mu_glu + y[2])) + y[1]*y[2]*mu_max_2/(g_2*std::pow(K_mu_glu + y[2], 2)) - y[1]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	J( 2 , 3 ) = 0;
	J( 2 , 4 ) = 0;
	J( 2 , 5 ) = 0;
	J( 3 , 0 ) = kB_max_1;
	J( 3 , 1 ) = 0;
	J( 3 , 2 ) = 0;
	J( 3 , 3 ) = -D;
	J( 3 , 4 ) = 0;
	J( 3 , 5 ) = 0;
	J( 4 , 0 ) = kA_1;
	J( 4 , 1 ) = 0;
	J( 4 , 2 ) = 0;
	J( 4 , 3 ) = 0;
	J( 4 , 4 ) = -D;
	J( 4 , 5 ) = 0;
	J( 5 , 0 ) = 0;
	J( 5 , 1 ) = 0;
	J( 5 , 2 ) = (1.0/2.0)*y[5]*y[2]*mu_max_1/std::pow(K_mu_glu + y[2], 2) - 1.0/2.0*y[5]*mu_max_1/(K_mu_glu + y[2]);
	J( 5 , 3 ) = 0;
	J( 5 , 4 ) = -std::pow(y[4], 2*n_A_I_1)*kI_max_1*n_A_I_1/(y[4]*std::pow(std::pow(y[4], n_A_I_1) + std::pow(K_A_I_1, n_A_I_1), 2)) + std::pow(y[4], n_A_I_1)*kI_max_1*n_A_I_1/(y[4]*(std::pow(y[4], n_A_I_1) + std::pow(K_A_I_1, n_A_I_1)));
	J( 5 , 5 ) = -1.0/2.0*y[2]*mu_max_1/(K_mu_glu + y[2]);

	dfdt[0] = 0.0;
	dfdt[1] = 0.0;
	dfdt[2] = 0.0;
	dfdt[3] = 0.0;
	dfdt[4] = 0.0;
	dfdt[5] = 0.0;

}
void Models::model_26(const ublas_vec_t  &y , ublas_vec_t &dydt , double t, std::vector <double> &part_params)
{
	//Unpack parameters
	const double D = part_params[0];
	const double g_1 = part_params[1];
	const double g_2 = part_params[2];
	const double K_A_I_1 = part_params[3];
	const double k_I_1 = part_params[4];
	const double K_mu_glu = part_params[5];
	const double k_omega_B_1 = part_params[6];
	const double kA_1 = part_params[7];
	const double kB_max_1 = part_params[8];
	const double kI_max_1 = part_params[9];
	const double mu_max_1 = part_params[10];
	const double mu_max_2 = part_params[11];
	const double n_A_I_1 = part_params[12];
	const double nI_1 = part_params[13];
	const double S0_glu = part_params[14];

	//Species order is: N_1 N_2 S_glu B_1 A_1 I_1 
	dydt[0] = -y[3]*y[0]*std::pow(k_I_1, nI_1)*k_omega_B_1/(std::pow(y[5], nI_1) + std::pow(k_I_1, nI_1)) - D*y[0] + y[0]*y[2]*mu_max_1/(K_mu_glu + y[2]);
	dydt[1] = -y[3]*y[1]*k_omega_B_1 - D*y[1] + y[1]*y[2]*mu_max_2/(K_mu_glu + y[2]);
	dydt[2] = D*(S0_glu - y[2]) - y[0]*y[2]*mu_max_1/(g_1*(K_mu_glu + y[2])) - y[1]*y[2]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	dydt[3] = -y[3]*D + y[0]*kB_max_1;
	dydt[4] = -y[4]*D + y[0]*kA_1;
	dydt[5] = -1.0/2.0*y[5]*y[2]*mu_max_1/(K_mu_glu + y[2]) + std::pow(K_A_I_1, n_A_I_1)*kI_max_1/(std::pow(y[4], n_A_I_1) + std::pow(K_A_I_1, n_A_I_1));

}
void Models::jac_26(const ublas_vec_t & y , ublas_mat_t &J , const double &/* t*/ , ublas_vec_t &dfdt, std::vector <double> &part_params)
{
	//Unpack parameters
	const double D = part_params[0];
	const double g_1 = part_params[1];
	const double g_2 = part_params[2];
	const double K_A_I_1 = part_params[3];
	const double k_I_1 = part_params[4];
	const double K_mu_glu = part_params[5];
	const double k_omega_B_1 = part_params[6];
	const double kA_1 = part_params[7];
	const double kB_max_1 = part_params[8];
	const double kI_max_1 = part_params[9];
	const double mu_max_1 = part_params[10];
	const double mu_max_2 = part_params[11];
	const double n_A_I_1 = part_params[12];
	const double nI_1 = part_params[13];
	const double S0_glu = part_params[14];

	J( 0 , 0 ) = -y[3]*std::pow(k_I_1, nI_1)*k_omega_B_1/(std::pow(y[5], nI_1) + std::pow(k_I_1, nI_1)) - D + y[2]*mu_max_1/(K_mu_glu + y[2]);
	J( 0 , 1 ) = 0;
	J( 0 , 2 ) = -y[0]*y[2]*mu_max_1/std::pow(K_mu_glu + y[2], 2) + y[0]*mu_max_1/(K_mu_glu + y[2]);
	J( 0 , 3 ) = -y[0]*std::pow(k_I_1, nI_1)*k_omega_B_1/(std::pow(y[5], nI_1) + std::pow(k_I_1, nI_1));
	J( 0 , 4 ) = 0;
	J( 0 , 5 ) = y[3]*std::pow(y[5], nI_1)*y[0]*std::pow(k_I_1, nI_1)*k_omega_B_1*nI_1/(y[5]*std::pow(std::pow(y[5], nI_1) + std::pow(k_I_1, nI_1), 2));
	J( 1 , 0 ) = 0;
	J( 1 , 1 ) = -y[3]*k_omega_B_1 - D + y[2]*mu_max_2/(K_mu_glu + y[2]);
	J( 1 , 2 ) = -y[1]*y[2]*mu_max_2/std::pow(K_mu_glu + y[2], 2) + y[1]*mu_max_2/(K_mu_glu + y[2]);
	J( 1 , 3 ) = -y[1]*k_omega_B_1;
	J( 1 , 4 ) = 0;
	J( 1 , 5 ) = 0;
	J( 2 , 0 ) = -y[2]*mu_max_1/(g_1*(K_mu_glu + y[2]));
	J( 2 , 1 ) = -y[2]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	J( 2 , 2 ) = -D + y[0]*y[2]*mu_max_1/(g_1*std::pow(K_mu_glu + y[2], 2)) - y[0]*mu_max_1/(g_1*(K_mu_glu + y[2])) + y[1]*y[2]*mu_max_2/(g_2*std::pow(K_mu_glu + y[2], 2)) - y[1]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	J( 2 , 3 ) = 0;
	J( 2 , 4 ) = 0;
	J( 2 , 5 ) = 0;
	J( 3 , 0 ) = kB_max_1;
	J( 3 , 1 ) = 0;
	J( 3 , 2 ) = 0;
	J( 3 , 3 ) = -D;
	J( 3 , 4 ) = 0;
	J( 3 , 5 ) = 0;
	J( 4 , 0 ) = kA_1;
	J( 4 , 1 ) = 0;
	J( 4 , 2 ) = 0;
	J( 4 , 3 ) = 0;
	J( 4 , 4 ) = -D;
	J( 4 , 5 ) = 0;
	J( 5 , 0 ) = 0;
	J( 5 , 1 ) = 0;
	J( 5 , 2 ) = (1.0/2.0)*y[5]*y[2]*mu_max_1/std::pow(K_mu_glu + y[2], 2) - 1.0/2.0*y[5]*mu_max_1/(K_mu_glu + y[2]);
	J( 5 , 3 ) = 0;
	J( 5 , 4 ) = -std::pow(y[4], n_A_I_1)*std::pow(K_A_I_1, n_A_I_1)*kI_max_1*n_A_I_1/(y[4]*std::pow(std::pow(y[4], n_A_I_1) + std::pow(K_A_I_1, n_A_I_1), 2));
	J( 5 , 5 ) = -1.0/2.0*y[2]*mu_max_1/(K_mu_glu + y[2]);

	dfdt[0] = 0.0;
	dfdt[1] = 0.0;
	dfdt[2] = 0.0;
	dfdt[3] = 0.0;
	dfdt[4] = 0.0;
	dfdt[5] = 0.0;

}
void Models::model_27(const ublas_vec_t  &y , ublas_vec_t &dydt , double t, std::vector <double> &part_params)
{
	//Unpack parameters
	const double D = part_params[0];
	const double g_1 = part_params[1];
	const double g_2 = part_params[2];
	const double K_A_I_1 = part_params[3];
	const double k_I_1 = part_params[4];
	const double K_mu_glu = part_params[5];
	const double k_omega_B_1 = part_params[6];
	const double kA_1 = part_params[7];
	const double kB_max_1 = part_params[8];
	const double kI_max_1 = part_params[9];
	const double mu_max_1 = part_params[10];
	const double mu_max_2 = part_params[11];
	const double n_A_I_1 = part_params[12];
	const double nI_1 = part_params[13];
	const double S0_glu = part_params[14];

	//Species order is: N_1 N_2 S_glu B_1 A_1 I_1 
	dydt[0] = -y[3]*y[0]*std::pow(k_I_1, nI_1)*k_omega_B_1/(std::pow(y[5], nI_1) + std::pow(k_I_1, nI_1)) - D*y[0] + y[0]*y[2]*mu_max_1/(K_mu_glu + y[2]);
	dydt[1] = -D*y[1] + y[1]*y[2]*mu_max_2/(K_mu_glu + y[2]);
	dydt[2] = D*(S0_glu - y[2]) - y[0]*y[2]*mu_max_1/(g_1*(K_mu_glu + y[2])) - y[1]*y[2]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	dydt[3] = -y[3]*D + y[0]*kB_max_1;
	dydt[4] = -y[4]*D + y[0]*kA_1;
	dydt[5] = -1.0/2.0*y[5]*y[2]*mu_max_1/(K_mu_glu + y[2]) + std::pow(K_A_I_1, n_A_I_1)*kI_max_1/(std::pow(y[4], n_A_I_1) + std::pow(K_A_I_1, n_A_I_1));

}
void Models::jac_27(const ublas_vec_t & y , ublas_mat_t &J , const double &/* t*/ , ublas_vec_t &dfdt, std::vector <double> &part_params)
{
	//Unpack parameters
	const double D = part_params[0];
	const double g_1 = part_params[1];
	const double g_2 = part_params[2];
	const double K_A_I_1 = part_params[3];
	const double k_I_1 = part_params[4];
	const double K_mu_glu = part_params[5];
	const double k_omega_B_1 = part_params[6];
	const double kA_1 = part_params[7];
	const double kB_max_1 = part_params[8];
	const double kI_max_1 = part_params[9];
	const double mu_max_1 = part_params[10];
	const double mu_max_2 = part_params[11];
	const double n_A_I_1 = part_params[12];
	const double nI_1 = part_params[13];
	const double S0_glu = part_params[14];

	J( 0 , 0 ) = -y[3]*std::pow(k_I_1, nI_1)*k_omega_B_1/(std::pow(y[5], nI_1) + std::pow(k_I_1, nI_1)) - D + y[2]*mu_max_1/(K_mu_glu + y[2]);
	J( 0 , 1 ) = 0;
	J( 0 , 2 ) = -y[0]*y[2]*mu_max_1/std::pow(K_mu_glu + y[2], 2) + y[0]*mu_max_1/(K_mu_glu + y[2]);
	J( 0 , 3 ) = -y[0]*std::pow(k_I_1, nI_1)*k_omega_B_1/(std::pow(y[5], nI_1) + std::pow(k_I_1, nI_1));
	J( 0 , 4 ) = 0;
	J( 0 , 5 ) = y[3]*std::pow(y[5], nI_1)*y[0]*std::pow(k_I_1, nI_1)*k_omega_B_1*nI_1/(y[5]*std::pow(std::pow(y[5], nI_1) + std::pow(k_I_1, nI_1), 2));
	J( 1 , 0 ) = 0;
	J( 1 , 1 ) = -D + y[2]*mu_max_2/(K_mu_glu + y[2]);
	J( 1 , 2 ) = -y[1]*y[2]*mu_max_2/std::pow(K_mu_glu + y[2], 2) + y[1]*mu_max_2/(K_mu_glu + y[2]);
	J( 1 , 3 ) = 0;
	J( 1 , 4 ) = 0;
	J( 1 , 5 ) = 0;
	J( 2 , 0 ) = -y[2]*mu_max_1/(g_1*(K_mu_glu + y[2]));
	J( 2 , 1 ) = -y[2]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	J( 2 , 2 ) = -D + y[0]*y[2]*mu_max_1/(g_1*std::pow(K_mu_glu + y[2], 2)) - y[0]*mu_max_1/(g_1*(K_mu_glu + y[2])) + y[1]*y[2]*mu_max_2/(g_2*std::pow(K_mu_glu + y[2], 2)) - y[1]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	J( 2 , 3 ) = 0;
	J( 2 , 4 ) = 0;
	J( 2 , 5 ) = 0;
	J( 3 , 0 ) = kB_max_1;
	J( 3 , 1 ) = 0;
	J( 3 , 2 ) = 0;
	J( 3 , 3 ) = -D;
	J( 3 , 4 ) = 0;
	J( 3 , 5 ) = 0;
	J( 4 , 0 ) = kA_1;
	J( 4 , 1 ) = 0;
	J( 4 , 2 ) = 0;
	J( 4 , 3 ) = 0;
	J( 4 , 4 ) = -D;
	J( 4 , 5 ) = 0;
	J( 5 , 0 ) = 0;
	J( 5 , 1 ) = 0;
	J( 5 , 2 ) = (1.0/2.0)*y[5]*y[2]*mu_max_1/std::pow(K_mu_glu + y[2], 2) - 1.0/2.0*y[5]*mu_max_1/(K_mu_glu + y[2]);
	J( 5 , 3 ) = 0;
	J( 5 , 4 ) = -std::pow(y[4], n_A_I_1)*std::pow(K_A_I_1, n_A_I_1)*kI_max_1*n_A_I_1/(y[4]*std::pow(std::pow(y[4], n_A_I_1) + std::pow(K_A_I_1, n_A_I_1), 2));
	J( 5 , 5 ) = -1.0/2.0*y[2]*mu_max_1/(K_mu_glu + y[2]);

	dfdt[0] = 0.0;
	dfdt[1] = 0.0;
	dfdt[2] = 0.0;
	dfdt[3] = 0.0;
	dfdt[4] = 0.0;
	dfdt[5] = 0.0;

}
void Models::model_28(const ublas_vec_t  &y , ublas_vec_t &dydt , double t, std::vector <double> &part_params)
{
	//Unpack parameters
	const double D = part_params[0];
	const double g_1 = part_params[1];
	const double g_2 = part_params[2];
	const double K_A_I_1 = part_params[3];
	const double K_mu_glu = part_params[4];
	const double k_omega_B_1 = part_params[5];
	const double kA_1 = part_params[6];
	const double kB_max_1 = part_params[7];
	const double kI_max_1 = part_params[8];
	const double mu_max_1 = part_params[9];
	const double mu_max_2 = part_params[10];
	const double n_A_I_1 = part_params[11];
	const double S0_glu = part_params[12];

	//Species order is: N_1 N_2 S_glu B_1 A_1 I_1 
	dydt[0] = -D*y[0] + y[0]*y[2]*mu_max_1/(K_mu_glu + y[2]);
	dydt[1] = -y[3]*y[1]*k_omega_B_1 - D*y[1] + y[1]*y[2]*mu_max_2/(K_mu_glu + y[2]);
	dydt[2] = D*(S0_glu - y[2]) - y[0]*y[2]*mu_max_1/(g_1*(K_mu_glu + y[2])) - y[1]*y[2]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	dydt[3] = -y[3]*D + y[0]*kB_max_1;
	dydt[4] = -y[4]*D + y[0]*kA_1;
	dydt[5] = std::pow(y[4], n_A_I_1)*kI_max_1/(std::pow(y[4], n_A_I_1) + std::pow(K_A_I_1, n_A_I_1)) - 1.0/2.0*y[5]*y[2]*mu_max_1/(K_mu_glu + y[2]);

}
void Models::jac_28(const ublas_vec_t & y , ublas_mat_t &J , const double &/* t*/ , ublas_vec_t &dfdt, std::vector <double> &part_params)
{
	//Unpack parameters
	const double D = part_params[0];
	const double g_1 = part_params[1];
	const double g_2 = part_params[2];
	const double K_A_I_1 = part_params[3];
	const double K_mu_glu = part_params[4];
	const double k_omega_B_1 = part_params[5];
	const double kA_1 = part_params[6];
	const double kB_max_1 = part_params[7];
	const double kI_max_1 = part_params[8];
	const double mu_max_1 = part_params[9];
	const double mu_max_2 = part_params[10];
	const double n_A_I_1 = part_params[11];
	const double S0_glu = part_params[12];

	J( 0 , 0 ) = -D + y[2]*mu_max_1/(K_mu_glu + y[2]);
	J( 0 , 1 ) = 0;
	J( 0 , 2 ) = -y[0]*y[2]*mu_max_1/std::pow(K_mu_glu + y[2], 2) + y[0]*mu_max_1/(K_mu_glu + y[2]);
	J( 0 , 3 ) = 0;
	J( 0 , 4 ) = 0;
	J( 0 , 5 ) = 0;
	J( 1 , 0 ) = 0;
	J( 1 , 1 ) = -y[3]*k_omega_B_1 - D + y[2]*mu_max_2/(K_mu_glu + y[2]);
	J( 1 , 2 ) = -y[1]*y[2]*mu_max_2/std::pow(K_mu_glu + y[2], 2) + y[1]*mu_max_2/(K_mu_glu + y[2]);
	J( 1 , 3 ) = -y[1]*k_omega_B_1;
	J( 1 , 4 ) = 0;
	J( 1 , 5 ) = 0;
	J( 2 , 0 ) = -y[2]*mu_max_1/(g_1*(K_mu_glu + y[2]));
	J( 2 , 1 ) = -y[2]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	J( 2 , 2 ) = -D + y[0]*y[2]*mu_max_1/(g_1*std::pow(K_mu_glu + y[2], 2)) - y[0]*mu_max_1/(g_1*(K_mu_glu + y[2])) + y[1]*y[2]*mu_max_2/(g_2*std::pow(K_mu_glu + y[2], 2)) - y[1]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	J( 2 , 3 ) = 0;
	J( 2 , 4 ) = 0;
	J( 2 , 5 ) = 0;
	J( 3 , 0 ) = kB_max_1;
	J( 3 , 1 ) = 0;
	J( 3 , 2 ) = 0;
	J( 3 , 3 ) = -D;
	J( 3 , 4 ) = 0;
	J( 3 , 5 ) = 0;
	J( 4 , 0 ) = kA_1;
	J( 4 , 1 ) = 0;
	J( 4 , 2 ) = 0;
	J( 4 , 3 ) = 0;
	J( 4 , 4 ) = -D;
	J( 4 , 5 ) = 0;
	J( 5 , 0 ) = 0;
	J( 5 , 1 ) = 0;
	J( 5 , 2 ) = (1.0/2.0)*y[5]*y[2]*mu_max_1/std::pow(K_mu_glu + y[2], 2) - 1.0/2.0*y[5]*mu_max_1/(K_mu_glu + y[2]);
	J( 5 , 3 ) = 0;
	J( 5 , 4 ) = -std::pow(y[4], 2*n_A_I_1)*kI_max_1*n_A_I_1/(y[4]*std::pow(std::pow(y[4], n_A_I_1) + std::pow(K_A_I_1, n_A_I_1), 2)) + std::pow(y[4], n_A_I_1)*kI_max_1*n_A_I_1/(y[4]*(std::pow(y[4], n_A_I_1) + std::pow(K_A_I_1, n_A_I_1)));
	J( 5 , 5 ) = -1.0/2.0*y[2]*mu_max_1/(K_mu_glu + y[2]);

	dfdt[0] = 0.0;
	dfdt[1] = 0.0;
	dfdt[2] = 0.0;
	dfdt[3] = 0.0;
	dfdt[4] = 0.0;
	dfdt[5] = 0.0;

}
void Models::model_29(const ublas_vec_t  &y , ublas_vec_t &dydt , double t, std::vector <double> &part_params)
{
	//Unpack parameters
	const double D = part_params[0];
	const double g_1 = part_params[1];
	const double g_2 = part_params[2];
	const double K_A_I_1 = part_params[3];
	const double K_mu_glu = part_params[4];
	const double k_omega_B_1 = part_params[5];
	const double kA_1 = part_params[6];
	const double kB_max_1 = part_params[7];
	const double kI_max_1 = part_params[8];
	const double mu_max_1 = part_params[9];
	const double mu_max_2 = part_params[10];
	const double n_A_I_1 = part_params[11];
	const double S0_glu = part_params[12];

	//Species order is: N_1 N_2 S_glu B_1 A_1 I_1 
	dydt[0] = -D*y[0] + y[0]*y[2]*mu_max_1/(K_mu_glu + y[2]);
	dydt[1] = -y[3]*y[1]*k_omega_B_1 - D*y[1] + y[1]*y[2]*mu_max_2/(K_mu_glu + y[2]);
	dydt[2] = D*(S0_glu - y[2]) - y[0]*y[2]*mu_max_1/(g_1*(K_mu_glu + y[2])) - y[1]*y[2]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	dydt[3] = -y[3]*D + y[0]*kB_max_1;
	dydt[4] = -y[4]*D + y[0]*kA_1;
	dydt[5] = -1.0/2.0*y[5]*y[2]*mu_max_1/(K_mu_glu + y[2]) + std::pow(K_A_I_1, n_A_I_1)*kI_max_1/(std::pow(y[4], n_A_I_1) + std::pow(K_A_I_1, n_A_I_1));

}
void Models::jac_29(const ublas_vec_t & y , ublas_mat_t &J , const double &/* t*/ , ublas_vec_t &dfdt, std::vector <double> &part_params)
{
	//Unpack parameters
	const double D = part_params[0];
	const double g_1 = part_params[1];
	const double g_2 = part_params[2];
	const double K_A_I_1 = part_params[3];
	const double K_mu_glu = part_params[4];
	const double k_omega_B_1 = part_params[5];
	const double kA_1 = part_params[6];
	const double kB_max_1 = part_params[7];
	const double kI_max_1 = part_params[8];
	const double mu_max_1 = part_params[9];
	const double mu_max_2 = part_params[10];
	const double n_A_I_1 = part_params[11];
	const double S0_glu = part_params[12];

	J( 0 , 0 ) = -D + y[2]*mu_max_1/(K_mu_glu + y[2]);
	J( 0 , 1 ) = 0;
	J( 0 , 2 ) = -y[0]*y[2]*mu_max_1/std::pow(K_mu_glu + y[2], 2) + y[0]*mu_max_1/(K_mu_glu + y[2]);
	J( 0 , 3 ) = 0;
	J( 0 , 4 ) = 0;
	J( 0 , 5 ) = 0;
	J( 1 , 0 ) = 0;
	J( 1 , 1 ) = -y[3]*k_omega_B_1 - D + y[2]*mu_max_2/(K_mu_glu + y[2]);
	J( 1 , 2 ) = -y[1]*y[2]*mu_max_2/std::pow(K_mu_glu + y[2], 2) + y[1]*mu_max_2/(K_mu_glu + y[2]);
	J( 1 , 3 ) = -y[1]*k_omega_B_1;
	J( 1 , 4 ) = 0;
	J( 1 , 5 ) = 0;
	J( 2 , 0 ) = -y[2]*mu_max_1/(g_1*(K_mu_glu + y[2]));
	J( 2 , 1 ) = -y[2]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	J( 2 , 2 ) = -D + y[0]*y[2]*mu_max_1/(g_1*std::pow(K_mu_glu + y[2], 2)) - y[0]*mu_max_1/(g_1*(K_mu_glu + y[2])) + y[1]*y[2]*mu_max_2/(g_2*std::pow(K_mu_glu + y[2], 2)) - y[1]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	J( 2 , 3 ) = 0;
	J( 2 , 4 ) = 0;
	J( 2 , 5 ) = 0;
	J( 3 , 0 ) = kB_max_1;
	J( 3 , 1 ) = 0;
	J( 3 , 2 ) = 0;
	J( 3 , 3 ) = -D;
	J( 3 , 4 ) = 0;
	J( 3 , 5 ) = 0;
	J( 4 , 0 ) = kA_1;
	J( 4 , 1 ) = 0;
	J( 4 , 2 ) = 0;
	J( 4 , 3 ) = 0;
	J( 4 , 4 ) = -D;
	J( 4 , 5 ) = 0;
	J( 5 , 0 ) = 0;
	J( 5 , 1 ) = 0;
	J( 5 , 2 ) = (1.0/2.0)*y[5]*y[2]*mu_max_1/std::pow(K_mu_glu + y[2], 2) - 1.0/2.0*y[5]*mu_max_1/(K_mu_glu + y[2]);
	J( 5 , 3 ) = 0;
	J( 5 , 4 ) = -std::pow(y[4], n_A_I_1)*std::pow(K_A_I_1, n_A_I_1)*kI_max_1*n_A_I_1/(y[4]*std::pow(std::pow(y[4], n_A_I_1) + std::pow(K_A_I_1, n_A_I_1), 2));
	J( 5 , 5 ) = -1.0/2.0*y[2]*mu_max_1/(K_mu_glu + y[2]);

	dfdt[0] = 0.0;
	dfdt[1] = 0.0;
	dfdt[2] = 0.0;
	dfdt[3] = 0.0;
	dfdt[4] = 0.0;
	dfdt[5] = 0.0;

}
void Models::model_30(const ublas_vec_t  &y , ublas_vec_t &dydt , double t, std::vector <double> &part_params)
{
	//Unpack parameters
	const double D = part_params[0];
	const double g_1 = part_params[1];
	const double g_2 = part_params[2];
	const double k_I_1 = part_params[3];
	const double K_mu_glu = part_params[4];
	const double k_omega_B_1 = part_params[5];
	const double kB_max_1 = part_params[6];
	const double kI_max_1 = part_params[7];
	const double mu_max_1 = part_params[8];
	const double mu_max_2 = part_params[9];
	const double nI_1 = part_params[10];
	const double S0_glu = part_params[11];

	//Species order is: N_1 N_2 S_glu B_1 I_1 
	dydt[0] = -y[3]*y[0]*std::pow(k_I_1, nI_1)*k_omega_B_1/(std::pow(y[4], nI_1) + std::pow(k_I_1, nI_1)) - D*y[0] + y[0]*y[2]*mu_max_1/(K_mu_glu + y[2]);
	dydt[1] = -y[3]*y[1]*k_omega_B_1 - D*y[1] + y[1]*y[2]*mu_max_2/(K_mu_glu + y[2]);
	dydt[2] = D*(S0_glu - y[2]) - y[0]*y[2]*mu_max_1/(g_1*(K_mu_glu + y[2])) - y[1]*y[2]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	dydt[3] = -y[3]*D + y[0]*kB_max_1;
	dydt[4] = -1.0/2.0*y[4]*y[2]*mu_max_1/(K_mu_glu + y[2]) + kI_max_1;

}
void Models::jac_30(const ublas_vec_t & y , ublas_mat_t &J , const double &/* t*/ , ublas_vec_t &dfdt, std::vector <double> &part_params)
{
	//Unpack parameters
	const double D = part_params[0];
	const double g_1 = part_params[1];
	const double g_2 = part_params[2];
	const double k_I_1 = part_params[3];
	const double K_mu_glu = part_params[4];
	const double k_omega_B_1 = part_params[5];
	const double kB_max_1 = part_params[6];
	const double kI_max_1 = part_params[7];
	const double mu_max_1 = part_params[8];
	const double mu_max_2 = part_params[9];
	const double nI_1 = part_params[10];
	const double S0_glu = part_params[11];

	J( 0 , 0 ) = -y[3]*std::pow(k_I_1, nI_1)*k_omega_B_1/(std::pow(y[4], nI_1) + std::pow(k_I_1, nI_1)) - D + y[2]*mu_max_1/(K_mu_glu + y[2]);
	J( 0 , 1 ) = 0;
	J( 0 , 2 ) = -y[0]*y[2]*mu_max_1/std::pow(K_mu_glu + y[2], 2) + y[0]*mu_max_1/(K_mu_glu + y[2]);
	J( 0 , 3 ) = -y[0]*std::pow(k_I_1, nI_1)*k_omega_B_1/(std::pow(y[4], nI_1) + std::pow(k_I_1, nI_1));
	J( 0 , 4 ) = y[3]*std::pow(y[4], nI_1)*y[0]*std::pow(k_I_1, nI_1)*k_omega_B_1*nI_1/(y[4]*std::pow(std::pow(y[4], nI_1) + std::pow(k_I_1, nI_1), 2));
	J( 1 , 0 ) = 0;
	J( 1 , 1 ) = -y[3]*k_omega_B_1 - D + y[2]*mu_max_2/(K_mu_glu + y[2]);
	J( 1 , 2 ) = -y[1]*y[2]*mu_max_2/std::pow(K_mu_glu + y[2], 2) + y[1]*mu_max_2/(K_mu_glu + y[2]);
	J( 1 , 3 ) = -y[1]*k_omega_B_1;
	J( 1 , 4 ) = 0;
	J( 2 , 0 ) = -y[2]*mu_max_1/(g_1*(K_mu_glu + y[2]));
	J( 2 , 1 ) = -y[2]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	J( 2 , 2 ) = -D + y[0]*y[2]*mu_max_1/(g_1*std::pow(K_mu_glu + y[2], 2)) - y[0]*mu_max_1/(g_1*(K_mu_glu + y[2])) + y[1]*y[2]*mu_max_2/(g_2*std::pow(K_mu_glu + y[2], 2)) - y[1]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	J( 2 , 3 ) = 0;
	J( 2 , 4 ) = 0;
	J( 3 , 0 ) = kB_max_1;
	J( 3 , 1 ) = 0;
	J( 3 , 2 ) = 0;
	J( 3 , 3 ) = -D;
	J( 3 , 4 ) = 0;
	J( 4 , 0 ) = 0;
	J( 4 , 1 ) = 0;
	J( 4 , 2 ) = (1.0/2.0)*y[4]*y[2]*mu_max_1/std::pow(K_mu_glu + y[2], 2) - 1.0/2.0*y[4]*mu_max_1/(K_mu_glu + y[2]);
	J( 4 , 3 ) = 0;
	J( 4 , 4 ) = -1.0/2.0*y[2]*mu_max_1/(K_mu_glu + y[2]);

	dfdt[0] = 0.0;
	dfdt[1] = 0.0;
	dfdt[2] = 0.0;
	dfdt[3] = 0.0;
	dfdt[4] = 0.0;

}
void Models::model_31(const ublas_vec_t  &y , ublas_vec_t &dydt , double t, std::vector <double> &part_params)
{
	//Unpack parameters
	const double D = part_params[0];
	const double g_1 = part_params[1];
	const double g_2 = part_params[2];
	const double k_I_1 = part_params[3];
	const double K_mu_glu = part_params[4];
	const double k_omega_B_1 = part_params[5];
	const double kB_max_1 = part_params[6];
	const double kI_max_1 = part_params[7];
	const double mu_max_1 = part_params[8];
	const double mu_max_2 = part_params[9];
	const double nI_1 = part_params[10];
	const double S0_glu = part_params[11];

	//Species order is: N_1 N_2 S_glu B_1 I_1 
	dydt[0] = -y[3]*y[0]*std::pow(k_I_1, nI_1)*k_omega_B_1/(std::pow(y[4], nI_1) + std::pow(k_I_1, nI_1)) - D*y[0] + y[0]*y[2]*mu_max_1/(K_mu_glu + y[2]);
	dydt[1] = -D*y[1] + y[1]*y[2]*mu_max_2/(K_mu_glu + y[2]);
	dydt[2] = D*(S0_glu - y[2]) - y[0]*y[2]*mu_max_1/(g_1*(K_mu_glu + y[2])) - y[1]*y[2]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	dydt[3] = -y[3]*D + y[0]*kB_max_1;
	dydt[4] = -1.0/2.0*y[4]*y[2]*mu_max_1/(K_mu_glu + y[2]) + kI_max_1;

}
void Models::jac_31(const ublas_vec_t & y , ublas_mat_t &J , const double &/* t*/ , ublas_vec_t &dfdt, std::vector <double> &part_params)
{
	//Unpack parameters
	const double D = part_params[0];
	const double g_1 = part_params[1];
	const double g_2 = part_params[2];
	const double k_I_1 = part_params[3];
	const double K_mu_glu = part_params[4];
	const double k_omega_B_1 = part_params[5];
	const double kB_max_1 = part_params[6];
	const double kI_max_1 = part_params[7];
	const double mu_max_1 = part_params[8];
	const double mu_max_2 = part_params[9];
	const double nI_1 = part_params[10];
	const double S0_glu = part_params[11];

	J( 0 , 0 ) = -y[3]*std::pow(k_I_1, nI_1)*k_omega_B_1/(std::pow(y[4], nI_1) + std::pow(k_I_1, nI_1)) - D + y[2]*mu_max_1/(K_mu_glu + y[2]);
	J( 0 , 1 ) = 0;
	J( 0 , 2 ) = -y[0]*y[2]*mu_max_1/std::pow(K_mu_glu + y[2], 2) + y[0]*mu_max_1/(K_mu_glu + y[2]);
	J( 0 , 3 ) = -y[0]*std::pow(k_I_1, nI_1)*k_omega_B_1/(std::pow(y[4], nI_1) + std::pow(k_I_1, nI_1));
	J( 0 , 4 ) = y[3]*std::pow(y[4], nI_1)*y[0]*std::pow(k_I_1, nI_1)*k_omega_B_1*nI_1/(y[4]*std::pow(std::pow(y[4], nI_1) + std::pow(k_I_1, nI_1), 2));
	J( 1 , 0 ) = 0;
	J( 1 , 1 ) = -D + y[2]*mu_max_2/(K_mu_glu + y[2]);
	J( 1 , 2 ) = -y[1]*y[2]*mu_max_2/std::pow(K_mu_glu + y[2], 2) + y[1]*mu_max_2/(K_mu_glu + y[2]);
	J( 1 , 3 ) = 0;
	J( 1 , 4 ) = 0;
	J( 2 , 0 ) = -y[2]*mu_max_1/(g_1*(K_mu_glu + y[2]));
	J( 2 , 1 ) = -y[2]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	J( 2 , 2 ) = -D + y[0]*y[2]*mu_max_1/(g_1*std::pow(K_mu_glu + y[2], 2)) - y[0]*mu_max_1/(g_1*(K_mu_glu + y[2])) + y[1]*y[2]*mu_max_2/(g_2*std::pow(K_mu_glu + y[2], 2)) - y[1]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	J( 2 , 3 ) = 0;
	J( 2 , 4 ) = 0;
	J( 3 , 0 ) = kB_max_1;
	J( 3 , 1 ) = 0;
	J( 3 , 2 ) = 0;
	J( 3 , 3 ) = -D;
	J( 3 , 4 ) = 0;
	J( 4 , 0 ) = 0;
	J( 4 , 1 ) = 0;
	J( 4 , 2 ) = (1.0/2.0)*y[4]*y[2]*mu_max_1/std::pow(K_mu_glu + y[2], 2) - 1.0/2.0*y[4]*mu_max_1/(K_mu_glu + y[2]);
	J( 4 , 3 ) = 0;
	J( 4 , 4 ) = -1.0/2.0*y[2]*mu_max_1/(K_mu_glu + y[2]);

	dfdt[0] = 0.0;
	dfdt[1] = 0.0;
	dfdt[2] = 0.0;
	dfdt[3] = 0.0;
	dfdt[4] = 0.0;

}
void Models::model_32(const ublas_vec_t  &y , ublas_vec_t &dydt , double t, std::vector <double> &part_params)
{
	//Unpack parameters
	const double D = part_params[0];
	const double g_1 = part_params[1];
	const double g_2 = part_params[2];
	const double K_mu_glu = part_params[3];
	const double k_omega_B_1 = part_params[4];
	const double kB_max_1 = part_params[5];
	const double mu_max_1 = part_params[6];
	const double mu_max_2 = part_params[7];
	const double S0_glu = part_params[8];

	//Species order is: N_1 N_2 S_glu B_1 
	dydt[0] = -y[3]*y[0]*k_omega_B_1 - D*y[0] + y[0]*y[2]*mu_max_1/(K_mu_glu + y[2]);
	dydt[1] = -y[3]*y[1]*k_omega_B_1 - D*y[1] + y[1]*y[2]*mu_max_2/(K_mu_glu + y[2]);
	dydt[2] = D*(S0_glu - y[2]) - y[0]*y[2]*mu_max_1/(g_1*(K_mu_glu + y[2])) - y[1]*y[2]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	dydt[3] = -y[3]*D + y[0]*kB_max_1;

}
void Models::jac_32(const ublas_vec_t & y , ublas_mat_t &J , const double &/* t*/ , ublas_vec_t &dfdt, std::vector <double> &part_params)
{
	//Unpack parameters
	const double D = part_params[0];
	const double g_1 = part_params[1];
	const double g_2 = part_params[2];
	const double K_mu_glu = part_params[3];
	const double k_omega_B_1 = part_params[4];
	const double kB_max_1 = part_params[5];
	const double mu_max_1 = part_params[6];
	const double mu_max_2 = part_params[7];
	const double S0_glu = part_params[8];

	J( 0 , 0 ) = -y[3]*k_omega_B_1 - D + y[2]*mu_max_1/(K_mu_glu + y[2]);
	J( 0 , 1 ) = 0;
	J( 0 , 2 ) = -y[0]*y[2]*mu_max_1/std::pow(K_mu_glu + y[2], 2) + y[0]*mu_max_1/(K_mu_glu + y[2]);
	J( 0 , 3 ) = -y[0]*k_omega_B_1;
	J( 1 , 0 ) = 0;
	J( 1 , 1 ) = -y[3]*k_omega_B_1 - D + y[2]*mu_max_2/(K_mu_glu + y[2]);
	J( 1 , 2 ) = -y[1]*y[2]*mu_max_2/std::pow(K_mu_glu + y[2], 2) + y[1]*mu_max_2/(K_mu_glu + y[2]);
	J( 1 , 3 ) = -y[1]*k_omega_B_1;
	J( 2 , 0 ) = -y[2]*mu_max_1/(g_1*(K_mu_glu + y[2]));
	J( 2 , 1 ) = -y[2]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	J( 2 , 2 ) = -D + y[0]*y[2]*mu_max_1/(g_1*std::pow(K_mu_glu + y[2], 2)) - y[0]*mu_max_1/(g_1*(K_mu_glu + y[2])) + y[1]*y[2]*mu_max_2/(g_2*std::pow(K_mu_glu + y[2], 2)) - y[1]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	J( 2 , 3 ) = 0;
	J( 3 , 0 ) = kB_max_1;
	J( 3 , 1 ) = 0;
	J( 3 , 2 ) = 0;
	J( 3 , 3 ) = -D;

	dfdt[0] = 0.0;
	dfdt[1] = 0.0;
	dfdt[2] = 0.0;
	dfdt[3] = 0.0;

}
void Models::model_33(const ublas_vec_t  &y , ublas_vec_t &dydt , double t, std::vector <double> &part_params)
{
	//Unpack parameters
	const double D = part_params[0];
	const double g_1 = part_params[1];
	const double g_2 = part_params[2];
	const double K_mu_glu = part_params[3];
	const double k_omega_B_1 = part_params[4];
	const double kB_max_1 = part_params[5];
	const double mu_max_1 = part_params[6];
	const double mu_max_2 = part_params[7];
	const double S0_glu = part_params[8];

	//Species order is: N_1 N_2 S_glu B_1 
	dydt[0] = -y[3]*y[0]*k_omega_B_1 - D*y[0] + y[0]*y[2]*mu_max_1/(K_mu_glu + y[2]);
	dydt[1] = -D*y[1] + y[1]*y[2]*mu_max_2/(K_mu_glu + y[2]);
	dydt[2] = D*(S0_glu - y[2]) - y[0]*y[2]*mu_max_1/(g_1*(K_mu_glu + y[2])) - y[1]*y[2]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	dydt[3] = -y[3]*D + y[0]*kB_max_1;

}
void Models::jac_33(const ublas_vec_t & y , ublas_mat_t &J , const double &/* t*/ , ublas_vec_t &dfdt, std::vector <double> &part_params)
{
	//Unpack parameters
	const double D = part_params[0];
	const double g_1 = part_params[1];
	const double g_2 = part_params[2];
	const double K_mu_glu = part_params[3];
	const double k_omega_B_1 = part_params[4];
	const double kB_max_1 = part_params[5];
	const double mu_max_1 = part_params[6];
	const double mu_max_2 = part_params[7];
	const double S0_glu = part_params[8];

	J( 0 , 0 ) = -y[3]*k_omega_B_1 - D + y[2]*mu_max_1/(K_mu_glu + y[2]);
	J( 0 , 1 ) = 0;
	J( 0 , 2 ) = -y[0]*y[2]*mu_max_1/std::pow(K_mu_glu + y[2], 2) + y[0]*mu_max_1/(K_mu_glu + y[2]);
	J( 0 , 3 ) = -y[0]*k_omega_B_1;
	J( 1 , 0 ) = 0;
	J( 1 , 1 ) = -D + y[2]*mu_max_2/(K_mu_glu + y[2]);
	J( 1 , 2 ) = -y[1]*y[2]*mu_max_2/std::pow(K_mu_glu + y[2], 2) + y[1]*mu_max_2/(K_mu_glu + y[2]);
	J( 1 , 3 ) = 0;
	J( 2 , 0 ) = -y[2]*mu_max_1/(g_1*(K_mu_glu + y[2]));
	J( 2 , 1 ) = -y[2]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	J( 2 , 2 ) = -D + y[0]*y[2]*mu_max_1/(g_1*std::pow(K_mu_glu + y[2], 2)) - y[0]*mu_max_1/(g_1*(K_mu_glu + y[2])) + y[1]*y[2]*mu_max_2/(g_2*std::pow(K_mu_glu + y[2], 2)) - y[1]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	J( 2 , 3 ) = 0;
	J( 3 , 0 ) = kB_max_1;
	J( 3 , 1 ) = 0;
	J( 3 , 2 ) = 0;
	J( 3 , 3 ) = -D;

	dfdt[0] = 0.0;
	dfdt[1] = 0.0;
	dfdt[2] = 0.0;
	dfdt[3] = 0.0;

}
void Models::model_34(const ublas_vec_t  &y , ublas_vec_t &dydt , double t, std::vector <double> &part_params)
{
	//Unpack parameters
	const double D = part_params[0];
	const double g_1 = part_params[1];
	const double g_2 = part_params[2];
	const double K_mu_glu = part_params[3];
	const double k_omega_B_1 = part_params[4];
	const double kB_max_1 = part_params[5];
	const double kI_max_1 = part_params[6];
	const double mu_max_1 = part_params[7];
	const double mu_max_2 = part_params[8];
	const double S0_glu = part_params[9];

	//Species order is: N_1 N_2 S_glu B_1 I_1 
	dydt[0] = -D*y[0] + y[0]*y[2]*mu_max_1/(K_mu_glu + y[2]);
	dydt[1] = -y[3]*y[1]*k_omega_B_1 - D*y[1] + y[1]*y[2]*mu_max_2/(K_mu_glu + y[2]);
	dydt[2] = D*(S0_glu - y[2]) - y[0]*y[2]*mu_max_1/(g_1*(K_mu_glu + y[2])) - y[1]*y[2]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	dydt[3] = -y[3]*D + y[0]*kB_max_1;
	dydt[4] = -1.0/2.0*y[4]*y[2]*mu_max_1/(K_mu_glu + y[2]) + kI_max_1;

}
void Models::jac_34(const ublas_vec_t & y , ublas_mat_t &J , const double &/* t*/ , ublas_vec_t &dfdt, std::vector <double> &part_params)
{
	//Unpack parameters
	const double D = part_params[0];
	const double g_1 = part_params[1];
	const double g_2 = part_params[2];
	const double K_mu_glu = part_params[3];
	const double k_omega_B_1 = part_params[4];
	const double kB_max_1 = part_params[5];
	const double kI_max_1 = part_params[6];
	const double mu_max_1 = part_params[7];
	const double mu_max_2 = part_params[8];
	const double S0_glu = part_params[9];

	J( 0 , 0 ) = -D + y[2]*mu_max_1/(K_mu_glu + y[2]);
	J( 0 , 1 ) = 0;
	J( 0 , 2 ) = -y[0]*y[2]*mu_max_1/std::pow(K_mu_glu + y[2], 2) + y[0]*mu_max_1/(K_mu_glu + y[2]);
	J( 0 , 3 ) = 0;
	J( 0 , 4 ) = 0;
	J( 1 , 0 ) = 0;
	J( 1 , 1 ) = -y[3]*k_omega_B_1 - D + y[2]*mu_max_2/(K_mu_glu + y[2]);
	J( 1 , 2 ) = -y[1]*y[2]*mu_max_2/std::pow(K_mu_glu + y[2], 2) + y[1]*mu_max_2/(K_mu_glu + y[2]);
	J( 1 , 3 ) = -y[1]*k_omega_B_1;
	J( 1 , 4 ) = 0;
	J( 2 , 0 ) = -y[2]*mu_max_1/(g_1*(K_mu_glu + y[2]));
	J( 2 , 1 ) = -y[2]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	J( 2 , 2 ) = -D + y[0]*y[2]*mu_max_1/(g_1*std::pow(K_mu_glu + y[2], 2)) - y[0]*mu_max_1/(g_1*(K_mu_glu + y[2])) + y[1]*y[2]*mu_max_2/(g_2*std::pow(K_mu_glu + y[2], 2)) - y[1]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	J( 2 , 3 ) = 0;
	J( 2 , 4 ) = 0;
	J( 3 , 0 ) = kB_max_1;
	J( 3 , 1 ) = 0;
	J( 3 , 2 ) = 0;
	J( 3 , 3 ) = -D;
	J( 3 , 4 ) = 0;
	J( 4 , 0 ) = 0;
	J( 4 , 1 ) = 0;
	J( 4 , 2 ) = (1.0/2.0)*y[4]*y[2]*mu_max_1/std::pow(K_mu_glu + y[2], 2) - 1.0/2.0*y[4]*mu_max_1/(K_mu_glu + y[2]);
	J( 4 , 3 ) = 0;
	J( 4 , 4 ) = -1.0/2.0*y[2]*mu_max_1/(K_mu_glu + y[2]);

	dfdt[0] = 0.0;
	dfdt[1] = 0.0;
	dfdt[2] = 0.0;
	dfdt[3] = 0.0;
	dfdt[4] = 0.0;

}
void Models::model_35(const ublas_vec_t  &y , ublas_vec_t &dydt , double t, std::vector <double> &part_params)
{
	//Unpack parameters
	const double D = part_params[0];
	const double g_1 = part_params[1];
	const double g_2 = part_params[2];
	const double K_mu_glu = part_params[3];
	const double k_omega_B_1 = part_params[4];
	const double kB_max_1 = part_params[5];
	const double mu_max_1 = part_params[6];
	const double mu_max_2 = part_params[7];
	const double S0_glu = part_params[8];

	//Species order is: N_1 N_2 S_glu B_1 
	dydt[0] = -D*y[0] + y[0]*y[2]*mu_max_1/(K_mu_glu + y[2]);
	dydt[1] = -y[3]*y[1]*k_omega_B_1 - D*y[1] + y[1]*y[2]*mu_max_2/(K_mu_glu + y[2]);
	dydt[2] = D*(S0_glu - y[2]) - y[0]*y[2]*mu_max_1/(g_1*(K_mu_glu + y[2])) - y[1]*y[2]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	dydt[3] = -y[3]*D + y[0]*kB_max_1;

}
void Models::jac_35(const ublas_vec_t & y , ublas_mat_t &J , const double &/* t*/ , ublas_vec_t &dfdt, std::vector <double> &part_params)
{
	//Unpack parameters
	const double D = part_params[0];
	const double g_1 = part_params[1];
	const double g_2 = part_params[2];
	const double K_mu_glu = part_params[3];
	const double k_omega_B_1 = part_params[4];
	const double kB_max_1 = part_params[5];
	const double mu_max_1 = part_params[6];
	const double mu_max_2 = part_params[7];
	const double S0_glu = part_params[8];

	J( 0 , 0 ) = -D + y[2]*mu_max_1/(K_mu_glu + y[2]);
	J( 0 , 1 ) = 0;
	J( 0 , 2 ) = -y[0]*y[2]*mu_max_1/std::pow(K_mu_glu + y[2], 2) + y[0]*mu_max_1/(K_mu_glu + y[2]);
	J( 0 , 3 ) = 0;
	J( 1 , 0 ) = 0;
	J( 1 , 1 ) = -y[3]*k_omega_B_1 - D + y[2]*mu_max_2/(K_mu_glu + y[2]);
	J( 1 , 2 ) = -y[1]*y[2]*mu_max_2/std::pow(K_mu_glu + y[2], 2) + y[1]*mu_max_2/(K_mu_glu + y[2]);
	J( 1 , 3 ) = -y[1]*k_omega_B_1;
	J( 2 , 0 ) = -y[2]*mu_max_1/(g_1*(K_mu_glu + y[2]));
	J( 2 , 1 ) = -y[2]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	J( 2 , 2 ) = -D + y[0]*y[2]*mu_max_1/(g_1*std::pow(K_mu_glu + y[2], 2)) - y[0]*mu_max_1/(g_1*(K_mu_glu + y[2])) + y[1]*y[2]*mu_max_2/(g_2*std::pow(K_mu_glu + y[2], 2)) - y[1]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	J( 2 , 3 ) = 0;
	J( 3 , 0 ) = kB_max_1;
	J( 3 , 1 ) = 0;
	J( 3 , 2 ) = 0;
	J( 3 , 3 ) = -D;

	dfdt[0] = 0.0;
	dfdt[1] = 0.0;
	dfdt[2] = 0.0;
	dfdt[3] = 0.0;

}
void Models::model_36(const ublas_vec_t  &y , ublas_vec_t &dydt , double t, std::vector <double> &part_params)
{
	//Unpack parameters
	const double D = part_params[0];
	const double g_1 = part_params[1];
	const double g_2 = part_params[2];
	const double K_A_I_1 = part_params[3];
	const double K_mu_glu = part_params[4];
	const double kA_1 = part_params[5];
	const double kI_max_1 = part_params[6];
	const double mu_max_1 = part_params[7];
	const double mu_max_2 = part_params[8];
	const double n_A_I_1 = part_params[9];
	const double S0_glu = part_params[10];

	//Species order is: N_1 N_2 S_glu A_1 I_1 
	dydt[0] = -D*y[0] + y[0]*y[2]*mu_max_1/(K_mu_glu + y[2]);
	dydt[1] = -D*y[1] + y[1]*y[2]*mu_max_2/(K_mu_glu + y[2]);
	dydt[2] = D*(S0_glu - y[2]) - y[0]*y[2]*mu_max_1/(g_1*(K_mu_glu + y[2])) - y[1]*y[2]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	dydt[3] = -y[3]*D + y[0]*kA_1;
	dydt[4] = std::pow(y[3], n_A_I_1)*kI_max_1/(std::pow(y[3], n_A_I_1) + std::pow(K_A_I_1, n_A_I_1)) - 1.0/2.0*y[4]*y[2]*mu_max_1/(K_mu_glu + y[2]);

}
void Models::jac_36(const ublas_vec_t & y , ublas_mat_t &J , const double &/* t*/ , ublas_vec_t &dfdt, std::vector <double> &part_params)
{
	//Unpack parameters
	const double D = part_params[0];
	const double g_1 = part_params[1];
	const double g_2 = part_params[2];
	const double K_A_I_1 = part_params[3];
	const double K_mu_glu = part_params[4];
	const double kA_1 = part_params[5];
	const double kI_max_1 = part_params[6];
	const double mu_max_1 = part_params[7];
	const double mu_max_2 = part_params[8];
	const double n_A_I_1 = part_params[9];
	const double S0_glu = part_params[10];

	J( 0 , 0 ) = -D + y[2]*mu_max_1/(K_mu_glu + y[2]);
	J( 0 , 1 ) = 0;
	J( 0 , 2 ) = -y[0]*y[2]*mu_max_1/std::pow(K_mu_glu + y[2], 2) + y[0]*mu_max_1/(K_mu_glu + y[2]);
	J( 0 , 3 ) = 0;
	J( 0 , 4 ) = 0;
	J( 1 , 0 ) = 0;
	J( 1 , 1 ) = -D + y[2]*mu_max_2/(K_mu_glu + y[2]);
	J( 1 , 2 ) = -y[1]*y[2]*mu_max_2/std::pow(K_mu_glu + y[2], 2) + y[1]*mu_max_2/(K_mu_glu + y[2]);
	J( 1 , 3 ) = 0;
	J( 1 , 4 ) = 0;
	J( 2 , 0 ) = -y[2]*mu_max_1/(g_1*(K_mu_glu + y[2]));
	J( 2 , 1 ) = -y[2]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	J( 2 , 2 ) = -D + y[0]*y[2]*mu_max_1/(g_1*std::pow(K_mu_glu + y[2], 2)) - y[0]*mu_max_1/(g_1*(K_mu_glu + y[2])) + y[1]*y[2]*mu_max_2/(g_2*std::pow(K_mu_glu + y[2], 2)) - y[1]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	J( 2 , 3 ) = 0;
	J( 2 , 4 ) = 0;
	J( 3 , 0 ) = kA_1;
	J( 3 , 1 ) = 0;
	J( 3 , 2 ) = 0;
	J( 3 , 3 ) = -D;
	J( 3 , 4 ) = 0;
	J( 4 , 0 ) = 0;
	J( 4 , 1 ) = 0;
	J( 4 , 2 ) = (1.0/2.0)*y[4]*y[2]*mu_max_1/std::pow(K_mu_glu + y[2], 2) - 1.0/2.0*y[4]*mu_max_1/(K_mu_glu + y[2]);
	J( 4 , 3 ) = -std::pow(y[3], 2*n_A_I_1)*kI_max_1*n_A_I_1/(y[3]*std::pow(std::pow(y[3], n_A_I_1) + std::pow(K_A_I_1, n_A_I_1), 2)) + std::pow(y[3], n_A_I_1)*kI_max_1*n_A_I_1/(y[3]*(std::pow(y[3], n_A_I_1) + std::pow(K_A_I_1, n_A_I_1)));
	J( 4 , 4 ) = -1.0/2.0*y[2]*mu_max_1/(K_mu_glu + y[2]);

	dfdt[0] = 0.0;
	dfdt[1] = 0.0;
	dfdt[2] = 0.0;
	dfdt[3] = 0.0;
	dfdt[4] = 0.0;

}
void Models::model_37(const ublas_vec_t  &y , ublas_vec_t &dydt , double t, std::vector <double> &part_params)
{
	//Unpack parameters
	const double D = part_params[0];
	const double g_1 = part_params[1];
	const double g_2 = part_params[2];
	const double K_A_I_1 = part_params[3];
	const double K_mu_glu = part_params[4];
	const double kA_1 = part_params[5];
	const double kI_max_1 = part_params[6];
	const double mu_max_1 = part_params[7];
	const double mu_max_2 = part_params[8];
	const double n_A_I_1 = part_params[9];
	const double S0_glu = part_params[10];

	//Species order is: N_1 N_2 S_glu A_1 I_1 
	dydt[0] = -D*y[0] + y[0]*y[2]*mu_max_1/(K_mu_glu + y[2]);
	dydt[1] = -D*y[1] + y[1]*y[2]*mu_max_2/(K_mu_glu + y[2]);
	dydt[2] = D*(S0_glu - y[2]) - y[0]*y[2]*mu_max_1/(g_1*(K_mu_glu + y[2])) - y[1]*y[2]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	dydt[3] = -y[3]*D + y[0]*kA_1;
	dydt[4] = -1.0/2.0*y[4]*y[2]*mu_max_1/(K_mu_glu + y[2]) + std::pow(K_A_I_1, n_A_I_1)*kI_max_1/(std::pow(y[3], n_A_I_1) + std::pow(K_A_I_1, n_A_I_1));

}
void Models::jac_37(const ublas_vec_t & y , ublas_mat_t &J , const double &/* t*/ , ublas_vec_t &dfdt, std::vector <double> &part_params)
{
	//Unpack parameters
	const double D = part_params[0];
	const double g_1 = part_params[1];
	const double g_2 = part_params[2];
	const double K_A_I_1 = part_params[3];
	const double K_mu_glu = part_params[4];
	const double kA_1 = part_params[5];
	const double kI_max_1 = part_params[6];
	const double mu_max_1 = part_params[7];
	const double mu_max_2 = part_params[8];
	const double n_A_I_1 = part_params[9];
	const double S0_glu = part_params[10];

	J( 0 , 0 ) = -D + y[2]*mu_max_1/(K_mu_glu + y[2]);
	J( 0 , 1 ) = 0;
	J( 0 , 2 ) = -y[0]*y[2]*mu_max_1/std::pow(K_mu_glu + y[2], 2) + y[0]*mu_max_1/(K_mu_glu + y[2]);
	J( 0 , 3 ) = 0;
	J( 0 , 4 ) = 0;
	J( 1 , 0 ) = 0;
	J( 1 , 1 ) = -D + y[2]*mu_max_2/(K_mu_glu + y[2]);
	J( 1 , 2 ) = -y[1]*y[2]*mu_max_2/std::pow(K_mu_glu + y[2], 2) + y[1]*mu_max_2/(K_mu_glu + y[2]);
	J( 1 , 3 ) = 0;
	J( 1 , 4 ) = 0;
	J( 2 , 0 ) = -y[2]*mu_max_1/(g_1*(K_mu_glu + y[2]));
	J( 2 , 1 ) = -y[2]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	J( 2 , 2 ) = -D + y[0]*y[2]*mu_max_1/(g_1*std::pow(K_mu_glu + y[2], 2)) - y[0]*mu_max_1/(g_1*(K_mu_glu + y[2])) + y[1]*y[2]*mu_max_2/(g_2*std::pow(K_mu_glu + y[2], 2)) - y[1]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	J( 2 , 3 ) = 0;
	J( 2 , 4 ) = 0;
	J( 3 , 0 ) = kA_1;
	J( 3 , 1 ) = 0;
	J( 3 , 2 ) = 0;
	J( 3 , 3 ) = -D;
	J( 3 , 4 ) = 0;
	J( 4 , 0 ) = 0;
	J( 4 , 1 ) = 0;
	J( 4 , 2 ) = (1.0/2.0)*y[4]*y[2]*mu_max_1/std::pow(K_mu_glu + y[2], 2) - 1.0/2.0*y[4]*mu_max_1/(K_mu_glu + y[2]);
	J( 4 , 3 ) = -std::pow(y[3], n_A_I_1)*std::pow(K_A_I_1, n_A_I_1)*kI_max_1*n_A_I_1/(y[3]*std::pow(std::pow(y[3], n_A_I_1) + std::pow(K_A_I_1, n_A_I_1), 2));
	J( 4 , 4 ) = -1.0/2.0*y[2]*mu_max_1/(K_mu_glu + y[2]);

	dfdt[0] = 0.0;
	dfdt[1] = 0.0;
	dfdt[2] = 0.0;
	dfdt[3] = 0.0;
	dfdt[4] = 0.0;

}
void Models::model_38(const ublas_vec_t  &y , ublas_vec_t &dydt , double t, std::vector <double> &part_params)
{
	//Unpack parameters
	const double D = part_params[0];
	const double g_1 = part_params[1];
	const double g_2 = part_params[2];
	const double K_mu_glu = part_params[3];
	const double kI_max_1 = part_params[4];
	const double mu_max_1 = part_params[5];
	const double mu_max_2 = part_params[6];
	const double S0_glu = part_params[7];

	//Species order is: N_1 N_2 S_glu I_1 
	dydt[0] = -D*y[0] + y[0]*y[2]*mu_max_1/(K_mu_glu + y[2]);
	dydt[1] = -D*y[1] + y[1]*y[2]*mu_max_2/(K_mu_glu + y[2]);
	dydt[2] = D*(S0_glu - y[2]) - y[0]*y[2]*mu_max_1/(g_1*(K_mu_glu + y[2])) - y[1]*y[2]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	dydt[3] = -1.0/2.0*y[3]*y[2]*mu_max_1/(K_mu_glu + y[2]) + kI_max_1;

}
void Models::jac_38(const ublas_vec_t & y , ublas_mat_t &J , const double &/* t*/ , ublas_vec_t &dfdt, std::vector <double> &part_params)
{
	//Unpack parameters
	const double D = part_params[0];
	const double g_1 = part_params[1];
	const double g_2 = part_params[2];
	const double K_mu_glu = part_params[3];
	const double kI_max_1 = part_params[4];
	const double mu_max_1 = part_params[5];
	const double mu_max_2 = part_params[6];
	const double S0_glu = part_params[7];

	J( 0 , 0 ) = -D + y[2]*mu_max_1/(K_mu_glu + y[2]);
	J( 0 , 1 ) = 0;
	J( 0 , 2 ) = -y[0]*y[2]*mu_max_1/std::pow(K_mu_glu + y[2], 2) + y[0]*mu_max_1/(K_mu_glu + y[2]);
	J( 0 , 3 ) = 0;
	J( 1 , 0 ) = 0;
	J( 1 , 1 ) = -D + y[2]*mu_max_2/(K_mu_glu + y[2]);
	J( 1 , 2 ) = -y[1]*y[2]*mu_max_2/std::pow(K_mu_glu + y[2], 2) + y[1]*mu_max_2/(K_mu_glu + y[2]);
	J( 1 , 3 ) = 0;
	J( 2 , 0 ) = -y[2]*mu_max_1/(g_1*(K_mu_glu + y[2]));
	J( 2 , 1 ) = -y[2]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	J( 2 , 2 ) = -D + y[0]*y[2]*mu_max_1/(g_1*std::pow(K_mu_glu + y[2], 2)) - y[0]*mu_max_1/(g_1*(K_mu_glu + y[2])) + y[1]*y[2]*mu_max_2/(g_2*std::pow(K_mu_glu + y[2], 2)) - y[1]*mu_max_2/(g_2*(K_mu_glu + y[2]));
	J( 2 , 3 ) = 0;
	J( 3 , 0 ) = 0;
	J( 3 , 1 ) = 0;
	J( 3 , 2 ) = (1.0/2.0)*y[3]*y[2]*mu_max_1/std::pow(K_mu_glu + y[2], 2) - 1.0/2.0*y[3]*mu_max_1/(K_mu_glu + y[2]);
	J( 3 , 3 ) = -1.0/2.0*y[2]*mu_max_1/(K_mu_glu + y[2]);

	dfdt[0] = 0.0;
	dfdt[1] = 0.0;
	dfdt[2] = 0.0;
	dfdt[3] = 0.0;

}
