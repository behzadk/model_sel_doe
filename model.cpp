#include <array>
#include <iostream>
#include <vector>
#include <math.h>
#include "model.h"

Models::Models() {
	// models_vec = { &Models::model_0, &Models::model_1, &Models::model_2, 
	// 	&Models::model_3, &Models::model_4, &Models::model_5, &Models::model_6};
	
	// models_vec = { &Models::spock_model};
	models_ublas_vec = { &Models::spock_model};

}

void Models::run_model(const std::vector <double> &y , std::vector <double> &dxdt , double t, std::vector <double> &part_params, int &model_ref)
{
	(this->*models_vec[model_ref])(y, dxdt, t, part_params);
}

void Models::run_model_ublas(const ublas_vec_t &y , ublas_vec_t &dxdt , double t, std::vector <double> &part_params, int &model_ref)
{
	(this->*models_ublas_vec[model_ref])(y, dxdt, t, part_params);
}



void Models::rpr_model(const ublas_vec_t  &y , ublas_vec_t &dxdt , double t, std::vector <double> &part_params)
{
	// Unpack parameters
	const double alpha0 = part_params[0];
	const double alpha_param = part_params[1];
	const double coeff = part_params[2];
	const double beta_param = part_params[3];
	//Functions (None)
	
	//Differential equations
	dxdt[0] = (-y[0] + (alpha_param /( 1 + pow(y[5],coeff))) + alpha0);
    dxdt[1] = (-y[1] + (alpha_param/ (1 + pow(y[3],coeff))) + alpha0);
    dxdt[2] = (-y[2] + (alpha_param / (1 + pow(y[4],coeff))) + alpha0);
    dxdt[3] = (-beta_param*(y[3] - y[0]));
    dxdt[4] = (-beta_param*(y[4] - y[1]));
    dxdt[5] = (-beta_param*(y[5] - y[2]));
}


void Models::spock_model(const ublas_vec_t  &y , ublas_vec_t &dxdt , double t, std::vector <double> &part_params)
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

	// std::cout << "y3" << y[3] << std::endl;
	// std::cout << "omega_c"<< omega_c << std::endl;
	// std::cout << "mux"<< mux << std::endl;
	// std::cout << "muc"<< muc << std::endl;
	// std::cout << "CL"<< CL << std::endl;
	// std::cout << "pL"<< pL << std::endl;
	// std::cout << "CFT"<< CFT << std::endl;
	// std::cout << "P_T"<< P_T << std::endl;
	// std::cout << "CB"<< CB << std::endl;
	// std::cout << "CFB"<< CFB << std::endl;
	// std::cout << "P_B"<< P_B << std::endl;
	// std::cout << "k_beta"<< k_beta << std::endl;
	// std::cout << "k_alpha"<< k_alpha << std::endl;



    dxdt[0] = (mux - D)*y[0];
    dxdt[1] = (muc - D - omega_c) * y[1];
	dxdt[2] = D * (S0 - y[2]) - (mux * y[0] / gx) - (muc * y[1] / gc);
	dxdt[3] = (k_beta * y[0]) - D * y[3];
	dxdt[4] = (k_alpha *  y[0]) - D * y[4];

}


void Models::model_0(const std::vector <double> &y , std::vector <double> &dxdt , double t, std::vector <double> &part_params)
{
const double D = part_params[0];
const double kA_a = part_params[1];
const double KB_y = part_params[2];
const double kBmax_y = part_params[3];
const double nB_y = part_params[4];
const double omega_max_y = part_params[5];
const double K_omega_y = part_params[6];
const double n_omega_y = part_params[7];
const double K_x = part_params[8];
const double g_x = part_params[9];
const double mu_max_x = part_params[10];
const double kA_b = part_params[11];
const double KB_z = part_params[12];
const double kBmax_z = part_params[13];
const double nB_z = part_params[14];
const double omega_max_z = part_params[15];
const double K_omega_z = part_params[16];
const double n_omega_z = part_params[17];
const double K_c = part_params[18];
const double g_c = part_params[19];
const double mu_max_c = part_params[20];
const double S0_glu = part_params[21];

// Order is: N_x, N_c, S_glu, A_a, A_b, B_y, B_z, 

dxdt[0] = ( ( mu_max_x * y[2] / ( K_x + y[2] ) )  - omega_max_y * pow( y[5] , n_omega_y ) / ( pow( K_omega_y , n_omega_y ) + pow( y[5] , n_omega_y ) ) - D) * y[0] ;
dxdt[1] = ( ( mu_max_c * y[2] / ( K_c + y[2] ) )  - omega_max_z * pow( y[6] , n_omega_z ) / ( pow( K_omega_z , n_omega_z ) + pow( y[6] , n_omega_z ) ) - D) * y[1] ;
dxdt[2] =  D * (S0_glu - y[2] ) - ( ( mu_max_x * y[2] / ( K_x + y[2] ) )  * y[0] / g_x )- ( ( mu_max_c * y[2] / ( K_c + y[2] ) )  * y[1] / g_c );
dxdt[3] =  + kA_a * y[0]  - D * y[3] ;
dxdt[4] =  + kA_b * y[1]  - D * y[4] ;
dxdt[5] =  + kBmax_y  * y[1]  - D * y[5] ;
dxdt[6] =  + kBmax_z  * ( pow( y[4] , nB_z ) / ( pow( KB_z, nB_z ) + pow( y[4] , nB_z ) ) ) * y[1]  - D * y[6] ;

}

void Models::model_1(const std::vector <double> &y , std::vector <double> &dxdt , double t, std::vector <double> &part_params)
{
const double D = part_params[0];
const double kA_a = part_params[1];
const double KB_y = part_params[2];
const double kBmax_y = part_params[3];
const double nB_y = part_params[4];
const double omega_max_y = part_params[5];
const double K_omega_y = part_params[6];
const double n_omega_y = part_params[7];
const double K_x = part_params[8];
const double g_x = part_params[9];
const double mu_max_x = part_params[10];
const double kA_b = part_params[11];
const double KB_z = part_params[12];
const double kBmax_z = part_params[13];
const double nB_z = part_params[14];
const double omega_max_z = part_params[15];
const double K_omega_z = part_params[16];
const double n_omega_z = part_params[17];
const double K_c = part_params[18];
const double g_c = part_params[19];
const double mu_max_c = part_params[20];
const double S0_glu = part_params[21];

// Order is: N_x, N_c, S_glu, A_a, A_b, B_y, B_z, 

dxdt[0] = ( ( mu_max_x * y[2] / ( K_x + y[2] ) )  - omega_max_y * pow( y[5] , n_omega_y ) / ( pow( K_omega_y , n_omega_y ) + pow( y[5] , n_omega_y ) ) - D) * y[0] ;
dxdt[1] = ( ( mu_max_c * y[2] / ( K_c + y[2] ) )  - omega_max_z * pow( y[6] , n_omega_z ) / ( pow( K_omega_z , n_omega_z ) + pow( y[6] , n_omega_z ) ) - D) * y[1] ;
dxdt[2] =  D * (S0_glu - y[2] ) - ( ( mu_max_x * y[2] / ( K_x + y[2] ) )  * y[0] / g_x )- ( ( mu_max_c * y[2] / ( K_c + y[2] ) )  * y[1] / g_c );
dxdt[3] =  + kA_a * y[0]  - D * y[3] ;
dxdt[4] =  + kA_b * y[1]  - D * y[4] ;
dxdt[5] =  + kBmax_y  * y[1]  - D * y[5] ;
dxdt[6] =  + kBmax_z  * ( pow( y[3] , nB_z ) / ( pow( KB_z, nB_z ) + pow( y[3] , nB_z ) ) ) * y[1]  - D * y[6] ;

}

void Models::model_2(const std::vector <double> &y , std::vector <double> &dxdt , double t, std::vector <double> &part_params)
{
const double D = part_params[0];
const double kA_a = part_params[1];
const double KB_y = part_params[2];
const double kBmax_y = part_params[3];
const double nB_y = part_params[4];
const double omega_max_y = part_params[5];
const double K_omega_y = part_params[6];
const double n_omega_y = part_params[7];
const double K_x = part_params[8];
const double g_x = part_params[9];
const double mu_max_x = part_params[10];
const double K_c = part_params[11];
const double g_c = part_params[12];
const double mu_max_c = part_params[13];
const double S0_glu = part_params[14];

// Order is: N_x, N_c, S_glu, A_a, B_y, 

dxdt[0] = ( ( mu_max_x * y[2] / ( K_x + y[2] ) )  - D) * y[0] ;
dxdt[1] = ( ( mu_max_c * y[2] / ( K_c + y[2] ) )  - omega_max_y * pow( y[4] , n_omega_y ) / ( pow( K_omega_y , n_omega_y ) + pow( y[4] , n_omega_y ) ) - D) * y[1] ;
dxdt[2] =  D * (S0_glu - y[2] ) - ( ( mu_max_x * y[2] / ( K_x + y[2] ) )  * y[0] / g_x )- ( ( mu_max_c * y[2] / ( K_c + y[2] ) )  * y[1] / g_c );
dxdt[3] =  + kA_a * y[0]  - D * y[3] ;
dxdt[4] =  + kBmax_y  * ( pow( KB_y , nB_y ) / ( pow( KB_y , nB_y ) + pow( y[3] , nB_y ) ) ) * y[0]  - D * y[4] ;

}

void Models::model_3(const std::vector <double> &y , std::vector <double> &dxdt , double t, std::vector <double> &part_params)
{
const double D = part_params[0];
const double KB_y = part_params[1];
const double K_c = part_params[2];
const double K_omega_y = part_params[3];
const double K_x = part_params[4];
const double S0_glu = part_params[5];
const double g_c = part_params[6];
const double g_x = part_params[7];
const double kA_a = part_params[8];
const double kBmax_y = part_params[9];
const double mu_max_c = part_params[10];
const double mu_max_x = part_params[11];
const double nB_y = part_params[12];
const double n_omega_y = part_params[13];
const double omega_max_y = part_params[14];



// Order is: N_x, N_c, S_glu, A_a, B_y, 

dxdt[0] = ( ( mu_max_x * y[2] / ( K_x + y[2] ) )  - D) * y[0] ;
dxdt[1] = ( ( mu_max_c * y[2] / ( K_c + y[2] ) )  - omega_max_y * pow( y[4] , n_omega_y ) / ( pow( K_omega_y , n_omega_y ) + pow( y[4] , n_omega_y ) ) - D) * y[1] ;
dxdt[2] =  D * (S0_glu - y[2] ) - ( ( mu_max_x * y[2] / ( K_x + y[2] ) )  * y[0] / g_x )- ( ( mu_max_c * y[2] / ( K_c + y[2] ) )  * y[1] / g_c );
dxdt[3] =  + kA_a * y[1]  - D * y[3] ;
dxdt[4] =  + kBmax_y  * ( pow( y[3] , nB_y ) / ( pow( KB_y, nB_y ) + pow( y[3] , nB_y ) ) ) * y[0]  - D * y[4] ;

}

void Models::model_4(const std::vector <double> &y , std::vector <double> &dxdt , double t, std::vector <double> &part_params)
{
const double D = part_params[0];
const double KB_y = part_params[1];
const double kBmax_y = part_params[2];
const double nB_y = part_params[3];
const double omega_max_y = part_params[4];
const double K_omega_y = part_params[5];
const double n_omega_y = part_params[6];
const double K_x = part_params[7];
const double g_x = part_params[8];
const double mu_max_x = part_params[9];
const double kA_b = part_params[10];
const double K_c = part_params[11];
const double g_c = part_params[12];
const double mu_max_c = part_params[13];
const double S0_glu = part_params[14];

// Order is: N_x, N_c, S_glu, A_b, B_y, 

dxdt[0] = ( ( mu_max_x * y[2] / ( K_x + y[2] ) )  - D) * y[0] ;
dxdt[1] = ( ( mu_max_c * y[2] / ( K_c + y[2] ) )  - omega_max_y * pow( y[4] , n_omega_y ) / ( pow( K_omega_y , n_omega_y ) + pow( y[4] , n_omega_y ) ) - D) * y[1] ;
dxdt[2] =  D * (S0_glu - y[2] ) - ( ( mu_max_x * y[2] / ( K_x + y[2] ) )  * y[0] / g_x )- ( ( mu_max_c * y[2] / ( K_c + y[2] ) )  * y[1] / g_c );
dxdt[3] =  + kA_b * y[1]  - D * y[3] ;
dxdt[4] =  + kBmax_y  * ( pow( KB_y , nB_y ) / ( pow( KB_y , nB_y ) + pow( y[3] , nB_y ) ) ) * y[0]  - D * y[4] ;

}

void Models::model_5(const std::vector <double> &y , std::vector <double> &dxdt , double t, std::vector <double> &part_params)
{
const double D = part_params[0];
const double kA_a = part_params[1];
const double KB_y = part_params[2];
const double kBmax_y = part_params[3];
const double nB_y = part_params[4];
const double omega_max_y = part_params[5];
const double K_omega_y = part_params[6];
const double n_omega_y = part_params[7];
const double KB_z = part_params[8];
const double kBmax_z = part_params[9];
const double nB_z = part_params[10];
const double omega_max_z = part_params[11];
const double K_omega_z = part_params[12];
const double n_omega_z = part_params[13];
const double K_x = part_params[14];
const double g_x = part_params[15];
const double mu_max_x = part_params[16];
const double K_c = part_params[17];
const double g_c = part_params[18];
const double mu_max_c = part_params[19];
const double S0_glu = part_params[20];

// Order is: N_x, N_c, S_glu, A_a, B_y, B_z, 

dxdt[0] = ( ( mu_max_x * y[2] / ( K_x + y[2] ) )  - omega_max_y * pow( y[4] , n_omega_y ) / ( pow( K_omega_y , n_omega_y ) + pow( y[4] , n_omega_y ) ) - D) * y[0] ;
dxdt[1] = ( ( mu_max_c * y[2] / ( K_c + y[2] ) )  - omega_max_z * pow( y[5] , n_omega_z ) / ( pow( K_omega_z , n_omega_z ) + pow( y[5] , n_omega_z ) ) - D) * y[1] ;
dxdt[2] =  D * (S0_glu - y[2] ) - ( ( mu_max_x * y[2] / ( K_x + y[2] ) )  * y[0] / g_x )- ( ( mu_max_c * y[2] / ( K_c + y[2] ) )  * y[1] / g_c );
dxdt[3] =  + kA_a * y[0]  - D * y[3] ;
dxdt[4] =  + kBmax_y  * ( pow( y[3] , nB_y ) / ( pow( KB_y, nB_y ) + pow( y[3] , nB_y ) ) ) * y[0]  - D * y[4] ;
dxdt[5] =  + kBmax_z  * ( pow( KB_z , nB_z ) / ( pow( KB_z , nB_z ) + pow( y[3] , nB_z ) ) ) * y[0]  - D * y[5] ;

}

void Models::model_6(const std::vector <double> &y , std::vector <double> &dxdt , double t, std::vector <double> &part_params)
{
const double D = part_params[0];
const double kA_a = part_params[1];
const double KB_y = part_params[2];
const double kBmax_y = part_params[3];
const double nB_y = part_params[4];
const double omega_max_y = part_params[5];
const double K_omega_y = part_params[6];
const double n_omega_y = part_params[7];
const double KB_z = part_params[8];
const double kBmax_z = part_params[9];
const double nB_z = part_params[10];
const double omega_max_z = part_params[11];
const double K_omega_z = part_params[12];
const double n_omega_z = part_params[13];
const double K_x = part_params[14];
const double g_x = part_params[15];
const double mu_max_x = part_params[16];
const double kA_b = part_params[17];
const double K_c = part_params[18];
const double g_c = part_params[19];
const double mu_max_c = part_params[20];
const double S0_glu = part_params[21];

// Order is: N_x, N_c, S_glu, A_a, A_b, B_y, B_z, 

dxdt[0] = ( ( mu_max_x * y[2] / ( K_x + y[2] ) )  - omega_max_y * pow( y[5] , n_omega_y ) / ( pow( K_omega_y , n_omega_y ) + pow( y[5] , n_omega_y ) ) - D) * y[0] ;
dxdt[1] = ( ( mu_max_c * y[2] / ( K_c + y[2] ) )  - omega_max_z * pow( y[6] , n_omega_z ) / ( pow( K_omega_z , n_omega_z ) + pow( y[6] , n_omega_z ) ) - D) * y[1] ;
dxdt[2] =  D * (S0_glu - y[2] ) - ( ( mu_max_x * y[2] / ( K_x + y[2] ) )  * y[0] / g_x )- ( ( mu_max_c * y[2] / ( K_c + y[2] ) )  * y[1] / g_c );
dxdt[3] =  + kA_a * y[0]  - D * y[3] ;
dxdt[4] =  + kA_b * y[1]  - D * y[4] ;
dxdt[5] =  + kBmax_y  * ( pow( y[3] , nB_y ) / ( pow( KB_y, nB_y ) + pow( y[3] , nB_y ) ) ) * y[0]  - D * y[5] ;
dxdt[6] =  + kBmax_z  * ( pow( y[4] , nB_z ) / ( pow( KB_z, nB_z ) + pow( y[4] , nB_z ) ) ) * ( pow( KB_z , nB_z ) / ( pow( KB_z , nB_z ) + pow( y[3] , nB_z ) ) ) * y[0]  - D * y[6] ;

}

