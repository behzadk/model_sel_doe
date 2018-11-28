#include "model.h"
#include <array>
#include <iostream>
#include <vector>
#include <math.h>

using namespace std;
// typedef void (Models::*model_t)(const std::vector<double> &, std::vector<double> &, double t, std::vector<double>&);

Models::Models() {
	// models vec is initiated at construction. Contains all models
	models_vec = {&Models::rpr_model, &Models::model_lv};
}

void Models::run_model(const std::vector <double> &y , std::vector <double> &dxdt , double t, std::vector <double> &part_params)
{
	(this->*models_vec[0])(y, dxdt, t, part_params);
}

void Models::model_lv(const std::vector <double> &y , std::vector <double> &dxdt , double t, std::vector <double> &part_params)
{
	// Unpack parameters
	const int A = part_params[0];
	const int B = part_params[1];
	const int C = part_params[2];
	const int D = part_params[3];

	//Functions (None)
	
	//Differential equations
	dxdt[0] = A * y[0] - B * y[0] * y[1];
	dxdt[1] = -C * y[1] + D * y[0] * y[1];
}

void Models::rpr_model(const std::vector <double> &y , std::vector <double> &dxdt , double t, std::vector <double> &part_params)
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


void Models::hello_world(){
	std::cout << "hello_world" << std::endl;
}



