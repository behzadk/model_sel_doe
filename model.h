// Header guard
#ifndef __MODELS_H_INCLUDED__
#define __MODELS_H_INCLUDED__

using namespace std;
#include <array>
#include <iostream>
#include <functional> // for std::function
#include <vector>

class Models{
	typedef void (Models::*model_t)(const std::vector<double> &, std::vector<double> &, double, std::vector<double>&);

	public:
		int x = 1;
		Models();

		template <std::size_t T>
		void set_params(std::array<double, T> params) {
				std::cout << params[0] << std::endl;
		    }

		void run_model(const std::vector <double> &, std::vector <double> &, double, std::vector <double>&);
		void model_lv(const std::vector <double> &, std::vector <double> &, double, std::vector <double>&);
		void rpr_model(const std::vector <double> &, std::vector <double> &, double, std::vector <double>&);
		void hello_world();

		std::vector<model_t> models_vec;

};

#endif
