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

		void run_model(const std::vector <double> &, std::vector <double> &, double, std::vector <double>&, int&);
		void spock_model(const std::vector <double> &, std::vector <double> &, double, std::vector <double>&);

		void model_0(const std::vector <double> &, std::vector <double> &, double, std::vector <double>&);
		void model_1(const std::vector <double> &, std::vector <double> &, double, std::vector <double>&);
		void model_2(const std::vector <double> &, std::vector <double> &, double, std::vector <double>&);
		void model_3(const std::vector <double> &, std::vector <double> &, double, std::vector <double>&);
		void model_4(const std::vector <double> &, std::vector <double> &, double, std::vector <double>&);
		void model_5(const std::vector <double> &, std::vector <double> &, double, std::vector <double>&);
		void model_6(const std::vector <double> &, std::vector <double> &, double, std::vector <double>&);

		std::vector<model_t> models_vec;

};

#endif
