// Header guard
#ifndef __MODELS_H_INCLUDED__
#define __MODELS_H_INCLUDED__

using namespace std;
#include <array>
#include <iostream>
#include <vector>
#include <boost/python.hpp>
#include <boost/python/args.hpp>
#include <boost/thread/thread.hpp>
#include <boost/phoenix/core.hpp>

#include <boost/phoenix/core.hpp>
#include <boost/phoenix/operator.hpp>
#include <boost/numeric/odeint.hpp>


class Models{
	typedef boost::numeric::ublas::vector< double > ublas_vec_t;
	typedef boost::numeric::ublas::matrix< double > ublas_mat_t;

	typedef void (Models::*model_t)(const std::vector<double> &, std::vector<double> &, double, std::vector<double>&);
	typedef void (Models::*model_ublas_t)(const ublas_vec_t  &, ublas_vec_t &, double, std::vector<double>&);


	public:
		int x = 1;
		Models();

		template <std::size_t T>
		void set_params(std::array<double, T> params) {
				std::cout << params[0] << std::endl;
		    }

		void run_model(const std::vector <double> &, std::vector <double> &, double, std::vector <double>&, int&);
		void run_model_ublas(const ublas_vec_t  &, ublas_vec_t &, double, std::vector <double>&, int&);

		void spock_model(const ublas_vec_t  & , ublas_vec_t & , double , std::vector <double> &);
		void rpr_model(const ublas_vec_t  & , ublas_vec_t & , double , std::vector <double> &);

		void model_0(const std::vector <double> &, std::vector <double> &, double, std::vector <double>&);
		void model_1(const std::vector <double> &, std::vector <double> &, double, std::vector <double>&);
		void model_2(const std::vector <double> &, std::vector <double> &, double, std::vector <double>&);
		void model_3(const std::vector <double> &, std::vector <double> &, double, std::vector <double>&);
		void model_4(const std::vector <double> &, std::vector <double> &, double, std::vector <double>&);
		void model_5(const std::vector <double> &, std::vector <double> &, double, std::vector <double>&);
		void model_6(const std::vector <double> &, std::vector <double> &, double, std::vector <double>&);

		std::vector<model_t> models_vec;
		std::vector<model_ublas_t> models_ublas_vec;

};

#endif
