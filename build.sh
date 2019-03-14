#!/bin/bash

inputs_folder=./input_files_two_species/

LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/behzad/Documents/barnes_lab/cplusplus_software/GSL/lib
export LD_LIBRARY_PATH

g++ -std=c++11 -g -shared -o population_modules.so -Wall -fPIC -fopenmp \
particle_sim_opemp.cpp $inputs_folder/model.cpp distances.cpp population.cpp \
-lboost_system -lboost_python-py36 -lgsl -lgslcblas -lm \
-lpython3.6m -I/usr/include/python3.6m/ \
-lboost_numpy3 \
-I/home/behzad/Documents/barnes_lab/cplusplus_software/GSL/include \
-L/home/behzad/Documents/barnes_lab/cplusplus_software/GSL/lib \
-I $inputs_folder


