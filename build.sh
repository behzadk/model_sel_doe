#!/bin/bash

inputs_folder=./input_files/input_files_two_species_0/
# inputs_folder=./input_files/input_files_one_species_0/
# inputs_folder=./input_files/input_files_two_species_spock_manu_1/
inputs_folder=./input_files/input_files_three_species_0/
# inputs_folder=./input_files/input_files_one_species_0/

kissfft_folder=./kissfft/

LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/behzad/Documents/barnes_lab/cplusplus_software/GSL/lib
export LD_LIBRARY_PATH

LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/behzad/Documents/barnes_lab/cplusplus_software/speed_test/repressilator/cpp/kissfft
export LD_LIBRARY_PATH

g++ -std=c++11 -g -w -shared -o population_modules.so -Wall -fPIC -fopenmp  \
particle_sim_opemp.cpp $inputs_folder/model.cpp distances.cpp population.cpp $kissfft_folder/kiss_fft.c $kissfft_folder/kiss_fftr.c \
-lboost_system -lboost_python-py36 -lgsl -lgslcblas -lm \
-lpython3.6m -I/usr/include/python3.6m/ \
-lboost_numpy3 \
-I/home/behzad/Documents/barnes_lab/cplusplus_software/GSL/include \
-L/home/behzad/Documents/barnes_lab/cplusplus_software/GSL/lib \
-L/home/behzad/Documents/barnes_lab/cplusplus_software/kissfft/tools \
-L/home/behzad/Documents/barnes_lab/cplusplus_software/kissfft \
-I $inputs_folder

