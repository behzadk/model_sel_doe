#!/bin/bash

sudo xcode-select --switch /Library/Developer/CommandLineTools/

inputs_folder=./input_files/input_files_two_species_5/
# inputs_folder=./input_files/input_files_one_species_0/
# inputs_folder=./input_files/input_files_two_species_spock_manu_2/
# inputs_folder=./input_files/input_files_three_species_0/
# inputs_folder=./input_files/input_files_pred_prey_prey_0/
# inputs_folder=./input_files/input_files_three_species_3/
# inputs_folder=./input_files/input_files_three_species_7/
# inputs_folder=./input_files/input_files_two_species_3/

# inputs_folder=./input_files/input_files_one_species_0/
# inputs_folder=./input_files/input_files_two_species_auxos_0/
# inputs_folder=./input_files/input_files_normalise_compare/

kissfft_folder=./kissfft/


export LDFLAGS="-L/usr/local/opt/llvm/lib"
export CPPFLAGS="-I/usr/local/opt/llvm/include"


LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/behzad/Documents/barnes_lab/cplusplus_software/GSL/lib
export LD_LIBRARY_PATH

LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/behzad/Documents/barnes_lab/cplusplus_software/speed_test/repressilator/cpp/kissfft
export LD_LIBRARY_PATH

gcc-8 -std=c++11 -g -w -shared -o population_modules.so -Wall -fPIC -fopenmp  \
-I/usr/local/include  \
particle_sim_opemp.cpp $inputs_folder/model.cpp distances.cpp population.cpp $kissfft_folder/kiss_fft.c $kissfft_folder/kiss_fftr.c \
-lboost_system -lboost_python-py36 -lgsl -lgslcblas -lm \
-lpython3.6m -I/usr/include/python3.6m/ \
-lboost_numpy3 \
-I/home/behzad/Documents/barnes_lab/cplusplus_software/GSL/include \
-L/home/behzad/Documents/barnes_lab/cplusplus_software/GSL/lib \
-L/home/behzad/Documents/barnes_lab/cplusplus_software/kissfft/tools \
-L/home/behzad/Documents/barnes_lab/cplusplus_software/kissfft \
-I $inputs_folder

