#!/bin/bash -l

cpp_dir=$HOME'/software/cpp_consortium_sim/model_sel_doe/'
venv=$cpp_dir'venv/cpp_py/bin/activate'
inputs_folder=./input_files_three_species_0/
inputs_folder=./input_files_two_species_spock_manu_0/
inputs_folder=./input_files_two_species_spock_manu_1/

module unload compilers
module load compilers/gnu/4.9.2
module load eigen
module load python3/recommended

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/lustre/home/ucbtbdk/Scratch/cpp_consortium_sim/model_sel_doe

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/lustre/home/ucbtbdk/software/boost_1_65_1/libs
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/lustre/home/ucbtbdk/software/boost_1_65_1/stage/lib

export INCLUDE=$INCLUDE:/lustre/home/ucbtbdk/software/boost_1_65_1/

export C_INCLUDE_PATH=$C_INCLUDE_PATH:/lustre/home/ucbtbdk/software/boost_1_65_1/stage/lib
export C_INCLUDE_PATH=$C_INCLUDE_PATH:/lustre/home/ucbtbdk/software/boost_1_65_1/libs
export C_INCLUDE_PATH=$C_INCLUDE_PATH:/lustre/home/ucbtbdk/software/boost_1_65_1


LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/behzad/Documents/barnes_lab/cplusplus_software/speed_test/repressilator/cpp/kissfft
export LD_LIBRARY_PATH

export LIBRARY_PATH=$LIBRARY_PATH:/lustre/home/ucbtbdk/software/boost_1_65_1/stage/lib
export LIBRARY_PATH=$LIBRARY_PATH:/lustre/home/ucbtbdk/software/boost_1_65_1/libs

kissfft_folder=./kissfft/


# echo $venv
# source $venv
python3 -V

# 1. Force bash as the executing shell.
#$ -S /bin/bash

# 2. Request ten minutes of wallclock time (format hours:minutes:seconds).
#$ -l h_rt=4:0:0

# 3. Request 1 gigabyte of RAM (must be an integer)
#$ -l mem=1G

# 5. Set the name of the job.
#$ -N test_cpp
g++ -std=c++11 -g -w -shared -o population_modules.so -Wall -fPIC -fopenmp  \
particle_sim_opemp.cpp $inputs_folder/model.cpp distances.cpp population.cpp $kissfft_folder/kiss_fft.c $kissfft_folder/kiss_fftr.c \
-I/lustre/home/ucbtbdk/software/boost_1_65_1 -lboost_system -lboost_python3 -lpython3.6m \
-I/shared/ucl/apps/python/bundles/python3-3.0.0/venv/include/python3.6m/ \
-I $inputs_folder \
-L/home/behzad/Documents/barnes_lab/cplusplus_software/kissfft/tools \
-L/home/behzad/Documents/barnes_lab/cplusplus_software/kissfft

# ./build.sh
# python3 run_boost_rpr.py


echo "finished"
