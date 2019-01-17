#!/bin/bash -l
#!/usr/bin/env python

cpp_dir=$HOME'/software/cpp_consortium_sim/model_sel_doe/'
venv=$cpp_dir'venv/cpp_py/bin/activate'

module unload compilers
module load compilers/gnu/4.9.2
module load python3/recommended
module load boost/1_63_0/gnu-4.9.2

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ucbtbdk/Scratch/cpp_consortium_sim/model_sel_doe
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ucbtbdk/software/boost_1_63_0/libs

echo $LD_LIBRARY_PATH

# echo $venv
# source $venv
python3 -V

# 1. Force bash as the executing shell.
#$ -S /bin/bash

# 2. Request ten minutes of wallclock time (format hours:minutes:seconds).
#$ -l h_rt=0:10:0

# 3. Request 1 gigabyte of RAM (must be an integer)
#$ -l mem=1G

# 5. Set the name of the job.
#$ -N test_cpp

g++ -std=c++11 -g -shared -o population_modules.so -Wall -fPIC -fopenmp  particle_sim_opemp.cpp model.cpp distances.cpp population.cpp -lboost_system -lboost_python -lpython3.6m -I/shared/ucl/apps/python/bundles/python3-3.0.0/venv/include/python3.6m/

# ./build.sh
# python3 run_boost_rpr.py


echo "finished"
