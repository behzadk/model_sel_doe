#!/bin/bash

 
g++ -std=c++11 -g -shared -o population_modules.so -Wall -fPIC -fopenmp particle_sim_opemp.cpp model.cpp distances.cpp population.cpp -lboost_system -lboost_python-py36 -lpython3.6m -I/usr/include/python3.6m/ 

