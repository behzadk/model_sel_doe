#!/bin/bash

# export CPLUS_INCLUDE_PATH="$CPLUS_INCLUDE_PATH:/usr/include/python3.6m/"
# export CPLUS_INCLUDE_PATH="$CPLUS_INCLUDE_PATH:/home/behzad/Documents/barnes_lab/cplusplus_software/boost_1_61_0/boost/"
export PGI=/opt/pgi;
export PATH=/opt/pgi/linux86-64/18.4/bin:$PATH;
export MANPATH=$MANPATH:/opt/pgi/linux86-64/18.4/man;
export LM_LICENSE_FILE=$LM_LICENSE_FILE:/opt/pgi/license.dat;

boost_path="/home/behzad/Documents/barnes_lab/cplusplus_software/boost_1_61_0/"
pyconfig_path="/usr/include/python3.6m/"

py3lib="/usr/lib/x86_64-linux-gnu/libboost_python-py36.so"
py3libboost_system="/usr/lib/x86_64-linux-gnu/libboost_system.so.1.65.1"

out_file_name="test_out"

# Build including folder
# g++ -std=c++11 lv_boost.cpp -I ${boost_path} -I ${pyconfig_path} -L ${py3lib}  -L ${py3libboost_system}  -o lv_exe
# g++ -std=c++11 test.cpp -I ${boost_path} -I ${pyconfig_path} -L ${py3lib}  -L ${py3libboost_system} -o ${out_file_name} -c
# g++ -std=c++11 test_boost.cpp -I ${boost_path} -I ${pyconfig_path} -L ${py3lib}  -L ${py3libboost_system} -o ${out_file_name}.so



# g++ -shared -o lv_c_mod.so -fPIC lv_boost.cpp -lboost_python-py36 -lpython3.6m -I/usr/include/python3.6m/
# g++ -shared -o lv_openmp_mod.so -Wall -fPIC -fopenmp lv_boost_openmp.cpp -lboost_system -lboost_python-py36 -lpython3.6m -I/usr/include/python3.6m/
# g++  -o particle -Wall -fPIC -fopenmp particle_sim_opemp.cpp -lboost_system -lboost_python-py36 -lpython3.6m -I/usr/include/python3.6m/
# g++  -std=c++11 -o particle -Wall -fPIC -fopenmp main.cpp model.cpp -lboost_system -lboost_python-py36 -lpython3.6m -I/usr/include/python3.6m/ 


# g++  -std=c++1z -shared -o particle_sim.so -Wall -fPIC -fopenmp particle_sim_opemp.cpp model.cpp -lboost_system -lboost_python-py36 -lpython3.6m -I/usr/include/python3.6m/ 
# g++  -std=c++11 -shared -o particle_sim.so -Wall -fPIC  -fopenmp particle_sim_opemp.cpp model.cpp  -lboost_system -lboost_python-py36 -lpython3.6m -I/usr/include/python3.6m/ 
 
g++ -std=c++11 -g -shared -o population_modules.so -Wall -fPIC -fopenmp -fopt-info-note-omp particle_sim_opemp.cpp model.cpp distances.cpp population.cpp -lboost_system -lboost_python3 -lpython3.6m -I/usr/include/python3.6m/ 




# pgc++ -shared -o particle_sim.so particle_sim_opemp.cpp model.cpp -acc -Minfo=accel -lboost_system -lboost_python-py36 -lpython3.6m -I/usr/include/python3.6m/ 

# sudo /opt/pgi/linux86-64/2018/bin/makelocalrc -o /opt/pgi/linux86-64/18.4
# sudo /opt/pgi/linux86-64/2018/bin/makelocalrc -o /opt/pgi/linux86-64/18.4/

# Run test app
# ./${out_file_name}