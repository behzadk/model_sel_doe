#$ -l tmem=1G,h_vmem=1G,h_rt=96:0:0,tscratch=4G
#$ -S /bin/bash
#$ -N bk_three_spec
#$ -R y
#$ -t 300-325
#$ -pe smp 30

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib64
export PATH=/share/apps/python-3.7.2-shared/bin:${PATH}
export PYTHONPATH=$PYTHONPATH:/share/apps/python-3.7.2-shared/bin

export LD_LIBRARY_PATH=/share/apps/python-3.7.2-shared/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=/share/apps/boost-1.72-python3/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=/share/apps/gcc-9.2.0/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=/share/apps/gcc-9.2.0/lib64:${LD_LIBRARY_PATH}

export INCLUDE=$INCLUDE:/share/apps/python-3.7.2-shared/
export INCLUDE=$INCLUDE:/share/apps/boost-1.72-python3/include

export C_INCLUDE_PATH=/share/apps/python-3.7.2-shared/include:${C_INCLUDE_PATH}
export C_INCLUDE_PATH=$C_INCLUDE_PATH:/share/apps/boost-1.72-python3
export C_INCLUDE_PATH=/share/apps/python-3.7.2-shared:${C_INCLUDE_PATH}


function finish {
    rm -rf /scratch0/karkaria/chaos_three_species_$SGE_TASK_ID
}

kissfft_folder=./kissfft/

source ~/venv/bin/activate

export OMP_NUM_THREADS=30
echo "$OMP_NUM_THREADS"

echo "Making output dir in tmpdir"
mkdir -p /scratch0/karkaria/chaos_three_species_$SGE_TASK_ID/output

cd $HOME/cpp_consortium_sim

echo "copying inputfiles to tmpdir"
cp -R input_files /scratch0/karkaria/chaos_three_species_$SGE_TASK_ID

cp ./*.py /scratch0/karkaria/chaos_three_species_$SGE_TASK_ID
cp ./*.cpp /scratch0/karkaria/chaos_three_species_$SGE_TASK_ID
cp ./*.h /scratch0/karkaria/chaos_three_species_$SGE_TASK_ID
cp ./*.so /scratch0/karkaria/chaos_three_species_$SGE_TASK_ID
cp -R exp_configs /scratch0/karkaria/chaos_three_species_$SGE_TASK_ID

cd /scratch0/karkaria/chaos_three_species_$SGE_TASK_ID
pwd

echo "Running experiment"
python3 run_boost_rpr.py --config ./exp_configs/chaos_three_species.yaml --exp_suffix $SGE_TASK_ID  >> $HOME/cpp_consortium_sim/output/log.chaos_three_species_$SGE_TASK_ID

# spock_two_spec_SMC_stable_1_$SGE_TASK_ID 8 >> $HOME/Scratch/cpp_consortium_sim/model_sel_doe/output/log.spock_two_spec_SMC_stable_1_$SGE_TASK_ID
cp /scratch0/karkaria/chaos_three_species_$SGE_TASK_ID/output/*.tar.gz $HOME/cpp_consortium_sim/output/
rm -r /scratch0/karkaria/chaos_three_species_$SGE_TASK_ID

trap finish EXIT ERR

echo "finished"
