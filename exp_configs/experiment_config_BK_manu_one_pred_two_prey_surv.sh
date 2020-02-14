experiment_name: BK_manu_one_pred_two_prey
inputs_folder: './input_files/input_files_pred_prey_prey_0/input_files/'
output_folder: './output/'

# Start time, end time, time interval
t_0: 0
t_end: 1500
dt: 0.5

final_epsilon: 
  - 1000

fit_species:
  - 0
  - 1
  - 2


population_size: 1000000
n_sims_batch: 50

## Distance options
# 0: stable steady state
# 1: oscillations
# 2: survival
distance_function_mode: 2

# Choose which algorithm to run
run_rejection: Y
run_SMC: N