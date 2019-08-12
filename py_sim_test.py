import scipy as sp
import numpy as np
import pandas as pd
import csv
from scipy.integrate import odeint
import run_boost_rpr
import os
from model_space import ModelSpace
from model_space import Model

import algorithm_utils as alg_utils
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import plotting

# np.seterr(all='raise')

def model_93_scaled(y, t, part_params):
    D = part_params[0]
    KB_mccB = part_params[1]
    KB_mccV = part_params[2]
    K_c = part_params[3]
    K_omega_mccB = part_params[4]
    K_omega_mccV = part_params[5]
    K_x = part_params[6]
    S0_glu = part_params[7]
    g_c = part_params[8]
    g_x = part_params[9]
    kA_1 = part_params[10]
    kA_2 = part_params[11]
    kBmax_mccB = part_params[12]
    kBmax_mccV = part_params[13]
    mu_max_c = part_params[14]
    mu_max_x = part_params[15]
    nB_mccB = part_params[16]
    nB_mccV = part_params[17]
    n_omega_mccB = part_params[18]
    n_omega_mccV = part_params[19]
    omega_max_mccB = part_params[20]
    omega_max_mccV = part_params[21]

    C = 1e10

    N_x = y[0]
    N_c = y[1]
    S_glu = y[2]
    B_mccV = y[3]
    B_mccB = y[4]
    A_2 = y[5]
    A_1 = y[6]

    y0_1 = D
    y0_2 = ( mu_max_x * S_glu / ( K_x + S_glu ) )
    y0_3 = ( omega_max_mccV * pow(B_mccV,n_omega_mccV) / ( pow(K_omega_mccV,n_omega_mccV) + pow(B_mccV,n_omega_mccV) ) )
    y0_4 = ( omega_max_mccB * pow(B_mccB,n_omega_mccB) / ( pow(K_omega_mccB, n_omega_mccB) + pow(B_mccB, n_omega_mccB) ) )

    y0_scaled = (-y0_1 + y0_2 - y0_3 - y0_4) * N_x

    y1_1 = D
    y1_2 = ( mu_max_c * S_glu / ( K_c + S_glu ) )
    y1_3 = ( omega_max_mccV * pow(B_mccV,n_omega_mccV) / ( pow(K_omega_mccV, n_omega_mccV) + pow(B_mccV,n_omega_mccV) ) )

    y1_scaled = (-y1_1 + y1_2 - y1_3) * N_c

    y2_scaled = ( D * ( S0_glu - S_glu ) ) - ( mu_max_x * S_glu / ( K_x + S_glu ) ) * N_x*C / g_x - ( mu_max_c * S_glu / ( K_c + S_glu ) ) * N_c * C / g_c 

    y3_scaled = ( - D * B_mccV ) +  kBmax_mccV  * ( pow(A_1, nB_mccV) / ( pow(KB_mccV,nB_mccV) + pow(A_1, nB_mccV) ) ) * N_x * C

    y4_scaled = ( - D * B_mccB ) +  kBmax_mccB  * ( pow(A_2, nB_mccB) / ( pow(KB_mccB,nB_mccB) + pow(A_2, nB_mccB) ) ) * N_c * C

    y5_scaled = ( - D * A_2 ) + kA_2 * N_x * C

    y6_scaled = ( - D * A_1 ) + kA_1 * N_c * C


    # y0 = ( - D * N_x ) + N_x  * ( mu_max_x * S_glu / ( K_x + S_glu ) ) - ( omega_max_mccV * pow(B_mccV,n_omega_mccV) / ( pow(K_omega_mccV,n_omega_mccV) + pow(B_mccV,n_omega_mccV) ) ) * N_x - ( omega_max_mccB * pow(B_mccB,n_omega_mccB) / ( pow(K_omega_mccB, n_omega_mccB) + pow(B_mccB, n_omega_mccB) ) ) * N_x

    # y1 = ( - D * N_c ) + N_c  * ( mu_max_c * S_glu / ( K_c + S_glu ) ) - ( omega_max_mccV * pow(B_mccV,n_omega_mccV) / ( pow(K_omega_mccV, n_omega_mccV) + pow(B_mccV,n_omega_mccV) ) ) * N_c

    # y2 = ( D * ( S0_glu - S_glu ) ) - ( mu_max_x * S_glu / ( K_x + S_glu ) ) * N_x / g_x - ( mu_max_c * S_glu / ( K_c + S_glu ) ) * N_c / g_c

    # y3 = ( - D * B_mccV ) +  kBmax_mccV  * ( pow(A_1, nB_mccV) / ( pow(KB_mccV,nB_mccV) + pow(A_1, nB_mccV) ) ) * N_x 

    # y4 = ( - D * B_mccB ) +  kBmax_mccB  * ( pow(A_2, nB_mccB) / ( pow(KB_mccB,nB_mccB) + pow(A_2, nB_mccB) ) ) * N_c 

    # y5 = ( - D * A_2 ) + kA_2 * N_x

    # y6 = ( - D * A_1 ) + kA_1 * N_c
 
    return [y0_scaled, y1_scaled, y2_scaled, y3_scaled, y4_scaled, y5_scaled, y6_scaled]

#    return [y0, y1, y2, y3, y4, y5, y6]


def model_93(y, t, part_params):
    D = part_params[0]
    KB_mccB = part_params[1]
    KB_mccV = part_params[2]
    K_c = part_params[3]
    K_omega_mccB = part_params[4]
    K_omega_mccV = part_params[5]
    K_x = part_params[6]
    S0_glu = part_params[7]
    g_c = part_params[8]
    g_x = part_params[9]
    kA_1 = part_params[10]
    kA_2 = part_params[11]
    kBmax_mccB = part_params[12]
    kBmax_mccV = part_params[13]
    mu_max_c = part_params[14]
    mu_max_x = part_params[15]
    nB_mccB = part_params[16]
    nB_mccV = part_params[17]
    n_omega_mccB = part_params[18]
    n_omega_mccV = part_params[19]
    omega_max_mccB = part_params[20]
    omega_max_mccV = part_params[21]

    N_x = y[0]
    N_c = y[1]
    S_glu = y[2]
    B_mccV = y[3]
    B_mccB = y[4]
    A_2 = y[5]
    A_1 = y[6]

    y0 = ( - D * N_x ) + N_x  * ( mu_max_x * S_glu / ( K_x + S_glu ) ) - ( omega_max_mccV * pow(B_mccV,n_omega_mccV) / ( pow(K_omega_mccV,n_omega_mccV) + pow(B_mccV,n_omega_mccV) ) ) * N_x - ( omega_max_mccB * pow(B_mccB,n_omega_mccB) / ( pow(K_omega_mccB, n_omega_mccB) + pow(B_mccB, n_omega_mccB) ) ) * N_x

    y1 = ( - D * N_c ) + N_c  * ( mu_max_c * S_glu / ( K_c + S_glu ) ) - ( omega_max_mccV * pow(B_mccV,n_omega_mccV) / ( pow(K_omega_mccV, n_omega_mccV) + pow(B_mccV,n_omega_mccV) ) ) * N_c

    y2 = ( D * ( S0_glu - S_glu ) ) - ( mu_max_x * S_glu / ( K_x + S_glu ) ) * N_x / g_x - ( mu_max_c * S_glu / ( K_c + S_glu ) ) * N_c / g_c

    y3 = ( - D * B_mccV ) +  kBmax_mccV  * ( pow(A_1, nB_mccV) / ( pow(KB_mccV,nB_mccV) + pow(A_1, nB_mccV) ) ) * N_x 

    y4 = ( - D * B_mccB ) +  kBmax_mccB  * ( pow(A_2, nB_mccB) / ( pow(KB_mccB,nB_mccB) + pow(A_2, nB_mccB) ) ) * N_c 

    y5 = ( - D * A_2 ) + kA_2 * N_x

    y6 = ( - D * A_1 ) + kA_1 * N_c


    return [y0, y1, y2, y3, y4, y5, y6]

def funcM5(y, t, part_params):
    D = part_params[0]
    KB_mccB = part_params[1]
    KB_mccV = part_params[2]
    K_c = part_params[3]
    K_omega_mccB = part_params[4]
    K_omega_mccV = part_params[5]
    K_x = part_params[6]
    S0_glu = part_params[7]
    g_c = part_params[8]
    g_x = part_params[9]
    kA_1 = part_params[10]
    kA_2 = part_params[11]
    kBmax_mccB = part_params[12]
    kBmax_mccV = part_params[13]
    mu_max_c = part_params[14]
    mu_max_x = part_params[15]
    nB_mccB = part_params[16]
    nB_mccV = part_params[17]
    n_omega_mccB = part_params[18]
    n_omega_mccV = part_params[19]
    omega_max_mccB = part_params[20]
    omega_max_mccV = part_params[21]

    N_x           = y[0]
    N_c           = y[1]
    S_glu         = y[2]
    B_mccB        = y[3]
    B_mccV        = y[4]
    A_1           = y[5]
    A_2           = y[6]

     
    dN_x = ( - D * N_x ) + N_x  * ( mu_max_x * S_glu / ( K_x + S_glu ) ) - ( omega_max_mccV * pow(B_mccV,n_omega_mccV) / ( pow(K_omega_mccV, n_omega_mccV) + pow(B_mccV,n_omega_mccV) ) ) * N_x - ( omega_max_mccB * pow(B_mccB,n_omega_mccB) / ( pow(K_omega_mccB, n_omega_mccB) + pow(B_mccB,n_omega_mccB) ) ) * N_x

    dN_c = ( - D * N_c ) + N_c  * ( mu_max_c * S_glu / ( K_c + S_glu ) ) - ( omega_max_mccV * pow(B_mccV,n_omega_mccV) / ( pow(K_omega_mccV,n_omega_mccV) + pow(B_mccV,n_omega_mccV) ) ) * N_c
    
    dS_glu = ( D * ( S0_glu - S_glu ) ) - ( mu_max_x * S_glu / ( K_x + S_glu ) ) * N_x / g_x - ( mu_max_c * S_glu / ( K_c + S_glu ) ) * N_c / g_c

    dB_mccV = ( - D * B_mccV ) +  kBmax_mccV  * ( pow(A_1, nB_mccV) / ( pow(KB_mccV, nB_mccV) + pow(A_1, nB_mccV) ) ) * N_x 

    dB_mccB = ( - D * B_mccB ) +  kBmax_mccB  * ( pow(A_2, nB_mccB) / ( pow(KB_mccB, nB_mccB) + pow(A_2, nB_mccB) ) ) * N_c 

    dA_2 = ( - D * A_2 ) + kA_2 * N_x

    dA_1 = ( - D * A_1 ) + kA_1 * N_x + kA_1 * N_c

  
    
    return [ dN_x, dN_c, dS_glu, dB_mccB, dB_mccV, dA_1, dA_2 ]


def run_test_model():
    pass

def run_model_93():
    # Set time points
    t_0 = 0
    t_end = 1000
    dt = 1
    
    input_folder = './input_files_two_species/priors/'
    output_folder = './output/test_sims/'

    plot_species = [0, 1]

    try:
        os.mkdir(output_folder)
        
    except FileExistsError:
        pass

    # Load models from input files
    model_list = []
    for i in range(int((len(os.listdir(input_folder))/2))):
        input_params = input_folder + "params_" + str(i) + ".csv"
        input_init_species = input_folder + "species_" + str(i) + ".csv"

        if i != 93:
            continue

        init_params = run_boost_rpr.import_input_file(input_params)
        init_species = run_boost_rpr.import_input_file(input_init_species)

        # init_species['S_glu'] = [0, 0]
        init_species['N_x'] = [init_species['N_x'][0]/1e10, init_species['N_x'][1]/1e10]
        init_species['N_c'] = [init_species['N_c'][0]/1e10, init_species['N_c'][1]/1e10]
        # init_params['D'] = [0.0, 0.0]
        # init_species['B_mccV'] = [0, 0]
        # init_species['B_mccB'] = [0, 0]
        # init_species['A_1'] = [0, 0]
        # init_species['A_2'] = [0, 0]
        # init_params['kA_1'] = [0.0, 0.0]
        # init_params['kA_2'] = [0.0, 0.0]
        # init_params['kBmax_mccV'] = [0.0, 0.0]
        # init_params['kBmax_mccB'] = [0.0, 0.0]
        # init_params['omega_max_mccV'] = [0.0, 0.0]
        # init_params['omega_max_mccB'] = [0.0, 0.0]

        model_new = Model(i, init_params, init_species)
        model_list.append(model_new)

    model_space = ModelSpace(model_list)
    batch_size = 25

    particle_models = model_space.sample_model_space(batch_size)
    init_states, input_params, model_refs = alg_utils.generate_particles(particle_models)

    print(np.shape(init_states[0]))
    print(np.shape(input_params[0]))

    t = np.arange(t_0, t_end, dt)
    
    output_folder = './output/test_sims/'
    out_path = output_folder + "batch_" + "batch_" + str(0) + "_plots.pdf"
    # Make new pdf
    pdf = PdfPages(out_path)

    negative_count = 0
    for idx in range(batch_size):
        sol_scaled = odeint(model_93_scaled, init_states[idx], t, args=(input_params[idx],))

        init_states[idx][0] = init_states[idx][0] * 1e10
        init_states[idx][1] = init_states[idx][1] * 1e10
        sol = odeint(model_93, init_states[idx], t, args=(input_params[idx],))

        print(sol_scaled[:, 0])
        # if np.min(sol) < 0 or np.isnan(sol).any():
        #     negative_count += 1
        #     print(np.min(sol))

        # plt.plot(t, sol[:, 0])
        # plt.plot(t, sol[:, 1])
        # plt.plot(t, sol_scaled[:, 0])
        # plt.plot(t, sol_scaled[:, 1])

        # Make new pdf
        idx_scaled = str(idx) + "_scaled"
        plotting.plot_simulation(pdf, idx, 93, sol, t, [0, 1])
        plotting.plot_simulation(pdf, idx_scaled, 93, sol_scaled, t, [0, 1])

    print("negative simulations: ", negative_count)
    pdf.close()

if __name__ == "__main__":
    run_model_93()
