import numpy as np
import scipy as sp
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import sympy as sp
import seaborn as sns
import math
import population_modules
import time

def one_pred_two_prey(y, t, params):

    # Parameters
    a_1 = params['a_1']                 # Rate of prey population increase
    a_2 = params['a_2']                 # Rate of prey population increase

    b = params['b']                 # Mortality rate
    beta_1 = params['beta_1']       # Reproduction rate per 1 H_1 eaten
    beta_2 = params['beta_2']       # Reproduction rate per 1 H_2 eaten

    alpha_1 = params['alpha_1']    # Rate of death by predation
    alpha_2 = params['alpha_2']    # Rate of death by predation

    alpha_1 = alpha_2
    beta_1 = beta_2

    # Unpack species
    H_1 = y[0]
    H_2 = y[1]
    P_1 = y[2]

    # H - prey
    dH_1 = H_1 * (a_1 - (alpha_1 * P_1))
    dH_2 = H_2 * (a_2 - (alpha_2 * P_1))

    # P - predator
    dP_1 = P_1 * ( (beta_1 * H_1) + (beta_2 * H_2)  - b)

    return [dH_1, dH_2, dP_1]


def lv_sympy():
    dH_1 = 'H_1 * (a_1 - (alpha_1 * P_1))'
    dH_1 = 'H_2 * (a_1 - (alpha_1 * P_1))'
    dP_1 = 'P_1 * ( (beta_1 * H_1) + (beta_2 * H_2)  - b)'

    species_names = ['H_1', 'H_2', 'P_1']
    order = sp.symbols(species_names)
    zeros_list = [0 for i in range(len(order))]
    symbolic_equations = sp.Matrix(zeros_list)
    
    idx = 0
    for eq in [dx, dy, dz]:
        symbolic_equations[idx] = sp.sympify(eq, locals=locals())
        print(symbolic_equations[idx])
        idx += 1


    J = symbolic_equations.jacobian(order)
    for idx_i in range(3):
        for idx_j in range(3):
            print(idx_i, idx_j)
            print(J[idx_i, idx_j])
            print("")

    # trace = -sigma + (sigma*(phi - z) - 1) + (-B - sigma**2*(-x + y)**2)

def lorenz_sympy():
    dx = 'R * (y - x)'
    dy = 'x * (R - z) - y'
    dz = 'x * y - B * z'

    species_names = ['x', 'y', 'z']

    order = sp.symbols(species_names)
    zeros_list = [0 for i in range(len(order))]
    symbolic_equations = sp.Matrix(zeros_list)
    
    idx = 0
    for eq in [dx, dy, dz]:
        symbolic_equations[idx] = sp.sympify(eq, locals=locals())
        print(symbolic_equations[idx])
        idx += 1


    J = symbolic_equations.jacobian(order)
    for idx_i in range(3):
        for idx_j in range(3):
            print(idx_i, idx_j)
            print(J[idx_i, idx_j])
            print("")

    # trace = -sigma + (sigma*(phi - z) - 1) + (-B - sigma**2*(-x + y)**2)

    trace = (J[0, 0] + J[1, 1] + J[2, 2])
    trace = trace.simplify()
    print(trace)


def lorenz_map_Jac_trace(params):
    R = params['R']
    B = params['B']
    return -B - R - 1

def lorenz_map(y_t, t, params):
    R = params['R']
    phi = params['phi']
    B = params['B']

    x = y_t[0]
    y = y_t[1]
    z = y_t[2]

    dx = R * (y - x)
    dy = x * (phi - z) - y
    dz = x * y - B * z

    return [dx, dy, dz]

def sample_modified_LV_input_params():
    params = {'G_1': [0.1, 10.0], 'H_1': [0.1, 10.0], 'H_2': [0.1, 10.0], 'P_1': [0.1, 10.0], 
    'a_1': [1.0], 'a_2': [1.0], 'a_g': [1.0], 'b': [1.5], 
    'beta_1': [0.75], 'beta_2': [0.75], 
    'alpha_1': [1.0], 'alpha_2': [1.0]}
    
    sampled_params = {}

    for key in params.keys():
        noisy_param = -1 
        while noisy_param < 0:
            noisy_param = params[key][0] + np.random.normal()

        sampled_params[key] = noisy_param

    for key in ['G_1', 'H_1', 'H_2', 'P_1']:
        param_high = params[key][1]
        param_low = params[key][0]
        sampled_params[key] = np.random.uniform(param_low, param_high)


    return sampled_params


def sample_LV_input_params():
    # a_1 a_2 rate of prey division
    # b rate of predator mortality
    # beta_1 beta_2 Reproduction rate per 1 H
    # alpha_1 alpha_2 Prey death per predator

    params = {'H_1': [0.1, 1.0], 'H_2': [0.1, 1.0], 'P_1': [0.1, 1.0], 
    'a_1': [1.0], 'a_2': [1.0], 'b': [1.5], 
    'beta_1': [0.75], 'beta_2': [0.75], 
    'alpha_1': [1.0], 'alpha_2': [1.0]}

    sampled_params = {}

    for key in params.keys():
        noisy_param = -1 
        while noisy_param < 0:
            noisy_param = params[key][0] + np.random.normal()

        sampled_params[key] = noisy_param

    for key in ['H_1', 'H_2', 'P_1']:
        param_high = params[key][1]
        param_low = params[key][0]
        sampled_params[key] = np.random.uniform(param_low, param_high)


    return sampled_params

def sample_lorenz_map_input_params():
    params = {'R': [5.0, 15.0], 'phi': [5.0, 30.0], 'B': [1.0, 3.0], 
    'x0': [0.0, 0.1], 'y0': [0.5, 1.5], 'z0': [0.01, 0.5]}
    # y0 = [0, 1.0, 0.1]

    sampled_params = {}

    for key in params.keys():
        param_high = params[key][1]
        param_low = params[key][0]
        sampled_params[key] = np.random.uniform(param_low, param_high)

    return sampled_params


def run_LV_separation(idx=0):
    print(idx)
    sampled_params_dict = sample_modified_LV_input_params()
    params_order = ['a_1', 'a_2', 'a_g', 'alpha_1', 'alpha_2', 'b', 'beta_1', 'beta_2']
    sampled_params_list = [sampled_params_dict[p] for p in params_order]

    # Unpack initial conditions
    init_G_1 = sampled_params_dict['G_1']
    init_H_1 = sampled_params_dict['H_1']
    init_H_2 = sampled_params_dict['H_2']
    init_P_1 = sampled_params_dict['P_1']
    y0 = [init_G_1, init_H_1, init_H_2, init_P_1]
    
    t_init = 1000
    sep_len = 1000
    dt = 0.01

    output = calculate_separation(one_pred_two_prey, y0, sampled_params_list, min_species_value=1e-7, t_init=t_init, dt=dt, sep_len=sep_len)
    separation_coefficients = output[1]
    dt = 0.01

    if output[0]:
        mean_sep_coeff = np.mean(separation_coefficients)
        print("mean_coeff: ", mean_sep_coeff)
        print(math.isnan(mean_sep_coeff))

        if not math.isnan(mean_sep_coeff):
            print("hit")
            t = np.arange(0, sep_len, dt)
            y0 = list(output[2])
            y0_theta = list(output[3])

            if 1:
                print(type(y0))
                print(type(sampled_params_list))
                pop_obj = population_modules.Population(2, 0.0, t_init,
                                         dt, [y0, y0_theta], [sampled_params_list, sampled_params_list], [0, 0],
                                         [0, 1, 2], 1e-9, 1e-4)

                pop_obj.generate_particles()
                start_time_sim = time.time()
                pop_obj.simulate_particles()
                end_time_sim = time.time()

                state_list = pop_obj.get_particle_state_list(0)
                state_list_theta = pop_obj.get_particle_state_list(1)
                time_points = pop_obj.get_timepoints_list()

                try:
                    sol = np.reshape(state_list, (len(time_points), 4))
                    sol_theta = np.reshape(state_list_theta, (len(time_points), 4))

                except(ValueError):
                    # print(len(state_list)/len(init_states[sim_idx]))
                    return [False, np.nan]

                t = time_points

                del pop_obj


            if 0:
                sol = odeint(one_pred_two_prey, y0, t, (sampled_params,), mxstep=5000000)
                sol_theta = odeint(one_pred_two_prey, y0_theta, t, (sampled_params,), mxstep=5000000)

            sol = sol[-1000:]
            sol_theta = sol_theta[-1000:]
            t = t[-1000:]

            output_name = "./LV_output/LV_" + str(idx) + ".pdf"
            plot_separation(sol, sol_theta, t, output_name, sep_coeff=np.mean(separation_coefficients))
        
        if not math.isnan(mean_sep_coeff):
            msg = "{0: .1f} , {1: .7f}\n".format(idx, mean_sep_coeff)
            with open(file="./LV_output/LV_sep_coeffs.csv", mode='a', buffering=1) as file:
                file.write(msg)


def plot_separation(sol, sol_theta, t, output_path, sep_coeff, log_timeseries=False):
        width_inches = 95*4 / 25.4
        height_inches = 51*4 / 25.4
        fig, axes = plt.subplots(figsize=(width_inches,height_inches), nrows=4, ncols=2)

        ax_row = axes[0]
        # Plot time series
        for idx, data in enumerate([sol, sol_theta]):
            ax = ax_row[idx]
            sns.lineplot(x=t, y=data[:, 0], ax=ax, estimator=None)
            sns.lineplot(x=t, y=data[:, 1], ax=ax, estimator=None)
            sns.lineplot(x=t, y=data[:, 2], ax=ax, estimator=None)

            if log_timeseries:
                ax.set_yscale('log')

        ax_row = axes[1]
        # Plot prey_1 predator phase
        for idx, data in enumerate([sol, sol_theta]):
            ax = ax_row[idx]
            # sns.lineplot(x=data[:, 0], y=data[:, 2], ax=ax, estimator=None)
            ax.plot(data[:, 0], data[:, 2])

        ax_row = axes[2]
        # Plot prey_2 predator phase
        for idx, data in enumerate([sol, sol_theta]):
            ax = ax_row[idx]
            # sns.lineplot(x=data[:, 1], y=data[:, 2], ax=ax, estimator=None, ci=None)
            ax.plot(data[:, 1], data[:, 2])
            # ax.fill_between(data[:, 1], data[:, 2], color="red", alpha=0.3)

        ax_row = axes[3]
        # Plot prey_2 predator phase
        for idx, data in enumerate([sol, sol_theta]):
            ax = ax_row[idx]
            # sns.lineplot(x=data[:, 1], y=data[:, 2], ax=ax, estimator=None, ci=None)
            ax.plot(data[:, 0], data[:, 1])
            # ax.fill_between(data[:, 1], data[:, 2], color="red", alpha=0.3)

        for ax_row in axes:
            for ax in ax_row:
                ax.unicode_minus = True

                ax.set_ylabel('')
                # ax.set(xlim=(-0.5,None))
                # ax.set(ylim=(-0))
                ax.spines["right"].set_visible(False)
                ax.spines["top"].set_visible(False)
                ax.spines["left"].set_alpha(0.5)
                ax.spines["bottom"].set_alpha(0.5)
                # ax.tick_params(labelsize=15)
                # ax.margins(x=0)
                # ax.margins(y=0)
                fig.tight_layout()
        
        plt.savefig(output_path, dpi=150)
        plt.title(str(sep_coeff))
        plt.close()
        plt.clf()

def calculate_separation(diff_eqs, y0, params, min_species_value, t_init, dt, sep_len):
    n_species = len(y0)

    # Iterate transient
    t = np.arange(0, t_init, dt)

    if 1:
        pop_obj = population_modules.Population(1, 0.0, t_init,
                                                 dt, [y0], [params], [0],
                                                 [0, 1, 2], 1e-9, 1e-4)
        pop_obj.generate_particles()
        start_time_sim = time.time()
        pop_obj.simulate_particles()
        end_time_sim = time.time()

        state_list = pop_obj.get_particle_state_list(0)
        time_points = pop_obj.get_timepoints_list()

        del pop_obj

        try:
            sol = np.reshape(state_list, (len(time_points), 4))

        except(ValueError):
            # print(len(state_list)/len(init_states[sim_idx]))
            time_points = range(int(len(state_list) / 4))
            state_list = np.reshape(state_list, (len(time_points), 4))
            return [False, np.nan]

    if 0:
        try:
            sol = odeint(diff_eqs, y0, t, (params,), mxstep=5000000)
        
        except ODEintWarning:
            return [False, np.nan]

    separation_coefficients = []
    
    y0 = sol[-1].copy()

    # Apply initial separation
    y0_theta = y0.copy()
    theta_0 = 10**-4
    # y0_theta = [y + theta_0 for y in y0]
    y0_theta[2] = y0_theta[2] + theta_0


    # Keep initial for simulation after
    y0_init = y0.copy()
    y0_theta_init = y0_theta.copy()


    y0 = list(y0)
    y0_theta = list(y0_theta)
    dt = 0.001
    t = np.arange(0, dt*2, dt)
    for i in range(sep_len):
        if 1:
            pop_obj = population_modules.Population(2, 0.0, dt*2,
                                                     dt, [y0, y0_theta], [params, params], [0, 0],
                                                     [0, 1, 2], 1e-9, 1e-4)
            
            pop_obj.generate_particles()
            start_time_sim = time.time()
            pop_obj.simulate_particles()
            end_time_sim = time.time()

            state_list = pop_obj.get_particle_state_list(0)
            state_list_theta = pop_obj.get_particle_state_list(1)
            time_points = pop_obj.get_timepoints_list()

            try:
                sol = np.reshape(state_list, (len(time_points), 4))
                sol_theta = np.reshape(state_list_theta, (len(time_points), 4))

            except(ValueError):
                # print(len(state_list)/len(init_states[sim_idx]))
                return [False, np.nan]


        if 0:
            try:
                sol = odeint(diff_eqs, y0, t, (params,), mxstep=5000000)
                sol_theta = odeint(diff_eqs, y0_theta, t, (params,), mxstep=5000000)
            
            except ODEintWarning as e:
                print(e)
                print("here")
                return [False, np.nan, y0_init, y0_theta_init]

        # Calculate step separation 
        theta_1 = 0
        for n in range(n_species):
            y_a = sol[:, n][-1]

            if y_a < min_species_value:
                return [False, np.nan, y0_init, y0_theta_init]


            y_b = sol_theta[:, n][-1]

            theta_1 += (y_a - y_b)**2 

        theta_1 = theta_1**0.5
        theta_1 = theta_1/dt

        y0 = sol[-1]
        y0_theta = sol_theta[-1]

        # Readjust orbit
        y_theta_end = sol_theta[-1]

        for n in range(n_species):
            y_a = y0[n]
            y_b = y_theta_end[n]

            y0_theta[n] = y_a + (theta_0 / theta_1) * (y_b - y_a)

        y0 = list(y0)
        y0_theta = list(y0_theta)

        separation_coefficients.append(np.log2(abs(theta_1/theta_0)))

    # print(separation_coefficients)
    # plt.plot(range(75), separation_coefficients)
    # plt.show()

        # print(theta_0)

    return [True, separation_coefficients, y0_init, y0_theta_init]


def run_lorenz_map(idx):

    t_init = 50000.0
    sep_len = 10000
    dt = 0.001

    sampled_params = sample_lorenz_map_input_params()

    y0 = [sampled_params['x0'], sampled_params['y0'], sampled_params['z0']]

    output = calculate_separation(lorenz_map, y0, sampled_params, 
        t_init=t_init, dt=dt, sep_len=10000)

    separation_coefficients = output[1]

    if output[0]:
        mean_sep_coeff = np.mean(separation_coefficients)

        if mean_sep_coeff != np.nan:
            print("hit")
            t = np.arange(0, 1000, dt)
            y0 = output[2]
            y0_theta = output[3]

            sol = odeint(lorenz_map, y0, t, (sampled_params,), mxstep=5000000)
            sol_theta = odeint(lorenz_map, y0_theta, t, (sampled_params,), mxstep=5000000)

            sol = sol[-30000:]
            sol_theta = sol_theta[-30000:]
            t = t[-30000:]

            output_name = "./lorenz_output/LM_" + str(idx) + ".pdf"
            plot_separation(sol, sol_theta, t, output_name, sep_coeff=np.mean(separation_coefficients))
        
        if mean_sep_coeff != np.nan:
            msg = "{0: .1f} , {1: .7f}\n".format(idx, mean_sep_coeff)
            with open(file="./lorenz_output/LM_sep_coeffs.csv", mode='a', buffering=1) as file:
                file.write(msg)



    # fig, ax = plt.subplots(nrows=1, ncols=1)
    # ax.plot(t, sol[:, 0])
    # ax.plot(t, sol[:, 1])
    # ax.plot(t, sol[:, 2])
    # fig.tight_layout()

    # plt.show()


def lyapunov_notes():
    # Lyapunov exponent quantifies the rate of separation 
    # of infinitesimally close trajectories.

    # There is a spectrum of Lyapunov exponents—equal in number to the 
    # dimensionality of the phase space.

    # It is common to refer to the largest one as the 
    # Maximal Lyapunov exponent (MLE), because it determines 
    # a notion of predictability for a dynamical system.

    # A positive MLE is usually taken as an indication that the system is chaotic 
    # (provided some other conditions are met, e.g., phase space compactness).

    # Note that an arbitrary initial separation vector will typically contain 
    # some component in the direction associated with the MLE, and because of the 
    # exponential growth rate, the effect of the other exponents will be obliterated over time. 

    # The Lyapunov exponents describe the behavior of vectors in the tangent space of the phase space 
    # and are defined from the Jacobian matrix


    # http://sprott.physics.wisc.edu/chaos/lyapexp.htm
    # http://csc.ucdavis.edu/~chaos/courses/nlp/Software/partH.html
    
    # The separation is calculated from the sum of the squares of the differences in each variable.  
    # So for a 2-dimensional system with variables x and y, 
    # the separation would be d = [(xa - xb)2 + (ya - yb)2]1/2, where the subscripts (a and b) 
    # denote the two orbits respectively.

    # If the system consists of ordinary differential equations (a flow) instead of 
    # difference equations (a map), the procedure is the same except that the resulting 
    # exponent is divided by the iteration step size so that it has units of inverse seconds 
    # instead of inverse iterations.  You will typically need millions of iterations of the 
    # differential equations to get a result good to better than about two significant digits
    pass



if __name__ == "__main__":
    base_idx = 0
    for i in range(int(1e6)):
        idx = base_idx + i
        # run_lorenz_map(idx)
        # exit()
        run_LV_separation(idx)
    exit()
    # lorenz_sympy()
    # exit()
    run_lorenz_map()

