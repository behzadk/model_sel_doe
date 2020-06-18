from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

def diff_eqs(y, t, part_params):
	C_OD = part_params[0];
	D = part_params[1];
	g_1 = part_params[2];
	K_mu_glu = part_params[3];
	mu_max_1 = part_params[4];
	S0_glu = part_params[5];

	N_1 = y[0]
	S_glu = y[1]

	dN_1 = ( - D * N_1 ) + N_1  * ( mu_max_1 * S_glu / ( K_mu_glu + S_glu ) )
	dS_glu = ( D * ( S0_glu - S_glu ) ) - ( mu_max_1 * S_glu / ( K_mu_glu + S_glu ) ) * N_1 * C_OD / g_1


	return [dN_1, dS_glu]



def model_7(y, t, part_params):

	C_extra = part_params[0];
	C_OD = part_params[1];
	D = part_params[2];
	g_1 = part_params[3];
	g_2 = part_params[4];
	K_A_B_1 = part_params[5];
	K_A_B_2 = part_params[6];
	K_mu_glu = part_params[7];
	k_omega_B_1 = part_params[8];
	k_omega_B_2 = part_params[9];
	kA_1 = part_params[10];
	kA_2 = part_params[11];
	kB_max_1 = part_params[12];
	kB_max_2 = part_params[13];
	mu_max_1 = part_params[14];
	mu_max_2 = part_params[15];
	n_A_B_1 = part_params[16];
	n_A_B_2 = part_params[17];
	n_omega = part_params[18];
	omega_max = part_params[19];
	S0_glu = part_params[20];


	N_1 = y[0]
	N_2 = y[1]
	S_glu = y[2]
	B_1 = y[3]
	B_2 = y[4]
	A_1 = y[5]
	A_2 = y[6]

	dN_1 = ( - D * N_1 ) + N_1  * ( mu_max_1 * S_glu / ( K_mu_glu + S_glu ) ) - (  omega_max * (C_extra * B_1)**n_omega / ( k_omega_B_1**n_omega + (C_extra *  B_1)**n_omega )  )  * N_1 

	dN_2 = ( - D * N_2 ) + N_2  * ( mu_max_2 * S_glu / ( K_mu_glu + S_glu ) ) - (  omega_max * (C_extra * B_2)**n_omega / ( k_omega_B_2**n_omega + (C_extra *  B_2)**n_omega )  )  * N_2 

	dS_glu = ( D * ( S0_glu - S_glu ) ) - ( mu_max_1 * S_glu / ( K_mu_glu + S_glu ) ) * N_1 * C_OD / g_1  - ( mu_max_2 * S_glu / ( K_mu_glu + S_glu ) ) * N_2 * C_OD / g_2 

	dB_1 = ( - D * B_1 ) +  kB_max_1  * ( (C_extra * A_1)**n_A_B_1 / ( K_A_B_1**n_A_B_1 + (C_extra * A_1)**n_A_B_1 ) ) * N_1 * C_OD / C_extra 

	dB_2 = ( - D * B_2 ) +  kB_max_2  * ( (C_extra * A_2)**n_A_B_2 / ( K_A_B_2**n_A_B_2 + (C_extra * A_2)**n_A_B_2 ) ) * N_2 * C_OD / C_extra 

	dA_1 = ( - D * A_1 ) + kA_1 * N_1 * C_OD / C_extra

	dA_2 = ( - D * A_2 ) + kA_2 * N_2 * C_OD / C_extra

	return [dN_1, dN_2, dS_glu, dB_1, dB_2, dA_1, dA_2]


def run_sim(t_0, t_end, dt, init_states, input_params):
	y0 = init_states
	# y0[1] = 0.02
	# print(y0)

	# # y0[0] = y0[0] * 1e8
	# input_params[0] = 1e9
	# input_params[1] = 0
	# input_params[2] = 10**11
	# input_params[4] = 0.6

	# input_params[5] = 0.02

	print(input_params)

	t = np.arange(t_0, t_end, step=dt)
	sol = odeint(model_7, y0, t, args=(input_params,))

	# sol[:, 0] = sol[:, 0] / 1e8
	end_N_1 = sol[:, 0][-1]
	end_N_2 = sol[:, 1][-1]

	if end_N_1 > 0.001 and end_N_2 > 0.001:
		plt.plot(t, sol[:, 0], sol[:, 1])
		plt.show()


