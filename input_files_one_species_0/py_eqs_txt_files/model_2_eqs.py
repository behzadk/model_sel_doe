	dN_1 = ( - D * N_1 ) + N_1  * ( mu_max_1 * S_glu / ( K_mu_glu + S_glu ) ) - ( ( omega_max_1 * B_1 ) * ( K_V_1 / ( V_1 + K_V_1 ) ) )  * N_1 

	dN_2 = ( - D * N_2 ) + N_2  * ( mu_max_2 * S_glu / ( K_mu_glu + S_glu ) ) - ( ( omega_max_1 * B_1 ) * ( K_V_1 / ( V_1 + K_V_1 ) ) )  * N_2 

	dS_glu = ( D * ( S0_glu - S_glu ) ) - ( mu_max_1 * S_glu / ( K_mu_glu + S_glu ) ) * N_1 * C / g_1  - ( mu_max_2 * S_glu / ( K_mu_glu + S_glu ) ) * N_2 * C / g_2 

	dB_1 = ( - D * B_1 ) +  kBmax_1  * ( A_1**nB_1 / ( KB_1**nB_1 + A_1**nB_1 ) ) * N_1 * C 

	dA_2 = ( - D * A_2 ) + kA_2 * N_1 * C

	dA_1 = ( - D * A_1 ) + kA_1 * N_1 * C

	dV_1 = ( - D * V_1 ) +  kV_max_1  * ( kV_1**nV_1  / ( kV_1**nV_1  + A_2**nV_1 ) ) * N_1 * C 

