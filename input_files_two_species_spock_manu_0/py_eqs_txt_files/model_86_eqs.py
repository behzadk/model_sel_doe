	dN_1 = ( - D * N_1 ) + N_1  * ( mu_max_1 * S_glu / ( K_mu_glu + S_glu ) ) - ( ( omega_max_1 * B_1**n_omega_1 / ( K_omega_1**n_omega_1 + B_1**n_omega_1 ) ) )  * N_1 

	dN_2 = ( - D * N_2 ) + N_2  * ( mu_max_2 * S_glu / ( K_mu_glu + S_glu ) ) * ( mu_max_2 * S_trp / ( K_mu_trp + S_trp ) )

	dS_trp = ( D * ( S0_trp - S_trp ) ) - ( mu_max_2 * S_trp / ( K_mu_trp + S_trp ) ) * N_2 * C / g_2  +  C * N_1 * p_trp 

	dS_glu = ( D * ( S0_glu - S_glu ) ) - ( mu_max_1 * S_glu / ( K_mu_glu + S_glu ) ) * N_1 * C / g_1  - ( mu_max_2 * S_glu / ( K_mu_glu + S_glu ) ) * N_2 * C / g_2 

	dB_1 = ( - D * B_1 ) +  kBmax_1  * ( KB_1**nB_1 / ( KB_1**nB_1 + A_1**nB_1 ) ) * N_1 * C 

	dA_1 = ( - D * A_1 ) + kA_1 * N_1 * C
