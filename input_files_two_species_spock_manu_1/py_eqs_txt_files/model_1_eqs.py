	dN_1 = ( - D * N_1 ) + N_1  * ( mu_max_1 * S_glu / ( K_mu_glu + S_glu ) ) - ( ( k_omega_B_1 * B_1 ) *  ( k_I_1**nI_1 / ( k_I_1**nI_1 + I_1**nI_1 ) )  )  * N_1 

	dN_2 = ( - D * N_2 ) + N_2  * ( mu_max_2 * S_glu / ( K_mu_glu + S_glu ) )

	dS_glu = ( D * ( S0_glu - S_glu ) ) - ( mu_max_1 * S_glu / ( K_mu_glu + S_glu ) ) * N_1 / g_1  - ( mu_max_2 * S_glu / ( K_mu_glu + S_glu ) ) * N_2 / g_2 

	dB_1 = ( - D * B_1 ) +  kB_max_1  * ( A_1**n_A_B_1 / ( K_A_B_1**n_A_B_1 + A_1**n_A_B_1 ) ) * N_1 

	dA_1 = ( - D * A_1 ) + kA_1 * N_1

	dI_1 =   +  kI_max_1  * ( A_1**n_A_I_1 / ( K_A_I_1**n_A_I_1  + A_1**n_A_I_1) ) - I_1  * ( ( mu_max_1 * S_glu / ( K_mu_glu + S_glu ) ) )  / 2 

