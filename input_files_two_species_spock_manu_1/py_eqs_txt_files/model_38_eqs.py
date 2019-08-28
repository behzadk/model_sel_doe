	dN_1 = ( - D * N_1 ) + N_1  * ( mu_max_1 * S_glu / ( K_mu_glu + S_glu ) )

	dN_2 = ( - D * N_2 ) + N_2  * ( mu_max_2 * S_glu / ( K_mu_glu + S_glu ) )

	dS_glu = ( D * ( S0_glu - S_glu ) ) - ( mu_max_1 * S_glu / ( K_mu_glu + S_glu ) ) * N_1 / g_1  - ( mu_max_2 * S_glu / ( K_mu_glu + S_glu ) ) * N_2 / g_2 

	dI_1 =   +  kI_max_1  - I_1  * ( ( mu_max_1 * S_glu / ( K_mu_glu + S_glu ) ) )  / 2 

