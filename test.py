import numpy as np


if __name__ == "__main__":
	mu_max = 1.5
	S = 4
	K_x = 0.4
	D = 0.1

	omega_max = 1
	beta = 0.5
	n=1
	k_omega = 0.3

	N = 1e3

	t1 = (mu_max * S) / (K_x + S)
	t2 = D
	t3 = omega_max * pow(beta, n) / k_omega + pow(beta, n)

	total = (t1 - t2 - t3) * N

	print(total)

	# Scaling
	c = 1e3
	N = 1

	# omega_max = omega_max/c
	# mu_max = mu_max/c
	# # S = S/c
	# D = D/c
	# # beta = beta/c

	t1 = (mu_max * S) / (K_x + S)
	t2 = D
	t3 = omega_max * pow(beta, n) / k_omega + pow(beta, n)

	total = (t1 - t2 - t3) * N * c

	print(total)
