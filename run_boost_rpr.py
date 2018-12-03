import algorithms
from model_space import Model
import xmltodict

def main():
    t_0 = 0
    t_end = 1000
    dt = 0.01

    model_rpr_prior_dict = {
        'alpha0': (0, 1),
        'alpha': (1000, 1000),
        'n': (1, 2),
        'beta': (1, 5),
    }

    model_lv_prior_dict = {
        'A': (0, 10),
        'B': (0, 10),
        'C': (0, 10),
        'D': (0, 10)
    }

    model_spock_dict = {
        'D': (0.01, 0.5),
        'mux_m': (0.4, 3.0),
        'muc_m': (0.4, 3.0),
        'Kx': (1.5e-5, 1.5e-5),
        'Kc': (1.5e-5, 1.5e-5),
        'omega_c_max': (0.5, 2),
        'K_omega': (1e-7, 1e-6),
        'n_omega': (1, 2),
        'S0': (4.0, 4.0),
        'gX': (1e12, 1e12),
        'gC': (1e12, 1e12),
        'C0L': (5e-5, 1e-4),
        'KDL': (3e-8, 8e-8),
        'nL': (2.0, 3.8),
        'K1L': (0.00015, 0.0004),
        'K2L': (2500, 5000),
        'ymaxL': (2500,  5000),
        'K1T': (0, 5000),
        'K2T': (0,  100),
        'ymaxT': (16, 27),
        'C0B': (2e-5, 1e-4),
        'LB': (0, 1e-2),
        'NB': (0.7, 1.1),
        'KDB': (0.002, 0.010),
        'K1B': (0.0, 0.08),
        'K2B': (800, 5000),
        'K3B': (1000, 50000),
        'ymaxB': (100, 500),
        'cgt': (0.0, 0.1),
        'k_alpha_max': (1e-22, 1e-15),
        'k_beta_max': (1e-22, 1e-15),
    }

    # Initialise models
    rpr_model = Model(0, model_rpr_prior_dict)
    lv_model = Model(1, model_lv_prior_dict, )
    spock_model = Model(2, model_spock_dict)

    model_list = [spock_model]

    # Initialise model space
    ABC = algorithms.ABC_rejection(t_0, t_end, dt, model_list, 3, 100, 2, 2)
    ABC.run_abc_rejection()

    # algorithms.abc_rejection(t_0, t_end, dt, model_list, particles_per_population=10,
    #               n_sims=50, n_species_fit=2, n_distances=2)


if __name__ == "__main__":
    main()




