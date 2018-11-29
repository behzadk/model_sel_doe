import algorithms
from model_space import Model


def main():
    t_0 = 0
    t_end = 100
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

    # Initialise models
    rpr_model = Model(0, model_rpr_prior_dict)
    lv_model = Model(1, model_lv_prior_dict, )
    model_list = [rpr_model, lv_model]

    # Initialise model space

    algorithms.abc_rejection(t_0, t_end, dt, model_list, particles_per_population=10,
                  n_sims=50, n_species_fit=2, n_distances=2)


if __name__ == "__main__":
    main()