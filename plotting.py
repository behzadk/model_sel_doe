import matplotlib.pyplot as plt
import numpy as np

def plot_simulation(out_pdf, sim_idx, model_ref, state, time_points, plot_species_idx, error_msg=None):
    plt_title = "Model idx: " + str(model_ref) + " Sim_idx: " + str(sim_idx)

    if error_msg is not None and error_msg is not np.nan and error_msg is not "":
        plt_title = plt_title + " error: " + error_msg

        # for col in range(np.shape(state)[1]):
        #     print(col)
        #     print(np.min(state[:, col]))
        # print("")




    fig = plt.figure()
    for i in plot_species_idx:
        plt.plot(time_points, state[:, i], label=str(i))
        # print("max: ", np.max(state[:, i]))
        # print("min: ", np.min(state[:, i]))

    plt.yscale('log')
    plt.legend()
    # plt.ylim(1, 1e12)
    plt.title(plt_title)
    out_pdf.savefig()
    plt.close()


