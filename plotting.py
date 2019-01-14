import matplotlib.pyplot as plt


def plot_simulation(out_pdf, model_ref, state, time_points, plot_species_idx):
    fig = plt.figure()
    for i in plot_species_idx:
        plt.plot(time_points, state[:, i])

    plt.yscale('log')
    # plt.ylim(1, 1e12)
    plt.title("Model idx: " + str(model_ref))
    out_pdf.savefig()
    plt.close()

