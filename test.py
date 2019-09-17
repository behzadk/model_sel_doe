import numpy as np
import population_modules as Population
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

step_size = 0.5

def get_frequency(x):
    fourier = np.fft.rfft(x)
    mag = abs(fourier)**2
    maxa = np.argmax(mag[1:]) + 1
    print("argmax: ", maxa)
    r = np.fft.fftfreq(len(x), d=step_size)
    # maxa = np.argmax(r[1:]) + 1
    print(r[maxa])
    # exit()
    f = abs(r[maxa]) * 2

    return f

def get_fft(x):
    fourier = np.fft.rfft(x)
    fourier = abs(fourier)**2
    maxa = np.argmax(fourier[1:]) + 1

    return fourier[maxa]


def fft_main():
    print("Starting test")
    angle = 0.0
    t_end = 5000
    f = 0.005
    amp = 10
    shift = 0


    time = np.arange(0, t_end, step_size)
    n_samples = len(time)

    signal = []

    for t in time:
        y = amp * np.cos(f * t * np.pi)
        signal.append(y)

    # y = amp * np.cos(f * time * np.pi)
    # sin_func = lambda t: np.sin(f * t)


    print("Python: ")
    print("True result")
    print("sample rate = ", 1/step_size)
    print("time increment = ", step_size)
    print("n_samples = ", len(time))
    print("frequency = ", f)
    print("period = ", 1/(f/step_size) *  1/step_size )
    print("")
    print("")

    print("")
    print("")
    print("cpp: ")
    Population.test_fft(f, amp, t_end, step_size)
    print("")
    print("")

    print("Using numpy fftfreq: ")
    fou_freq =  get_frequency(signal)
    print("np fftfreq output =  ", fou_freq)

    # fou_freq = fou_freq/step_size *  1/step_size
    print("np fftfreq output =  ", fou_freq)
    print("Fourier periodicity =  ", 1/fou_freq)
    print("")
    print("")

    print("Using numpy fft.rfft: ")
    rfft_out = get_fft(signal)
    print(rfft_out)


    plt.plot(time, signal)
    plt.show()


def linear_repression_model():
    B = np.linspace(1e-10, 100, 5)
    V = np.linspace(1e-10, 100, 5)
    K = 10


    for v in V:
        signal = []

        for b in B:
            # signal.append(b*K)
            print(((b +v)/v))
            signal.append((b +v)/v)



        plt.plot(B, signal)
    plt.show()

def test_this():
    model_0_prob = 0.8
    model_1_prob = 0.2
    n_samples = int(1e6)

    model_0_accepted_count = 0
    model_1_accepted_count = 0

    for i in range(n_samples):
        if np.random.randint(0, 2) == 0:
            if np.random.uniform(0, 1) < model_0_prob:
                model_0_accepted_count += 1

        else:
            if np.random.uniform(0, 1) < model_1_prob:
                model_1_accepted_count += 1

    print(model_0_accepted_count)
    print(model_1_accepted_count)

    total_accpeted = model_0_accepted_count + model_1_accepted_count

    model_0_posterior_prob = model_0_accepted_count/ total_accpeted
    model_1_posterior_prob = model_1_accepted_count/ total_accpeted

    print(model_0_posterior_prob/model_1_posterior_prob)


def alt_distance():
    distance_path = "./output/two_species_stable_0/Population_0/distances.csv"

    df = pd.read_csv(distance_path)
    df = df.loc[df['Accepted'] == True]
    df['d3'] = df['d3'].astype(np.float64)
    df['d6'] = df['d6'].astype(np.float64)
    df['model_ref'] = df['model_ref'].astype(int)

    print(list(df))
    print(len(df))
    # df = df.loc[df['d3'] > 0.001]
    # df = df.loc[df['d6'] > 0.001]

    model_idx = []
    n_accepted = []
    pop_diff = []
    d3 = []
    d6 = []

    for m in df.model_ref.unique():
        n_accepted.append(len(df.loc[df['model_ref'] == m]))
        model_idx.append(m)
        d3_vals = df.loc[df['model_ref'] == m]['d3'].values
        d6_vals = df.loc[df['model_ref'] == m]['d6'].values
        diff = []
        for i, j in zip(d3_vals, d6_vals):
            ratio = i/j
            if ratio < 1:
                ratio = 1/ratio
            diff.append(ratio)

        d3.append(np.mean(d3_vals))
        d6.append(np.mean(d6_vals))
        pop_diff.append(np.mean(diff))

    counts_df = pd.DataFrame()
    counts_df['model_ref'] = model_idx
    counts_df['n_accepted'] = n_accepted
    counts_df['pop_diff'] = pop_diff
    counts_df['d3'] = d3
    counts_df['d6'] = d6

    counts_df.sort_values(['pop_diff'], inplace=True, ascending=True)
    print(counts_df)


def model_sample_example():
    model_1 = {}
    model_2 = {}
    model_3 = {}

    models = [model_1, model_2, model_3]

    true_probability = [0.8, 0.5, 0.1]


    def update_weights(all_models, sigma=0.7):
        for m in all_models:
            m['prev_weight'] = m['current_weight']
            m['current_weight'] = 0

        for m_i in all_models:
            if m_i['pop_n_sims'] == 0:
                m_i['current_weight'] = 0
                continue

            bm = m_i['pop_n_accepted'] / m_i['pop_n_sims']

            denom_m = 0

            for m_j in all_models:
                if m_i == m_j:
                    denom_m += m_j['prev_weight']

                for particle in range(m_j['pop_n_sims']):
                    denom_m += m_j['prev_weight'] * np.random.normal(1, 0.5)

            m_i['current_weight'] = bm * denom_m

        # Normalise weights
        sum_weights = sum([m['current_weight'] for m in all_models])
        for m in all_models:
            m['current_weight'] = m['current_weight']/sum_weights
            # print(m['current_weight'])

        return all_models



    for idx, m in enumerate(models):
        m['idx'] = idx
        m['true_prob'] = true_probability[idx]
        m['current_weight'] = 1/len(models)
        m['prev_weight'] = None
        m['pop_n_sims'] = 0
        m['pop_n_accepted'] = 0

    n_pop = 1000

    population_num = 0
    while population_num < 1000:
        print(population_num)
        weights = [m['current_weight'] for m in models]
        population_accepted = 0


        # Simulate models for first t = 1
        while population_accepted < n_pop:
            m = np.random.choice(models, p=weights)
            m['pop_n_sims'] += 1

            # If accepted
            if np.random.uniform(0, 1) < m['true_prob']:
                m['pop_n_accepted'] += 1
                population_accepted += 1

        # Calculate new weights
        models = update_weights(models)

        # Reset population
        for m in models:
            if m['pop_n_sims'] == 0:
                print(m['idx'], 0)

            else:
                print(m['idx'], m['current_weight'])

            m['pop_n_sims'] = 0
            m['pop_n_accepted'] = 0

        print("")
        population_num += 1





if __name__ == "__main__":
    model_sample_example()
    # alt_distance()
    # test_this()
    # linear_repression_model()
    # main()