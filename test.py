import numpy as np
# import population_modules as Population
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

def change_this(a_list):
    a_list[0] = 1234

def weights_help():
    prior_a = [1e-22, 1e-15]
    print(prior_a)
    change_this(prior_a)
    print(prior_a)
    exit()
    prior_a = [abs(np.log(prior_a[0])), abs(np.log(prior_a[1]))]
    prior_b = [1, 1e-2]
    prior_b = [abs(np.log(prior_b[0])), abs(np.log(prior_b[1]))]

    scale_a = max(prior_a) - min(prior_a)
    kern_a = [-scale_a/2, scale_a/2]

    scale_b = max(prior_b) - min(prior_b)
    kern_b = [-scale_b/2, scale_b/2]

    x_0_a = np.log(1e-20)
    x_0_b = 3

    max_a_val = x_0_a + kern_a[1]
    min_a_val = x_0_a + kern_a[0]

    max_b_val = x_0_b + kern_b[1]
    min_b_val = x_0_b + kern_b[0]

    denom_a = max_a_val - min_a_val
    denom_b = max_b_val - min_b_val

    p_x_a = 1/denom_a
    p_x_b = 1/denom_b

    print(p_x_a)
    print(p_x_b)

def repressor_test():
    k_a = 0.5
    b = np.arange(0, 2, 0.1)
    rpr_hill = lambda k, x: k / (k + x)
    omega_max = 1
    k_I_1 = 1e-20
    nI_1 = 1.5
    n_omega = 1.5
    k_omega_B_1 = 1e-17

    omega_immun = lambda B_1, I_1:  (  omega_max * B_1**n_omega / ( k_omega_B_1**n_omega + B_1**n_omega )  *  ( k_I_1**nI_1 / ( k_I_1**nI_1 + I_1**nI_1 ) )  )
    vals = []

    B_val = 1e-19
    I_vals = np.arange(np.log(1e-25), np.log(1e-16))
    I_vals = np.exp(I_vals)
    print(I_vals)
    for conc in I_vals:
        v = omega_immun(B_val, conc)
        vals.append(v)

    print(vals)
    # for conc in b:
    #     v = rpr_hill(k_a, conc)
    #     print(v)
    #     vals.append(rpr_hill(k_a, conc))

    # print(vals)
    plt.plot(I_vals, vals)
    plt.xscale('log')
    # plt.yscale('log')
    plt.show()


def growth_test():
    k_I_1 = 1e-50
    I_1 = 1e-20
    k_omega_B_1 = 1e8
    B_1 = 1e-20
    nI_1 = 1.5

    dN_1 = - ( ( k_omega_B_1 * B_1 ) *  ( k_I_1**nI_1 / ( k_I_1**nI_1 + I_1**nI_1 ) )  )
    print(dN_1)



def histogram_comparison():
    top_val = 1
    bottom_val = 1e8
    n_bins = 1000

    x = np.random.uniform(bottom_val, top_val, 100000)
    # hist, bins, _ = plt.hist(x, bins=n_bins)
    # plt.close()

    values, bins, _ = plt.hist(x, bins=n_bins, density=True)
    # plt.xscale('log')
    # plt.show()
    print(bins)
    
    d = []
    for idx in range(len(np.diff(bins))):
        d.append(values[idx] * np.diff(bins)[idx])

    print(sum(d))
    exit()

    area = sum(np.diff(bins)*values) 
    print(sum(values))
    print(sum(bins))
    print(area)
    exit()
    for idx in range(len(np.diff(bins))):
        comb_vals.append(values[idx] * np.diff(bins)[idx])

    print(np.std(comb_vals))

    # plt.show()

    y = np.exp(np.random.uniform(np.log(bottom_val), np.log(top_val), 100000))
    hist, bins, _ = plt.hist(y, bins=n_bins)
    logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]), n_bins)

    values, bins, _ = plt.hist(y, bins=logbins)
    comb_vals = []
    for idx in range(len(np.diff(bins))):
        comb_vals.append(values[idx] * np.diff(bins)[idx])

    print(np.std(comb_vals))
    exit()

    # plt.show()
    area = sum(np.diff(bins)*values)
    print(area)


def norm_test():
    omega_max = 10
    B_2 = 0.00001
    n_omega = 2
    k_omega_B_2 = 0.001
    

    x = (  omega_max * (B_2)**n_omega / ( k_omega_B_2**n_omega + (B_2)**n_omega )  )

    print(x)


    C_extra = 0.0001
    B_2 = B_2/C_extra
    x = (  omega_max * (C_extra * B_2)**n_omega / ( k_omega_B_2**n_omega + (C_extra * B_2)**n_omega )  )
    print(x)




if __name__ == "__main__":
    norm_test()
    exit()
    histogram_comparison()
    exit()
    growth_test()
    exit()
    repressor_test()
    # weights_help()
    exit()
    model_sample_example()
    # alt_distance()
    # test_this()
    # linear_repression_model()
    # main()