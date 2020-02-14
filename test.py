import numpy as np
# import population_modules as Population
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import odeint

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


def non_dim_test():
    x = 2
    y = 5

    dx = 3*x + 2*y - 10
    dy = 2*x + y*y + 7

    print(dx)
    print(dy)

    c = 0.1 # x norm val
    k = 0.01 # y norm val

    x = x/c
    y = y/k

    d_norm_x = 3*x*c + 2*y*k -10
    d_norm_y = 2*x*c + (y*k)**2 + 7

    print(d_norm_x)
    print(d_norm_y)


def non_dim_test_2():
    D = 0.01
    mu_max_1 = 0.5
    S_glu = 4.0
    K_mu_glu = 0.01
    omega_max = 1.0
    n_omega = 2
    k_omega_B_1 = 0.01
    kB_max_1 = 2

    A_1 = 2
    n_A_B_1 = 2
    K_A_B_1 = 0.1

    N_1 = 100
    B_1 = 2

    C_extra = 1
    C_OD = 1
    dN_1 = ( - D * N_1 ) + N_1  * ( mu_max_1 * S_glu / ( K_mu_glu + S_glu ) ) - (  omega_max * (B_1)**n_omega / ( k_omega_B_1**n_omega + (B_1)**n_omega )  )  * N_1 
    dB_1 = ( - D * B_1 ) +  kB_max_1  * ( A_1**n_A_B_1 / ( K_A_B_1**n_A_B_1 + A_1**n_A_B_1 ) ) * N_1



    N_0 = 100
    B_0 = 1000
    Nq = N_1 / N_0
    Bq = B_1 / B_0


    dNq = ( - D * Nq ) + Nq  * ( mu_max_1 * S_glu / ( K_mu_glu + S_glu ) ) - (  omega_max * (Bq*B_0)**n_omega / ( k_omega_B_1**n_omega + (Bq*B_0)**n_omega )  ) * Nq 
    dBq = ( - D * Bq ) +  kB_max_1  * ( A_1**n_A_B_1 / ( K_A_B_1**n_A_B_1 + A_1**n_A_B_1 ) ) * Nq*N_0 / B_0


    print("")
    print(N_1 + dN_1)
    print(B_1 + dB_1)

    print((Nq + dNq) * N_0)
    print((Bq + dBq) * B_0)



def non_dim_test_3():
    a = 2
    b = 5
    c = 10

    x = 3


    dx = a + b*x + c*x**2


    x_0 = 100
    q = x / x_0

    dq = (a / x_0) + b * q + (c*x_0) * q**2

    print(dx)
    print(dq)

    print(x + dx)
    print((q + dq) * x_0)


def non_dim_test_4():
    D = 0.01
    mu_max_1 = 2.0
    mu_max_2 = 3.0
    mu_max_3 = 2.5

    K_mu_glu = 0.1
    S0_glu = 4.0
    g_1 = g_2 = g_3 = 100

    omega_max = 1.0
    n_omega = 2.0
    k_omega_B_2 = k_omega_B_1 = k_omega_B_3 = 0.001
    kB_max_1 = kB_max_2 = kB_max_3 = 1.2

    K_A_B_2 = K_A_B_1 = K_A_B_3 = 0.1
    n_A_B_2 = n_A_B_1 = n_A_B_3 = 2.0

    kA_1 = kA_2 = kA_3 = 0.1




    C_OD = 1
    C_extra = 1

    N_1 = 0.5 / C_OD
    N_2 = 1.5 / C_OD
    N_3 = 0.25 / C_OD


    B_1 = 0.1 / C_extra
    B_2 = 1 / C_extra
    B_3 = 0.0001 / C_extra

    A_1 = 0.1 / C_extra
    A_2 = 0.2 / C_extra
    A_3 = 0.4 / C_extra

    S_glu = 3.0


    dN_1 = ( - D * N_1 ) + N_1  * ( mu_max_1 * S_glu / ( K_mu_glu + S_glu ) ) - (  omega_max * (C_extra * B_3)**n_omega / ( k_omega_B_3**n_omega + (C_extra *  B_3)**n_omega )  )  * N_1 

    dN_2 = ( - D * N_2 ) + N_2  * ( mu_max_2 * S_glu / ( K_mu_glu + S_glu ) ) - (  omega_max * (C_extra * B_1)**n_omega / ( k_omega_B_1**n_omega + (C_extra *  B_1)**n_omega )  )  * N_2 

    dN_3 = ( - D * N_3 ) + N_3  * ( mu_max_3 * S_glu / ( K_mu_glu + S_glu ) ) - (  omega_max * (C_extra * B_1)**n_omega / ( k_omega_B_1**n_omega + (C_extra *  B_1)**n_omega )  )  * N_3 

    dS_glu = ( D * ( S0_glu - S_glu ) ) - ( mu_max_1 * S_glu / ( K_mu_glu + S_glu ) ) * N_1 * C_OD / g_1  - ( mu_max_2 * S_glu / ( K_mu_glu + S_glu ) ) * N_2 * C_OD / g_2  - ( mu_max_3 * S_glu / ( K_mu_glu + S_glu ) ) * N_3 * C_OD / g_3 

    dB_3 = ( - D * B_3 ) +  kB_max_3  * ( K_A_B_3**n_A_B_3 / ( K_A_B_3**n_A_B_3 + (C_extra * A_2)**n_A_B_3 ) ) * N_3 * C_OD / C_extra 

    dB_1 = ( - D * B_1 ) +  kB_max_1  * ( (C_extra * A_1)**n_A_B_1 / ( K_A_B_1**n_A_B_1 + (C_extra * A_1)**n_A_B_1 ) ) * N_1 * C_OD / C_extra  +  kB_max_1  * ( K_A_B_1**n_A_B_1 / ( K_A_B_1**n_A_B_1 + (C_extra * A_1)**n_A_B_1 ) ) * N_2 * C_OD / C_extra 

    dA_1 = ( - D * A_1 ) + kA_1 * N_1 * C_OD / C_extra + kA_1 * N_2 * C_OD / C_extra

    dA_2 = ( - D * A_2 ) + kA_2 * N_3 * C_OD / C_extra


    d_raw_list = [dN_1, dN_2, dN_3, dS_glu, dB_1, dB_3, dA_2, dA_1]


    C_OD = 100
    C_extra = 1000

    N_1 = 0.5 / C_OD
    N_2 = 1.5 / C_OD
    N_3 = 0.25 / C_OD


    B_1 = 0.1 / C_extra
    B_2 = 1 / C_extra
    B_3 = 0.0001 / C_extra

    A_1 = 0.1 / C_extra
    A_2 = 0.2 / C_extra
    A_3 = 0.4 / C_extra

    S_glu = 3.0


    dN_1 = ( - D * N_1 ) + N_1  * ( mu_max_1 * S_glu / ( K_mu_glu + S_glu ) ) - (  omega_max * (C_extra * B_3)**n_omega / ( k_omega_B_3**n_omega + (C_extra *  B_3)**n_omega )  )  * N_1 

    dN_2 = ( - D * N_2 ) + N_2  * ( mu_max_2 * S_glu / ( K_mu_glu + S_glu ) ) - (  omega_max * (C_extra * B_1)**n_omega / ( k_omega_B_1**n_omega + (C_extra *  B_1)**n_omega )  )  * N_2 

    dN_3 = ( - D * N_3 ) + N_3  * ( mu_max_3 * S_glu / ( K_mu_glu + S_glu ) ) - (  omega_max * (C_extra * B_1)**n_omega / ( k_omega_B_1**n_omega + (C_extra *  B_1)**n_omega )  )  * N_3 

    dS_glu = ( D * ( S0_glu - S_glu ) ) - ( mu_max_1 * S_glu / ( K_mu_glu + S_glu ) ) * N_1 * C_OD / g_1  - ( mu_max_2 * S_glu / ( K_mu_glu + S_glu ) ) * N_2 * C_OD / g_2  - ( mu_max_3 * S_glu / ( K_mu_glu + S_glu ) ) * N_3 * C_OD / g_3 

    dB_3 = ( - D * B_3 ) +  kB_max_3  * ( K_A_B_3**n_A_B_3 / ( K_A_B_3**n_A_B_3 + (C_extra * A_2)**n_A_B_3 ) ) * N_3 * C_OD / C_extra 

    dB_1 = ( - D * B_1 ) +  kB_max_1  * ( (C_extra * A_1)**n_A_B_1 / ( K_A_B_1**n_A_B_1 + (C_extra * A_1)**n_A_B_1 ) ) * N_1 * C_OD / C_extra  +  kB_max_1  * ( K_A_B_1**n_A_B_1 / ( K_A_B_1**n_A_B_1 + (C_extra * A_1)**n_A_B_1 ) ) * N_2 * C_OD / C_extra 

    dA_1 = ( - D * A_1 ) + kA_1 * N_1 * C_OD / C_extra + kA_1 * N_2 * C_OD / C_extra

    dA_2 = ( - D * A_2 ) + kA_2 * N_3 * C_OD / C_extra


    d_non_dim = [dN_1*C_OD, dN_2*C_OD, dN_3*C_OD, dS_glu, dB_1*C_extra, dB_3*C_extra, dA_2*C_extra, dA_1*C_extra]

    print(d_raw_list)
    print(d_non_dim)



def non_dim_odeint_test():
    def non_dim_diff_eqs(y, t):

        i = y[0]
        j = y[1]

        di = 3*i - 2*j - 10
        dj = 2*i - j + 7

        return [di, dj]

    def diff_eqs_exp(y, t):

        i = y[0]
        j = y[1]

        di = i + j
        dj = j + i

        return [di, dj]

    def non_dim_diff_eqs_exp(y, t):

        i = y[0]
        j = y[1]

        i_norm = 1000
        j_norm = 100


        di = i + j
        dj = j + i

        return [di, dj]


    i_init = 1
    j_init = 1

    y_init = [i_init, j_init]

    t = np.linspace(0, 1)

    sol = odeint(diff_eqs_exp, y_init, t)

    plt.plot(t, sol[:, 0])
    plt.plot(t, sol[:, 1])
    plt.show()


    i_norm = 1000
    j_norm = 100
    i_init = 1/i_norm
    j_init = 1/j_norm

    y_init = [i_init, j_init]

    t = np.linspace(0, 1)

    sol = odeint(non_dim_diff_eqs_exp, y_init, t)

    plt.plot(t, sol[:, 0]*i_norm)
    plt.plot(t, sol[:, 1]*j_norm)
    plt.show()


if __name__ == "__main__":
    non_dim_test_4()
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