import numpy as np
import population_modules as Population
import numpy as np
import matplotlib.pyplot as plt

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
    mu_max_1 = 2
    K_mu_glu = 5

    y = [3, 2, 9, 3, 1, 4, 7]

    x = 1.0/2.0*y[6]*y[2]*mu_max_1/(K_mu_glu + y[2]);

    I_1 = y[6]
    S_glu = y[2]

    z = I_1  * ( ( mu_max_1 * S_glu / ( K_mu_glu + S_glu ) ) )  / 2 

    print(x)
    print(z)

if __name__ == "__main__":
    test_this()
    # linear_repression_model()
    # main()