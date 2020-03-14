"""
Script for plotting the data. The plotted points has been manually extracted
from the simulation stats files. This should be automated in a near future.
"""
import matplotlib.pyplot as plt
import math

def plot_ratio_versus_mse():
    plt.close()

    db3_bytes = [1979199, 984154, 476231, 232624, 119857]
    db3_ratio = [3913906.0/bytes for bytes in db3_bytes]
    db3_mse = [135.3038, 138.25817, 226.73653, 378.79831, 22082.098355]
    db3_mse_log = [math.log10(mse) for mse in db3_mse]
    plt.plot(db3_ratio, db3_mse_log, 'k--d', label='DB3 wavelet')

    haar_bytes = [1978558, 893775, 476149, 232244, 119795]
    haar_ratio = [3913906.0/bytes for bytes in haar_bytes]
    haar_mse = [41.97500, 123.8325830, 384.3810070, 558.2101109, 22083.299518]
    haar_mse_log = [math.log10(mse) for mse in haar_mse]
    plt.plot(haar_ratio, haar_mse_log, 'k:d', label='Haar wavelet')

    sym4_bytes = [1979143, 984227, 476370, 232739, 119874]
    sym4_ratio = [3913906.0/bytes for bytes in sym4_bytes]
    sym4_mse = [133.635628, 134.68086, 224.993069, 371.410950, 22539.549]
    sym4_mse_log = [math.log10(mse) for mse in sym4_mse]
    plt.plot(sym4_ratio, sym4_mse_log, 'k-d', label='Sym4 wavelet')

    legend = plt.legend(loc='upper center', shadow=True)

    plt.xlabel('Compression ratio (n.u.)')
    plt.ylabel('MSE Log10 (accel. raw units)')
    plt.title('Compression Analysis')
    plt.grid()
    plt.savefig('wavelet_compression_ratio_log_mse.png')


def plot_mse_versus_white_noise():
    plt.close()

    # n_mean = 0.0
    # n_dev = 0.15, 0.25, 0.35
    # gauss_noise = np.random.normal(n_mean, n_dev, data_len)
    # noise_power = np.sum(np.square(gauss_noise))
    white_noise_power = [3652.45664, 10227.6732099, 19891.190353]
    #white_noise_power = [format(power, "1.2E") for power in white_noise_power]

    bior1_3_mse = [270.0524425, 3681.5748, 898111.5114]
    bior1_3_mse_log = [math.log10(mse) for mse in bior1_3_mse]
    plt.plot(white_noise_power, bior1_3_mse_log, 'k--d', label='BiOr1.3')

    haar_mse = [384.381007, 7516.31693, 948304.224143]
    haar_mse_log = [math.log10(mse) for mse in haar_mse]
    plt.plot(white_noise_power, haar_mse_log, 'k:d', label='Haar')

    db3_mse = [226.7365, 13985.2452214, 1115247.784]
    db3_mse_log = [math.log10(mse) for mse in db3_mse]
    plt.plot(white_noise_power, db3_mse_log, 'k-d', label='Db3')

    rbio2_2_mse = [282.3072719, 4395.8359, 1215528.566568]
    rbio2_2_mse_log = [math.log10(mse) for mse in rbio2_2_mse]
    plt.plot(white_noise_power, rbio2_2_mse_log, 'b--d', label='RBiOr2.2')

    sym4_mse = [224.99306, 641.740627, 326215.0037]
    sym4_mse_log = [math.log10(mse) for mse in sym4_mse]
    plt.plot(white_noise_power, sym4_mse_log, 'r:d', label='Sym4')

    legend = plt.legend(loc='upper left', shadow=True)

    plt.xlabel('Noise signal power (V*V)')
    plt.ylabel('MSE Log10 (accel. raw units)')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.title('White Noise Vs Wavelet Decomposition')
    plt.grid()
    plt.savefig('white_noise_vs_wav_decomp.png')


def plot_mse_versus_pink_noise():
    plt.close()

    # n_mean = 0.0
    # n_dev = 0.15, 0.25, 0.35
    # state = np.random.RandomState()
    # uneven = N % 2
    # uneven = data_len % 2
    # N = data_len
    # X = state.randn(N // 2 + 1 + uneven) + 1j * state.randn(N // 2 + 1 + uneven)
    # S = np.sqrt(np.arange(len(X)) + 1.)
    # y = (np.fft.irfft(X / S)).real
    # if uneven:
    #     y = y[:-1]
    # pink_noise = np.add((y / max(y) * (1 + n_dev)), n_mean)
    # pink_power = np.sum(np.square(pink_noise))

    pink_noise_power = [13275.95523, 15685.20230, 18295.219968]
    #pink_noise_power = [format(power, "1.2E") for power in pink_noise_power]

    bior1_3_mse = [270.05244258, 635.016655, 45101.0914384]
    bior1_3_mse_log = [math.log10(mse) for mse in bior1_3_mse]
    plt.plot(pink_noise_power, bior1_3_mse_log, 'k--d', label='BiOr1.3')

    haar_mse = [384.3810030, 88968.934166, 566807.71666]
    haar_mse_log = [math.log10(mse) for mse in haar_mse]
    plt.plot(pink_noise_power, haar_mse_log, 'k:d', label='Haar')

    db3_mse = [6840.9296, 27035.46244, 19858.054628]
    db3_mse_log = [math.log10(mse) for mse in db3_mse]
    plt.plot(pink_noise_power, db3_mse_log, 'k-d', label='Db3')

    rbio2_2_mse = [1921.644648, 59571.10770, 33276.65120]
    rbio2_2_mse_log = [math.log10(mse) for mse in rbio2_2_mse]
    plt.plot(pink_noise_power, rbio2_2_mse_log, 'b--d', label='RBiOr2.2')

    sym4_mse = [6986.9234300, 16470.96655, 72354.86996]
    sym4_mse_log = [math.log10(mse) for mse in sym4_mse]
    plt.plot(pink_noise_power, sym4_mse_log, 'r:d', label='Sym4')

    legend = plt.legend(loc='upper left', shadow=True)

    plt.xlabel('Noise signal power (V*V)')
    plt.ylabel('MSE Log10 (accel. raw units)')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.title('Pink Noise Vs Wavelet Decomposition')
    plt.grid()
    plt.savefig('pink_noise_vs_wav_decomp.png')


def plot_mse_versus_brown_noise():
    plt.close()

    # n_mean = 0.0
    # n_dev = 0.15, 0.25, 0.35
    # state = np.random.RandomState()  # if state is None else state
    # uneven = N % 2
    # X = state.randn(N // 2 + 1 + uneven) + 1j * state.randn(N // 2 + 1 + uneven)
    # S = (np.arange(len(X)) + 1)  # Filter
    # y = (np.fft.irfft(X / S)).real
    # if uneven:
    #     y = y[:-1]
    # brown_noise = np.add((y / max(y) * n_dev), n_mean)
    # brown_power = np.sum(np.square(brown_noise))

    brown_noise_power = [7973.436719, 9489.0486581, 11136.4529390]
    #pink_noise_power = [format(power, "1.2E") for power in brown_noise_power]

    bior1_3_mse = [25565.26892, 72050.1661, 27711.7207833]
    bior1_3_mse_log = [math.log10(mse) for mse in bior1_3_mse]
    plt.plot(brown_noise_power, bior1_3_mse_log, 'k--d', label='BiOr1.3')

    rbio2_2_mse = [282.3072719, 324689.12853, 88390.87261]
    rbio2_2_mse_log = [math.log10(mse) for mse in rbio2_2_mse]
    plt.plot(brown_noise_power, rbio2_2_mse_log, 'b--d', label='RBiOr2.2')

    sym4_mse = [224.993069, 224.993069, 224.993069]
    sym4_mse_log = [math.log10(mse) for mse in sym4_mse]
    plt.plot(brown_noise_power, sym4_mse_log, 'r:d', label='Sym4')

    legend = plt.legend(loc='upper left', shadow=True)

    plt.xlabel('Noise signal power (V*V)')
    plt.ylabel('MSE Log10 (accel. raw units)')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.title('Brown Noise Vs Wavelet Decomposition')
    plt.grid()
    plt.savefig('brown_noise_vs_wav_decomp.png')

plot_mse_versus_brown_noise()
