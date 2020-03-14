"""
    Set of stats calculation utilities functions.
    This code has been developed by Juan Sandubete Lopez.
"""

import numpy as np

def sigs_corr(first_signal, second_signal):
    """
    Get the correlation between the specified signals.
    """
    sig_corr = np.correlate(first_signal, second_signal, "same")
    return sig_corr/max(sig_corr)

def sig_autocorr(signal):
    """
    Get the auto-correlation for the specified signal.
    """
    sig_corr = np.correlate(signal, signal, "same")
    return sig_corr/max(sig_corr)

def get_delay(first_signal, second_signal):
    """
    Get an approximation of the delay value of the same signal.
    """
    corr = list(sigs_corr(first_signal, second_signal)).index(1.0)
    autocorr = list(sig_autocorr(first_signal)).index(1.0)
    return corr - autocorr

def correct_deco_len(signal, delay, orig_len):
    last_index = len(signal) - delay - orig_len
    return (signal[delay:-last_index]).reset_index(drop=True)

def get_mse(first_signal, second_signal):
    """
    Determine the Mean Square Error of the two signals.
    """
    if len(first_signal) == len(second_signal):
        mse = np.mean(np.square(np.subtract(first_signal, second_signal)))
    else:
        delay = get_delay(first_signal, second_signal)
        sig_cor = correct_deco_len(second_signal, delay,
                                        len(first_signal))
        mse = np.mean(np.square(np.subtract(first_signal, sig_cor)).values)
    return mse

def get_pds(signal):
    power = 20*np.log10(np.abs(np.fft.rfft(signal)))
    return power

def get_sig_power(signal):
    return np.sum(np.square(signal))

def get_values_over_thld(signal, thld):
    cont = 0.0
    for value in signal:
        if value > thld:
            cont += 1.0
    return cont/len(signal)

def get_values_under_thld(signal, thld):
    cont = 0.0
    for value in signal:
        if value < thld:
            cont += 1.0
    return cont/len(signal)
