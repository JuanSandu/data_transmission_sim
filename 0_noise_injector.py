'''
    Noise Injector:
    Injects several types of random noise and other disturbant effects to a
    given signal (loaded from a CSV file to a pandas dataframe).
    The applied effects can be selected via script arguments.
    This code has been developed by Juan Sandubete Lopez.
'''

import numpy as np
import pandas as pd
import util_functions as utils
from sklearn.preprocessing import normalize
import sys
import json
import matplotlib.pyplot as plt

class Noise_Injector:
    def __init__(self, sim_id, signals_names, config_dict):
        # Do the things you need here for the simulation
        self.sim_id = sim_id
        self.module_name = "noise_injector"
        self.signals_names = signals_names
        self.noise_list = config_dict["noise_sources"]
        self.white_noise = config_dict["white_noise"]
        self.brown_noise = config_dict["brown_noise"]
        self.pink_noise = config_dict["pink_noise"]

    def inject_white_noise(self, signal):
        gauss_noise = np.random.normal(self.white_noise["mean"],
                                       self.white_noise["dev"], len(signal))
        noisy_signal = np.add(signal, gauss_noise)
        return noisy_signal

    def inject_brown_noise(self, signal):
        # Following code has been adapted from:
        # https://github.com/python-acoustics
        # Create the signal curve in frequency domain and transform it to time
        # domain again. Add it to the input signal.
        N = len(signal)
        state = np.random.RandomState()  # if state is None else state
        uneven = N % 2
        X = state.randn(N // 2 + 1 + uneven) + 1j * state.randn(N // 2 + 1 + uneven)
        S = (np.arange(len(X)) + 1)  # Filter
        y = (np.fft.irfft(X / S)).real
        if uneven:
            y = y[:-1]
        brown_noise = np.add((y / max(y) * self.brown_noise["dev"]),
                              self.brown_noise["mean"])
        noisy_signal = np.add(signal, brown_noise)
        return noisy_signal

    def inject_pink_noise(self, signal):
        # Following code has been adapted from:
        # https://github.com/python-acoustics
        # To sum up, it creates a PDS curve and applies iRFFT to it.
        N = len(signal)
        state = np.random.RandomState()  #  if state is None else state
        uneven = N % 2
        X = state.randn(N // 2 + 1 + uneven) + 1j * state.randn(N // 2 + 1 + uneven)
        S = np.sqrt(np.arange(len(X)) + 1.)  # +1 to avoid divide by zero
        y = (np.fft.irfft(X / S)).real
        if uneven:
            y = y[:-1]
        pink_noise = np.add((y / max(y) * (1 + self.pink_noise["dev"])),
                            self.pink_noise["mean"])
        noisy_signal = np.add(signal, pink_noise)
        return noisy_signal

    def apply_echo_effect(self):
        pass

    def apply_noise_and_effects(self, signals_df, signals_names):
        # Output dataframe is generated as a modifies copy of the original one
        output_signal_df = signals_df.copy()

        # Apply noise over all selected signals
        for noise in self.noise_list:
            for signal in signals_names:
                print("Injection of {} in signal {}".format(noise, signal))
                if noise == "white_noise":
                    output_signal_df[signal] = \
                        self.inject_white_noise(signals_df[signal].tolist())
                elif noise == "brown_noise":
                    output_signal_df[signal] = \
                        self.inject_brown_noise(signals_df[signal].tolist())
                elif noise == "pink_noise":
                    output_signal_df[signal] = \
                        self.inject_pink_noise(signals_df[signal].tolist())

        return output_signal_df

    def run(self, signals_df, stage_str):
        print("Noise injection process started.")
        # Apply the noise and other effects over the original signals
        output_signal_df = self.apply_noise_and_effects(signals_df,
                                                        self.signals_names)
        # Save the results
        utils.save_data_to_csv(self.sim_id, self.module_name, stage_str,
                               output_signal_df)
        print("Noise injection process finished.")


def launch_module():
    # ------ CLASS LAUNCHER ------
    # Get the arguments
    sim_id = sys.argv[1]
    stage = sys.argv[2]

    # Load the configuration file
    with open("sim_configuration.json", "r") as config_file:
        config_dict = json.loads(config_file.read())
    orignal_signals_df_name = config_dict["general"]["signals_df_name"]

    # Get the appropriated signal
    if int(stage) == 0:
        signals_df = utils.get_original_dataframe(orignal_signals_df_name)
    else:
        filename = utils.get_last_stage_filename(sim_id)
        signals_df = utils.get_signals_dataframe(sim_id, filename, True)

    # Run the rest of the class stuff
    noise_injector = Noise_Injector(sim_id, config_dict["general"]
                                    ["signals_names"],
                                    config_dict["noise_injector"])
    noise_injector.run(signals_df, str(stage))

    if config_dict["noise_injector"]["plot_signals"]:
        print("Plotting signals...")
        signals_filename = utils.get_last_stage_filename(sim_id)
        for signal_name in config_dict["general"]["signals_names"]:
            utils.plot_signal_from_file(sim_id, signals_filename,
                                        signal_name, stage,
                                        noise_injector.module_name)
    print("Closing noise injection process.")


launch_module()
