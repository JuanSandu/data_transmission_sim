'''
    Characteristics extractor:
    Calculate some characteristics, indicators and statistics from the selected
    signals in order to get conclusions from the data.

    This code has been developed by Juan Sandubete Lopez.
'''

import pywt
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import util_functions as utils
import sys
import json
import pylab
import stats_functions as stfc

class Statistics_extractor:
    def __init__(self, sim_id, general_config, config_dict, stage):
        # Do the things you need here for the simulation
        self.sim_id = sim_id
        self.stage = stage
        self.signals_names = general_config["signals_names"]
        self.module_name = "stats_extractor"
        self.stats_list = config_dict["stats_list"]
        self.stats_to_plot = config_dict["stats_to_plot"]
        self.sample_rate_s = general_config["sample_rate_s"]
        self.time_domain_stats = config_dict["time_domain_stats"]

    def plot_stat(self, stats_dict, sig_name, stat):
        if len(stats_dict[stat][sig_name]) > 1:
            save_path = "data/simulations/" + str(self.sim_id) + "/graphs/" \
                        + str(self.stage) + "_" + str(self.module_name) + "_" \
                        + stat + "_" + sig_name + ".png"
            # Estimate the x values (time or frequency)
            rate = 1 / self.sample_rate_s
            signal_len = len(stats_dict[stat][sig_name])
            if stat in self.time_domain_stats:
                x_values = np.arange(0, self.sample_rate_s * signal_len,
                                     self.sample_rate_s)
                xlabel = "Time (ms)"
            else:
                x_values = np.linspace(0, rate/2, signal_len)
                xlabel = "Frequency (Hz)"
            # To be sure about lenghts..
            x_values = x_values[0:signal_len]
            plt.plot(x_values, stats_dict[stat][sig_name])
            plt.title(stat)
            plt.xlabel(xlabel)
            plt.ylabel("Accel")  # TODO generalize this
            plt.grid()
            plt.savefig(save_path)
            plt.close()
        else:
            print("Stat {} of signal {} is not long enough to be plotted."
                  .format(stat, sig_name))

    def save_data(self, output_dict):
        for stat in self.stats_list:
            if hasattr(output_dict[stat][self.signals_names[0]], '__len__'):
                # Save the results
                # For each stat, create a csv with all the signals same stat data.
                print("Saving {} into file...".format(stat))
                # Save in CSV file only signals
                csv_name = self.module_name + "_" + stat
                utils.save_data_to_csv(self.sim_id, csv_name, self.stage,
                                       pd.DataFrame(output_dict[stat], columns=
                                       self.signals_names))
            else:
                print("Saving {} into file...".format(stat))
                utils.save_stats(self.sim_id, self.module_name, self.stage,
                                 stat, self.signals_names, output_dict[stat])

    def run(self, first_signals_df, second_signals_df):
        print("Statistics extraction started.")
        # output_df = pd.DataFrame()
        output_dict = {}
        for stat in self.stats_list:
            output_dict[stat] = {}

        for signal in self.signals_names:
            first_signal = first_signals_df[signal]
            second_signal = second_signals_df[signal]

            for stat in self.stats_list:
                if stat == "Correlation":
                    output_dict[stat][signal] = stfc.sigs_corr(first_signal,
                                                               second_signal)
                    print("{} max for signal {} the is at {}".format(stat,
                          signal, list(output_dict[stat][signal]).index(1.0)))

                elif stat == "Autocorrelation":
                    output_dict[stat][signal] = stfc.sig_autocorr(first_signal)
                    print("{} max for signal {} is at {}".format(stat, signal,
                          list(output_dict[stat][signal]).index(1.0)))

                elif stat == "Delay":
                    output_dict[stat][signal] = stfc.get_delay(first_signal,
                                                               second_signal)
                    print("{} for signal {} is {}".format(stat, signal,
                          output_dict[stat][signal]))

                elif stat == "MSE":
                    output_dict[stat][signal] = stfc.get_mse(first_signal,
                                                             second_signal)
                    print("For signal {} the MSE is: {}".format(signal,
                          output_dict[stat][signal]))

                elif stat == "PDS":
                    # Calculate the Power Density Spectrum (PDS)
                    output_dict[stat][signal] = stfc.get_pds(first_signal)
                    print("Power Density Spectrum calculated for signal {}"
                          .format(signal))

                elif stat == "PDScomparison":
                    # Calculate the difference between power frequency spectrum
                    # of the signal before and after the transmission
                    try:
                        pds_dif = np.subtract(output_dict["PDS"][signal],
                                    stfc.get_pds(first_signal))
                    except:
                        pds_dif = np.subtract(stfc.get_pds(second_signal),
                                    stfc.get_pds(first_signal))
                    output_dict[stat][signal] = pds_dif
                    print("PDS difference calculated for signal {}"
                          .format(signal))

                else:
                    print("Unknown requireed KPI.")

                # Plot the signal if necessary
                if (stat in self.stats_to_plot) and (stat in self.stats_list):
                    self.plot_stat(output_dict, signal, stat)

        # Save the data
        self.save_data(output_dict)

        print("Statistics extraction process finished.")


def launch_module():
    # ------ CLASS LAUNCHER ------
    # Get the arguments
    sim_id = sys.argv[1]  # Simulation Id
    stage = sys.argv[2]  # Stage of the simulation
    first_signals_file = sys.argv[3]  # Name of the first file
    second_signals_file = sys.argv[4]  # Name of the second file

    # Load the configuration file
    with open("sim_configuration.json", "r") as config_file:
        config_dict = json.loads(config_file.read())
    original_signals_df_name = config_dict["general"]["signals_df_name"]

    # Get the appropriated signal
    try:
        first_signals_df = utils.get_signals_dataframe(sim_id,
                                                       first_signals_file)
    except:
        first_signals_df = utils.get_original_dataframe(first_signals_file)
    second_signals_df = utils.get_signals_dataframe(sim_id, second_signals_file)

    # Run the rest of the class stuff
    stats_extrc = Statistics_extractor(sim_id, config_dict["general"],
                                       config_dict["stats_extractor"], stage)
    stats_extrc.run(first_signals_df, second_signals_df)

    print("Closing statistics extraction process.")


launch_module()
