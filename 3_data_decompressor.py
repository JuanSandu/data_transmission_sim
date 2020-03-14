'''
    Data decompressor:
    Get the original signal from the compressed one.
'''

import pywt
import pandas as pd
import numpy as np
import util_functions as utils
import sys
import json

class Data_Decompressor:
    def __init__(self, sim_id, signals_names, config_dict,
                 original_signal_filename):
        # Do the things you need here for the simulation
        self.sim_id = sim_id
        self.signals_names = signals_names
        self.module_name = "data_decompressor"
        self.wv_sel = config_dict["wavelet_family"]
        self.wv_order = config_dict["compression_level"]
        self.original_signal_df = \
                utils.get_original_dataframe(original_signal_filename)
        self.original_signal_len = len(self.original_signal_df)
        print("Original signal lenght: {}".format(self.original_signal_len))

    def wv_signal_decomprs(self, signal_df):
        """
        Make a compression of the input signals with a discrete wavelet
        transform.
        """
        signal_wv_list = []
        # and decompress data effectively. take=self.original_signal_len
        for signal_name in self.signals_names:
            temp_wv_list = pywt.upcoef('a', signal_df[signal_name], self.wv_sel,
                                       level=self.wv_order).tolist()
            signal_wv_list.append(temp_wv_list)
            print("Signal length after decompression: {}"
                  .format(len(signal_wv_list[0])))
        # Create and return the output dataframe with the compressed signals
        output_df = pd.DataFrame()
        for i, signal in enumerate(self.signals_names):
            output_df[signal] = \
                        self.correct_deco_len(self.original_signal_df[signal],
                                              signal_wv_list[i])
        return output_df

    def correct_deco_len(self, first_signal, second_signal):
        """
        Dismiss the elements of the signal which have been created only because
        of the decompression process in the edges.
        Important: Second signal must have a displacement in time greater or
                   equal to zero for this to work properly.
        """
        corr = np.correlate(first_signal, second_signal, "same")
        autocorr = np.correlate(first_signal, first_signal, "same")
        delay = list(corr/max(corr)).index(1.0) - \
                list(autocorr/max(autocorr)).index(1.0)
        last_index = len(second_signal) - delay - len(first_signal)
        if delay <= 0:
            print("ERROR: Calculated delay is lower than zero.")
            dif = len(second_signal) - len(first_signal)
            return (second_signal[int(dif/2):-int(dif/2)])
        else:
            return (second_signal[delay:-last_index])

    def run(self, original_signal_df, stage_str):
        print("Data decompression started.")
        # Apply the noise and other effects over the original signals
        output_signal_df = self.wv_signal_decomprs(original_signal_df)

        # Save the results
        utils.save_data_to_csv(self.sim_id, self.module_name, stage_str,
                               output_signal_df)
        print("Data decompression process finished.")


def launch_module():
    # ------ CLASS LAUNCHER ------
    # Get the arguments
    sim_id = sys.argv[1]
    stage = sys.argv[2]

    # Load the configuration file
    with open("sim_configuration.json", "r") as config_file:
        config_dict = json.loads(config_file.read())
    original_signals_df_name = config_dict["general"]["signals_df_name"]

    # Get the appropriated signal
    if int(stage) == 0:
        signals_df = utils.get_original_dataframe(original_signals_df_name)
    else:
        filename = utils.get_last_stage_filename(sim_id)
        signals_df = utils.get_signals_dataframe(sim_id, filename)

    # Run the rest of the class stuff
    data_decompressor = Data_Decompressor(sim_id, config_dict["general"]
                                          ["signals_names"],
                                          config_dict["data_compressor"],
                                          original_signals_df_name)
    data_decompressor.run(signals_df, str(stage))

    if config_dict["data_compressor"]["plot_signals"]:
        print("Plotting signal...")
        signals_filename = utils.get_last_stage_filename(sim_id)
        for signal_name in config_dict["general"]["signals_names"]:
            utils.plot_signal_from_file(sim_id, signals_filename,
                                        signal_name, stage,
                                        data_decompressor.module_name)
    print("Closing data decompression process.")


launch_module()
