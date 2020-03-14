'''
    Data compressor:
    Apply the wavelet transform to a given signal (dataframe from a given CSV
    file) to an specified level of precission and creates an output CSV file
    with the processed signal.
    This code has been developed by Juan Sandubete Lopez.
'''

import pywt
import pandas as pd
import util_functions as utils
import sys
import json

class Data_Compressor:
    def __init__(self, sim_id, signals_names, config_dict):
        # Do the things you need here for the simulation
        self.sim_id = sim_id
        self.signals_names = signals_names
        self.module_name = "data_compressor"
        self.wv_sel = config_dict["wavelet_family"]
        self.wv_order = config_dict["compression_level"]

    def wv_signal_comprs(self, original_signal_df, signals_names):
        """
        Make a compression of the input signals with a discrete wavelet
        transform.
        """
        # TODO It is still necessary to see how to use it in order to compress
        signal_wv_list = []
        # and decompress data effectively.
        for signal_name in signals_names:
            print("Compression of signal {} with wavelet {}"
                  .format(signal_name, self.wv_sel))
            signal_wv_list.append(pywt.downcoef("a",
                                  original_signal_df[signal_name], self.wv_sel,
                                  level=self.wv_order).tolist())
        # Create and return the output dataframe with the compressed signals
        output_df = pd.DataFrame()
        for i, signal in enumerate(signals_names):
            output_df[signal] = signal_wv_list[i]
        return output_df

    def wv_object_creation(self, original_signal_df, signals_names):
        wv_objct = pywt.Wavelet(self.wv_sel)

        for signal_name in signals_names:
            signal_wv_list, _, _, _ = pywt.wavedec(original_signal_df
                                                    [signal_name], wv_objct)
            signal_wv_list.append(signal_wv_list)
        # Create and return the output dataframe with the compressed signals
        return pd.DataFrame(signal_wv_list, columns = signals_names)

    def run(self, original_signal_df, stage_str):
        print("Data compression started.")
        # Apply the noise and other effects over the original signals
        output_signal_df = self.wv_signal_comprs(original_signal_df,
                                                 self.signals_names)
        # Save the results
        utils.save_data_to_csv(self.sim_id, self.module_name, stage_str,
                               output_signal_df)
        print("Data compression process finished.")


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
        signals_df = utils.get_signals_dataframe(sim_id, filename)

    # Run the rest of the class stuff
    data_compressor = Data_Compressor(sim_id, config_dict["general"]
                                      ["signals_names"],
                                      config_dict["data_compressor"])
    data_compressor.run(signals_df, str(stage))

    if config_dict["data_compressor"]["plot_signals"]:
        print("Plotting signal...")
        signals_filename = utils.get_last_stage_filename(sim_id)
        for signal_name in config_dict["general"]["signals_names"]:
            utils.plot_signal_from_file(sim_id, signals_filename,
                                        signal_name, stage,
                                        data_compressor.module_name)
    print("Closing data compression process.")


launch_module()
