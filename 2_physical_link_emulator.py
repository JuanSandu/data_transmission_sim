'''
    Physical link emulator:
    Apply some physical effects like atenuation, random gaps and others, to a
    loaded signal (dataframe from a CSV file) and creates an output CSV file
    with the processed signal.
    This code has been developed by Juan Sandubete Lopez.
'''

import pandas as pd
import sys
import util_functions as utils
import json
import numpy as np
import math
import matplotlib.pyplot as plt

class Physical_Link_Emulator:
    def __init__(self, sim_id, signals_names, config):
        # Do the things you need here for the simulation
        self.sim_id = sim_id
        self.module_name = "physical_link_emulator"
        self.signals_names = signals_names
        self.medium = config["medium"]
        self.signal_freq = config["sig_transm_hz"]
        self.signal_resolution_s = config["signal_resolution_s"]
        self.wavelength_m = config["wavelength_m"]
        self.sig_space_perctg = config["sig_space_perctg"]
        self.empty_thld = config["empty_thld"]
        self.high_thld = config["high_thld"]
        self.empty_level_volts = config["empty_level_volts"]
        self.high_level_volts = config["high_level_volts"]
        self.low_level_volts = config["low_level_volts"]
        self.max_integer_part = config["max_integer_part"]
        self.max_decimals = config["max_decimals"]
        self.signal_period = 1.0/self.signal_freq
        self.signal_speed = self.signal_freq * self.wavelength_m
        self.smpls_per_bit = int(round(self.signal_period /
                                   self.signal_resolution_s))
        self.ro = config["water"]["ro"]
        self.distance = config["water"]["distance"]

    def digital_to_analog(self, signals_df):
        """
        Take the digital input signal and make it become an analog one. For the
        moment, output signal consists only of a rectangular signal meaning ones
        and zeros.
        """
        output_sig_df = pd.DataFrame()
        prev_bit = 0 # bin(0).replace('0b', '')
        for signal in self.signals_names:
            output_sig = []
            print_process = False
            for sig_value in signals_df[signal]:
                split_float = self.float_to_int_couple(sig_value)
                # print(split_float)
                for value in split_float:
                    output_sig += self.insert_sig_empty()
                    binary_value = self.int_to_bin(value)
                    for sig_bit in binary_value:
                        # TODO I think the bit logic can be done more efficiently
                        if (int(sig_bit) == 1) and (int(sig_bit) - int(prev_bit)) == 0:
                            # Continuous high level
                            if print_process:
                                print("High: {}, {}".format(int(sig_bit), int(prev_bit)))
                            output_sig += self.insert_high_level()
                        elif (int(sig_bit) == 0) and (int(sig_bit) - int(prev_bit)) == 0:
                            # Continuous low level
                            if print_process:
                                print("Low: {}, {}".format(int(sig_bit), int(prev_bit)))
                            output_sig += self.insert_low_level()
                        elif (int(sig_bit) - int(prev_bit)) == -1:
                            if print_process:
                                print("High2Low: {}, {}".format(int(sig_bit), int(prev_bit)))
                            # High to low transition
                            output_sig += self.insert_high_to_low()
                        elif (int(sig_bit) - int(prev_bit)) == 1:
                            if print_process:
                                print("Low2High: {}, {}".format(int(sig_bit), int(prev_bit)))
                            # Low to high transition
                            output_sig += self.insert_low_to_high()
                        else:
                            print("Error in digital to analog transf. Non-expected"+
                                  "combination")
                        prev_bit = sig_bit
                    if print_process:
                        print_process = False
            # Closing empty values
            output_sig += self.insert_sig_empty()

            # Save the output analog signal into the dataframe
            output_sig_df[signal] = output_sig
            output_sig = []
        return output_sig_df

    def float_to_int_couple(self, sig_value):
        # TODO parametrize the number of decimals
        sig_value = '%.4f' % float(sig_value)  # Del some extra decimals
        split_float = str(sig_value).split('.')
        split_float[0] = str(min(int(split_float[0]), self.max_integer_part))
        if len(split_float[0]) < 4:
            split_float[0] = ''.join(['0'] * (4 - len(split_float[0]))) + \
                                split_float[0]
        return split_float

    def int_to_bin(self, value):
        binary = bin(int(value)).replace('0b', '')
        if len(binary) < 16:
            binary = ''.join(['0'] * (16 - len(binary))) + binary
        # print(binary)
        return binary

    def insert_sig_empty(self):
        time_s = self.signal_period * self.sig_space_perctg / 100.0
        samples = max(int(time_s / self.signal_resolution_s), 2)
        return [self.empty_level_volts] * samples

    def insert_high_level(self):
        return [self.high_level_volts] * self.smpls_per_bit

    def insert_low_level(self):
        return [self.low_level_volts] * self.smpls_per_bit

    def insert_high_to_low(self):
        output_sig = [self.low_level_volts] * self.smpls_per_bit
        n_transt_smpls = int(round(self.smpls_per_bit*0.05))
        if self.smpls_per_bit > n_transt_smpls:
            output_sig[0:n_transt_smpls] = \
                list(np.linspace(self.high_level_volts, self.low_level_volts,
                                 n_transt_smpls))
        return output_sig

    def insert_low_to_high(self):
        output_sig = [self.high_level_volts] * self.smpls_per_bit
        n_transt_smpls = int(round(self.smpls_per_bit*0.05))
        if self.smpls_per_bit > n_transt_smpls:
            output_sig[0:n_transt_smpls] = \
                list(np.linspace(self.low_level_volts, self.high_level_volts,
                                 n_transt_smpls))
        return output_sig

    def analog_to_digital(self, signals_df):
        """
        Take the "analogized" signal and extract the digital data from it. For
        the input signal, it separates data in bytes (checking where the empty
        spaces are) and tries to get the digital data from it.
        """
        output_sig_df = pd.DataFrame()
        for signal in self.signals_names:
            output_sig = []
            sig_raw_byte = []
            # Extract the bytes mask. True if value belongs to no-info data.
            bytes_mask = signals_df[signal] < self.empty_thld
            prev_smpl_was_empty = True
            data_in_smpl = False
            print_process = False
            for i, is_empty_smpl in enumerate(bytes_mask):
                # Byte event catcher logic
                # I guess it could be done tons more robust, but ...
                if prev_smpl_was_empty and not is_empty_smpl:
                    # Byte beginning
                    if print_process:
                        print("Byte begins: {}".format(i))
                    data_in_smpl = True
                elif not prev_smpl_was_empty and is_empty_smpl:
                    # Byte ending or (i == len(bytes_mask)-1)
                    if len(sig_raw_byte) > 2:
                        output_sig.append(
                            self.get_digtl_sig_from_raw(sig_raw_byte, print_process))
                    if print_process:
                        print("Byte ends: {}".format(i))
                        print(self.get_digtl_sig_from_raw(sig_raw_byte, False))
                        print_process = False
                    data_in_smpl = False
                    sig_raw_byte = []  # Clean the raw data buffer

                # Take the data into the buffer
                if data_in_smpl:
                    sig_raw_byte.append(signals_df[signal][i])
                # Update the byte detector bool
                prev_smpl_was_empty = is_empty_smpl
            # Save the output analog signal into the dataframe
            output_sig = self.int_couple_to_float(output_sig)
            output_sig_df[signal] = output_sig
            output_sig = []
        return output_sig_df

    def get_digtl_sig_from_raw(self, sig_raw_byte, print_process):
        raw_smpls = len(sig_raw_byte)
        if print_process:
            print("Length: {}".format(raw_smpls))
        byte_str = str()
        for i in range(int(math.floor(len(sig_raw_byte)/self.smpls_per_bit))):
            try:
                raw_bit = \
                    sig_raw_byte[i*self.smpls_per_bit:(i+1)*self.smpls_per_bit]
            except:
                if (i+1)*self.smpls_per_bit > len(sig_raw_byte):
                    raw_bit = sig_raw_byte[i*self.smpls_per_bit:]
                else:
                    print("Error extracting the bit. Length missmatch.")
            if np.mean(raw_bit) > self.high_thld:
                byte_str = "{}{}".format(byte_str, '1')
            else:
                byte_str = "{}{}".format(byte_str, '0')
            if print_process:
                print("Volts: {}".format(np.mean(raw_bit)))
                print("Array: {}".format(raw_bit))
        # print(byte_str)
        return int(byte_str, 2)

    def int_couple_to_float(self, signal):
        output_sig = []
        for i in range(int(len(signal)/2)):
            output_sig.append(signal[i*2] + (signal[i*2+1] / 10000.0))
        #print(output_sig)
        return output_sig

    def apply_water_effects_transmission(self, signals_df):
        # First approach to water transmission
        atenuation = min(1.0, self.ro * self.distance)
        for signal in self.signals_names:
            signals_df[signal] = map(lambda x: x * (1 - atenuation),
                                     signals_df[signal])
        return signals_df

    def apply_air_effects_transmission(self, original_signal_df, signals_names,
                                       distance, temperature):
        return original_signal_df

    def apply_space_effects_transmission(self, original_signal_df,
                                         signals_names, distance, temperature):
        return original_signal_df

    def run(self, signals_df, stage_str, mode):
        print("Transmission physics simulation started.")
        # Apply the noise and other effects over the original signals
        if mode == "dig2an":
            print("Digital signal to analog...")
            output_sig_df = self.digital_to_analog(signals_df)
            if self.medium == "water":
                output_sig_df = self.apply_water_effects_transmission(output_sig_df)
        elif mode == "an2dig":
            print("Analog signal to digital...")
            output_sig_df = self.analog_to_digital(signals_df)

        # Save the results
        utils.save_data_to_csv(self.sim_id, self.module_name, stage_str,
                               output_sig_df)
        print("Transmission physics simulation finished.")


def launch_module():
    # ------ CLASS LAUNCHER ------
    # Get the arguments
    sim_id = sys.argv[1]
    stage = sys.argv[2]
    mode = sys.argv[3]

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
    physics_sim = Physical_Link_Emulator(sim_id, config_dict["general"]
                                         ["signals_names"],
                                         config_dict["physical_link_emulator"])
    physics_sim.run(signals_df, str(stage), mode)

    if config_dict["physical_link_emulator"]["plot_signals"]:
        print("Plotting signal...")
        signals_filename = utils.get_last_stage_filename(sim_id)
        for signal_name in config_dict["general"]["signals_names"]:
            utils.plot_signal_from_file(sim_id, signals_filename,
                                        signal_name, stage,
                                        physics_sim.module_name)
    print("Closing data compression process.")


launch_module()
