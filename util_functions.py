"""
    Set of auxiliary functions for the signal transmission simulator.
    This code has been developed by Juan Sandubete Lopez.
"""
import pandas as pd
import os
from os import listdir, walk
from os.path import isfile, join
import matplotlib.pyplot as plt

def get_original_dataframe(dataset_name, reset_indx=True):
    """
    Given a dataset_name string, returns the corresponding pandas dataframe.
    """
    if os.path.splitext(dataset_name)[1] != ".csv":
        path = "data/original_data/" + dataset_name + ".csv"
    else:
        path = "data/original_data/" + dataset_name
    signals_df = pd.read_csv(path, index_col=[0])
    if reset_indx:
        return signals_df.reset_index()
    else:
        return signals_df


def get_signals_dataframe(sim_id, dataset_name, reset_indx=True):
    """
    Given a dataset_name string, returns the corresponding pandas dataframe.
    """
    if os.path.splitext(dataset_name)[1] != ".csv":
        path = "data/simulations/" + sim_id + "/signals/" + dataset_name \
               + ".csv"
    else:
        path = "data/simulations/" + sim_id + "/signals/" + dataset_name

    # Load the signal
    signals_df = pd.read_csv(path, index_col=[0])
    if reset_indx:
        return signals_df.reset_index(drop=True)
    else:
        return signals_df


def get_last_stage_filename(sim_id):
    """
    Returns the name of the file with the bigger associated stage number in its
    name.
    """
    path = "data/simulations/" + sim_id + "/signals"
    path = os.path.abspath(path)  # This is not strictly necessary
    filenames_list = []
    for (dirpath, dirnames, filenames) in walk(path):
        filenames_list.extend(filenames)
        break

    filename = filenames_list[0]
    for each_name in filenames_list:
        if int(each_name[0]) > int(filename[0]):
            filename = each_name
    return filename


def get_original_signal_len(dataset_name):
    path = "data/original_data/" + dataset_name + ".csv"
    signals_df = pd.read_csv(path, index_col=[0])
    return len(signals_df)

def save_data_to_csv(sim_id, origin_script, stage_str, signals_df):
    path_and_filename = ("data/simulations/" + str(sim_id) + "/signals/" \
                         + stage_str + "_" + origin_script + ".csv")
    signals_df.to_csv(path_and_filename)


def save_stats(sim_id, origin_script, stage_str, stat, signals, stat_dict):
    path_and_filename = ("data/simulations/" + str(sim_id) + "/signals/" \
                         + stage_str + "_stats" + ".csv")
    if not os.path.isfile(path_and_filename):
        header = "stat"
        for signal in signals:
            header += "," + signal
        with open(path_and_filename, "a") as output_file:
            output_file.write(header + "\n")
    data_out = stat
    for signal in signals:
        data_out += "," + str(stat_dict[signal])
    with open(path_and_filename, "a") as output_file:
        output_file.write(data_out)
        output_file.write("\n")

def plot_signal_from_file(sim_id, dataset_name, signal_name, stage, block_name):
    if os.path.splitext(dataset_name)[1] != ".csv":
        dataset_name = dataset_name + ".csv"
    path = "data/simulations/" + str(sim_id) + "/signals/" \
           + dataset_name
    save_path = "data/simulations/" + str(sim_id) + "/graphs/" \
                + str(stage) + "_" + str(block_name) + "_" \
                + signal_name + ".png"
    signal_df = pd.read_csv(path, index_col=[0])

    signal_df[signal_name].plot(x='Time (ms)',y='Accel',color='blue')
    plt.savefig(save_path)
