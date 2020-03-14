#!/bin/bash

# Ensure you launch this script from the signal_and_mining_project path !

# Modes: run or train.
mode="run"  # "train"

# List of possible actions to be applyied to the signals:
# 0: Inject noise
# 1: Compress data
# 2: Emulate physical transmission
# 3: Decompress data
# 4: Extraction of statistics and KPIs
# 5: Classify signal noise
#
# For example, a normal one way cycle would be: (0 1 2 3 4 5)
actions=(1 2 0 2 3 4)

# Temporally solved...
emu_mode="dig2an"

# Names of the signals files from which stats will be extracted
signals_file1="complete_signal"
signals_file2="4_data_decompressor"

# Check last simulation id number and get the following simulation id number
current_dir=`pwd`
sims_dir="/data/simulations"
complete_dir="$current_dir$sims_dir"
sim_id=0
for prev_sim_id in "$complete_dir"/*
do
  if [ $(basename $prev_sim_id) -gt $sim_id ]
  then
    sim_id=$(basename $prev_sim_id)
  fi
done
sim_id=$((sim_id+1))

# Create the folder for the new simulation
sim_dir="$complete_dir/$sim_id"
mkdir $sim_dir

# Create a subdirectory to save the signals
signals_dir="$sim_dir/signals"
mkdir $signals_dir

# Create another subdirectory to save the graphs
graphs_dir="$sim_dir/graphs"
mkdir $graphs_dir

# Copy the configuration JSON into the created folder
cp sim_configuration.json data/simulations/$(basename $sim_dir)/sim_configuration.json

echo
echo "Starting simulation number $sim_id"
echo
# Run selected mode
if [ "$mode" = "run" ]
then
  echo "Run mode selected."
  for ((stg=0; stg < ${#actions[@]}; ++stg)); do
    if [ "${actions["$stg"]}" = 0 ]; then
      # Inject noise
      echo
      echo "----------------------------"
      echo "Noise injection. Stage: $stg"
      echo "----------------------------"
      python 0_noise_injector.py $sim_id $stg

    elif [ "${actions["$stg"]}" = 1 ]; then
      # Compress data
      echo
      echo "-----------------------------"
      echo "Data compression. Stage: $stg"
      echo "-----------------------------"
      python 1_data_compressor.py $sim_id $stg

    elif [ "${actions["$stg"]}" = 2 ]; then
      # Emulate physical transmission
      echo
      echo "----------------------------------"
      echo "Physical transmission. Stage: $stg"
      echo "----------------------------------"
      python 2_physical_link_emulator.py $sim_id $stg $emu_mode
      emu_mode="an2dig"

    elif [ "${actions["$stg"]}" = 3 ]; then
      # Decompress data
      echo
      echo "-------------------------------"
      echo "Data decompression. Stage: $stg"
      echo "-------------------------------"
      python 3_data_decompressor.py $sim_id $stg

    elif [ "${actions["$stg"]}" = 4 ]; then
      # Extract stats and charasteristics
      echo
      echo "---------------------------------"
      echo "Characteristics extraction. Stage: $stg"
      echo "---------------------------------"
      python 4_stats_extractor.py $sim_id $stg $signals_file1 $signals_file2

    elif [ "${actions["$stg"]}" = 5 ]; then
      # Classify signals
      echo
      echo "---------------------------------"
      echo "Noise classification. Stage: $stg"
      echo "---------------------------------"

    else
      echo
      echo "Unknown action selected: $actions[$((stg-1))]."
    fi
done
fi
