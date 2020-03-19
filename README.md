# Data Transmission Simulator
Python simulator for data transmission over desired channel. 

*Advice: The current version is very incomplete. There are lot of bugs to be fixed and functionalities to be developed.*

This project aims to emulate the communication process over a simulated physical link (RF, water acoustics or whatever) considering the effect of the noise over the analogized signal (from a original digital one), so that, bit-flip due to noise, attenuation, echo and other problems sources can be analyzed by simulation.

Wavelets are implemented to be used for data compression before the emulation of the transmission. 

All modules can be configured by modifying the configuration JSON file, which has different section for each module.

## Launch
The simulator has been developed and run in Ubuntu 16.04 LTS, so there is not guarantee for it to work in another environment. Before trying to launch the simulator, check you have installed all required python libraries. A requirements file will be uploaded soon.

In order to launch the simulator, you have to modify the configuration file **sim_configuration.json** for the desired options and to modify the bash launcher to select the simulation sequence. To change the input signal of the transmission simulation, change the field *general -> signals_df_name* (without the file extension) and move your signal CSV file into the *original_data* folder, inside *Data*. 
And sure, change the name of the signal in *general -> signals_names* in the same configuration JSON file (for the moment, it is only possible to launch the simulation with one signal at the same time, there is a bug party with this). Tune the rest of the parameters as you like and, if you find more bugs and errors, notify them, please.

If the KPI extractor is used, select an origin signal name and another final signal name (as they may be compared for some of the KPI calculation). Afther that, change the access mode of the bash script (if necessary) using:

```
chmod 777 simulation_orchestrator
```

and run the simulation orchestrator:

```
./simulation_orchestrator.bash
```

The following is a simulation terminal output example:

```
Starting simulation number 64

Run mode selected.

-----------------------------
Data compression. Stage: 0
-----------------------------
Data compression started.
Compression of signal ax with wavelet sym4
Data compression process finished.
Plotting signal...
Closing data compression process.

----------------------------------
Physical transmission. Stage: 1
----------------------------------
Transmission physics simulation started.
Digital signal to analog...
Transmission physics simulation finished.
Closing data compression process.

----------------------------
Noise injection. Stage: 2
----------------------------
Noise injection process started.
Injection of white_noise in signal ax
Noise injection process finished.
Closing noise injection process.

----------------------------------
Physical transmission. Stage: 3
----------------------------------
Transmission physics simulation started.
Analog signal to digital...
Transmission physics simulation finished.
Closing data compression process.

-------------------------------
Data decompression. Stage: 4
-------------------------------
Original signal lenght: 162500
Data decompression started.
Signal length after decompression: 162586
Data decompression process finished.
Plotting signal...
Closing data decompression process.

---------------------------------
Characteristics extraction. Stage: 5
---------------------------------
Statistics extraction started.
Correlation max for signal ax the is at 81250
Delay for signal ax is 0
For signal ax the MSE is: 6014.30133836
Power Density Spectrum calculated for signal ax
PDS difference calculated for signal ax
Saving Correlation into file...
Saving Delay into file...
Saving MSE into file...
Saving PDS into file...
Saving PDScomparison into file...
Statistics extraction process finished.
Closing statistics extraction process.
```
