# Data Transmission Simulator
Python simulator for data transmission over desired channel. 

*Advice: The current version is very incomplete. There are lot of bugs to be fixed and functionalities to be developed.*

This project aims to emulate the communication process over a simulated physical link (RF, water acoustics or whatever) considering the effect of the noise over the analogized signal (from a original digital one), so that, bit-flip due to noise, attenuation, echo and other problems sources can be analyzed by simulation.

Wavelets are implemented to be used for data compression before the emulation of the transmission. 

All modules can be configured by modifying the configuration JSON file, which has different section for each module.

# Launch
In order to launch the simulator, you have to modify the configuration file for the desired options and to modify the bash launcher to select the simulation sequence and, if the KPI extractor is used, select an origin signal name and another final signal name (as they may be compared for some of the KPI calculation).
