'''
    Simulation Orchestrator:
    Executes every module in order to run a simulation sequence.
'''

import os

class Orchestrator:
    def __init__(self, sim_id, config_dict):
        self.sim_sequence = config_dict["sequence"]
        self.sim_id = sim_id

    def run():
        
