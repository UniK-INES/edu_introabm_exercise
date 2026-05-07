'''
Created on 03.06.2022

@author: Sascha Holzhauer
'''

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


from fire_evacuation.model import FireEvacuation, FireEvacuationScenario
from fire_evacuation.agent import Human

if __name__ == '__main__':

    scenario = FireEvacuationScenario(
        random_spawn = True,
        floor_size = 14,
        human_count = 70,
        alarm_believers_prop = 0.5,
        max_speed = 2,
        seed = 3
        )
        
    evacuation = FireEvacuation(scenario)
    
    # Run the model
    evacuation.run_for(1000)
        
    counter = 0
    steps2escape = 0
    for agent in evacuation.agents:
        if isinstance(agent, Human):
            counter +=1
            steps2escape += agent.numsteps2escape
            
    print("Avg. steps to escape: " + str(steps2escape/counter))
        