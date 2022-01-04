import gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env

import flexenv as flex
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy.random as rnd
import time
import random as rnd


def make_env(data,load_num,timestep,soc_max,eta,charge_lim):
    #prepare the data
    """
    Load and PV production data are loaded into the environment and we must choose the length of the trainning data
    
    arguments:
        
    data: a pandas dataframe with number of columns N=# of houses or loads and time-horizon T=Total number of timesteps 
    load_num: The index of the house to consider (Specific for this data set)
    timestep: the number of timesteps to consider (in this case we have 30 minutes X 1 year what means 1day=48 timesteps)
    soc_max: maximum battery state-of-charge
    eta: Charging efficiêncy
    charge_lim: maximum charging power
    
    
    """
    env_data=data.to_numpy()
    load=env_data[0:timestep,load_num]
    gen=abs(env_data[0:timestep,1])
    data=np.vstack((1.5*gen,6*load)).T
    
    #make the environment
    env=flex.FlexEnv(data,soc_max,eta,charge_lim)
    check_env(env,warn=True, skip_render_check=False)
    
    return env


