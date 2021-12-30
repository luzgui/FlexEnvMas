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
    env_data=data.to_numpy()
    load=env_data[0:timestep,load_num]
    gen=abs(env_data[0:timestep,1])
    data=np.vstack((1.5*gen,6*load)).T
    
    #make the environment
    env=flex.FlexEnv(data,soc_max,eta,charge_lim)
    check_env(env,warn=True, skip_render_check=False)
    
    return env


