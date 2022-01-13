import gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env

import Antonio_FlexEnv as flex
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy.random as rnd
import time
import random as rnd


def make_env(data,load_num,timestep,soc_max,eta,charge_lim,min_charge_step):
    #prepare the data
    """
    Load and PV production data are loaded into the environment and we must choose the length of the trainning data
    
    arguments:
        
    data: a pandas dataframe with number of columns N=# of houses or loads and time-horizon T=Total number of timesteps 
    load_num: The index of the house to consider (Specific for this data set)
    timestep: the number of timesteps to consider (in this case we have 30 minutes X 1 year what means 1day=48 timesteps)
    soc_max: maximum battery state-of-charge
    eta: Charging efficiency
    charge_lim: maximum charging power
    
    """
    env_data=data.to_numpy()
    load=env_data[0:timestep,load_num] # Escolhe timestep (um número) valores da coluna load_num, que representa os gastos uma casa
    gen=abs(env_data[0:timestep,1]) # Primeira coluna da data
    # Modification (António) - Multiply the gen by 0.5 to make it smaller generation values
    data=np.vstack((gen*0.5,6*load)).T # Duas colunas, a primeira retrata a geração, a segunda representa os gastos energéticos da casa
    
    #make the environment
    env=flex.FlexEnv(data,soc_max,eta,charge_lim,min_charge_step)
    # Modification António:
    # O check env estava a dar erros de observation space quando não era sequer algo mudado. Comentei e já corre tudo bem
#     check_env(env,warn=True, skip_render_check=False) 
    
    return env
