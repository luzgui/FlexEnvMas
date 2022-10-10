import gym

import ray #ray2.0 implementation

from ray import tune, air
# from ray.tune import Analysis
from ray.tune import analysis
from ray.tune import ExperimentAnalysis
from pathlib import Path

#PPO algorithm
from ray.rllib.algorithms.ppo import PPO #trainer
from ray.rllib.algorithms.ppo import PPOConfig #config


from ray.rllib.env.env_context import EnvContext

#models
from ray.rllib.models import ModelCatalog

#math + data
import pandas as pd
import numpy as np

#Plotting
import matplotlib.pyplot as plt

#System
import os
from os import path
import sys
import time

import datetime
from datetime import datetime

#Custom functions
from shiftenvRLlib import ShiftEnv
from auxfunctions_shiftenv import *
from plotutils import makeplot
from models2 import ActionMaskModel

from ray.rllib.utils.pre_checks import env

import random

from trainable import trainable

# from ray.tune.registry import register_env



from ray.rllib.examples.env.multi_agent import MultiAgentCartPole, make_multi_agent

from shiftenvRLlib_mas import ShiftEnvMas



cwd=os.getcwd()
datafolder=cwd + '/Data'

raylog=cwd + '/raylog'
#Add this folder to path


#%% Number of agents
num_agents=2


#%% Make Shiftable loads environment
#import raw data
# data=pd.read_csv(datafolder + '/env_data.csv', header = None).to_numpy()
# data_raw=pd.read_csv(datafolder + '/issda_data_halfyear.csv').to_numpy()
data_raw=pd.read_csv(datafolder + '/issda_data_halfyear.csv')

data=data_raw[['minutes','PV','id2000', 'id2001', 'id2002', 'id2004', 'id2005',
'id2006', 'id2007', 'id2008', 'id2009', 'id2010', 'id2011', 'id2013',
'id2015', 'id2017', 'id2018', 'id2019', 'id2022', 'id2023', 'id2024',
'id2025', 'id2027', 'id2028', 'id2029', 'id2032', 'id2034', 'id2035',
'id2036', 'id2037', 'id2038', 'id2039', 'id2041', 'id2042', 'id2045',
'id2046', 'id2047', 'id2048', 'id2049', 'id2052', 'id2053', 'id2054',
'id2055', 'id2056', 'id2057', 'id2058', 'id2059', 'id2062', 'id2064']]


tstep_size=30 # number of minutes in each timestep
# %% convert to env data
tstep_per_day=48 #number of timesteps per day
num_days=7 #number of days
# timesteps=tstep_per_day*num_days #number of timesteps to feed the agent
timesteps=len(data)-1

load_id=['id2000', 'id2001'] #ISDDA id of the load to consider

env_data=make_env_data_mas(data, timesteps, load_id, 0.5, num_agents)

## Shiftable profile example

# shiftprof=0.5*np.ones(6)
# shiftprof=np.array([0.5,0.3,0.2,0.4,0.8,0.3])
shiftprof=np.array([0.3,0.3,0.3,0.3,0.3,0.3])


#%% make train env

# reward_type='next_time_cost'
# reward_type='simple_cost'
# reward_type='shift_cost'
# reward_type='next_shift_cost'
# reward_type='gauss_shift_cost'
# reward_type='excess_cost'
reward_type='excess_cost_max'
# reward_type='excess_cost_3'




env_config={"step_size": tstep_size,
            'window_size':tstep_per_day,
            'tstep_per_day':tstep_per_day,
            "data": env_data,
            "reward_type": reward_type, 
            "profile": shiftprof, 
            "time_deliver": 37*tstep_size, 
            'done_condition': 'mode_window',
            'init_condition': 'mode_window',
            'tar_type':'bi',
            'env_info': 'training environment'}

env_config['num_agents']=8



from shiftenvRLlib_mas import ShiftEnvMas
shiftenvmas=ShiftEnvMas(env_config)

shiftenvmas.agents_id




shiftenvmas.t_ahead


obs=shiftenvmas.reset()




###################





# def env_creator(config):
#     return ShiftEnv(config)

# mas_cls=make_multi_agent(env_creator)
# masenv=mas_cls(env_config)


# action={k:random.randint(0,1) for k in range(env_config['num_agents'])}


# obs=masenv.reset()
# ob2=masenv.step(action)



















