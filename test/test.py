#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 11:05:42 2024

@author: omega
"""

import gymnasium as gym

import ray #ray2.0 implementation
from ray import tune, air
from ray.tune import analysis, ExperimentAnalysis, TuneConfig
from ray.tune.experiment import trial

#PPO algorithm
from ray.rllib.algorithms.ppo import PPO, PPOConfig #trainer and config
from ray.rllib.env.env_context import EnvContext
#models
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.pre_checks import env

#math + data
import pandas as pd
import numpy as np

#Plotting
import matplotlib.pyplot as plt
import seaborn as sns


#System
import os
from os import path
from pathlib import Path
import sys
import time
import datetime
from datetime import datetime

#Custom functions
# from plotutils import *
from models2 import ActionMaskModel, CCActionMaskModel
from models_rnn import LSTMActionMaskModel



import random

from trainable import *
from obs_wrapper import *

# from shiftenvRLlib_mas import ShiftEnvMas

from auxfunctions_CC import *

# Custom Model
ModelCatalog.register_custom_model('shift_mask', ActionMaskModel)
ModelCatalog.register_custom_model("cc_shift_mask", CCActionMaskModel)
ModelCatalog.register_custom_model("lstm_model", LSTMActionMaskModel)

from ray.rllib.env.wrappers.multi_agent_env_compatibility import MultiAgentEnvCompatibility
from ray.tune.registry import register_env
#profiling
import cProfile
import pstats
from pstats import SortKey
from icecream import ic

#pyomo
# from auxfunctions_opti import *
from pyomo.environ import *
from pyomo.opt import SolverFactory
import scipy.io as sio
import re 
from itertools import compress
from rich import print
from rich.console import Console
from rich.syntax import Syntax

from experiment_test import ExperimentTest
from testenv import TestEnv
from trainable import Trainable
from utilities import ConfigsParser
from community import Community
from state import StateVars
from environment import FlexEnv
from plots import Plots

from experiment_test import ExperimentTest
from testenv import TestEnv

from optimize import CommunityOptiModel


#paths

cwd=Path.cwd()
datafolder=cwd / 'Data'
raylog=cwd / 'raylog'
prof_folder=raylog / 'profiles'
resultsfolder=cwd / 'Results'
storage_path='/home/omega/Downloads/ShareIST'

configs_folder=cwd / 'configs'

#%% exp_name + get configs for experiment
exp_name=YAMLParser().load_yaml(configs_folder / 'exp_name.yaml')['exp_name']
configs=ConfigsParser(configs_folder, exp_name)

_, file_apps_conf, file_scene_conf, _ ,file_vars,file_experiment, ppo_config=configs.get_configs()

#%% get configs for testing environment
test_config_file=configs_folder / 'test_config.yaml'

test_params=YAMLParser().load_yaml(test_config_file)
test_name=test_params['test_name']
#control vars
shouldTest=test_params['shouldTest']
shouldOpti=test_params['shouldOpti']

test_configs=ConfigsParser(configs_folder, test_name)
# between training and testing the difference is the agents config and the problem config
file_ag_conf,_,_,file_prob_conf,_,_,_=test_configs.get_configs()

#%%Make test env
#dataset
gecad_dataset=datafolder / 'dataset_gecad_clean.csv'
         
test_com=Community(file_ag_conf,
              file_apps_conf,
              file_scene_conf,
              file_prob_conf,
              gecad_dataset)


com_vars=StateVars(file_vars)



#%%  Make environment   
test_env_config={'community': test_com,
            'com_vars': com_vars,
            'num_agents': test_com.num_agents}
   
tenvi=FlexEnv(test_env_config)


menvi=MultiAgentEnvCompatibility(tenvi)
# menvi._agent_ids=['ag1', 'ag2', 'ag3']

def env_creator(env_config):
    # return NormalizeObs(menv_base)  # return an env instance
    new_env=MultiAgentEnvCompatibility(tenvi)
    new_env._agents_ids=tenvi._agent_ids
    return new_env
    # return MultiAgentEnvCompatibility(envi)
    # return menv_base

register_env("flexenv", env_creator)

print('number of days for testing:', len(tenvi.allowed_inits))
#%% Test
if shouldTest:
    print('Testing...')
    time.sleep(3)
    #Trainable
    trainable_func=Trainable(file_experiment)._trainable
    #get checkpoint and create tester
    test=ExperimentTest(tenvi,
              exp_name, 
              raylog,
              file_experiment,
              trainable_func,
              test_params['checkpoint'])
    
    tester=test.get_tester(trainable_func)
    
    #Test environment
    env_tester=TestEnv(tenvi, tester, file_experiment,test_config_file)
    env_tester.n_episodes=len(tenvi.allowed_inits)-1
    full_state, env_state, metrics, results_filename_path=env_tester.test(results_path=resultsfolder)

    #%% Optimal Solution
if shouldOpti:
    print('Optimize...')
    time.sleep(3)

    folder=os.path.join(resultsfolder,'optimal_'+test_name)
    model=CommunityOptiModel(tenvi,folder)
    objectives, solutions=model.solve_model_yearly(save=True)


