#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 11:05:42 2024

@author: omega

This script performs tests whose actions are predefined

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
from rl.models.models2 import ActionMaskModel, CCActionMaskModel
from rl.models.models_rnn import LSTMActionMaskModel



import random

from trains.trainable import *
# from obs_wrapper import *

# from shiftenvRLlib_mas import ShiftEnvMas

# from rl.algos.auxfunctions_CC import *
from rl.algos.central_critic import *

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

from testings.experiment_test import ExperimentTest
from testings.testenv import TestEnv, DummyTester, BaselineFixedTest, TestEnvOpti
from trains.trainable import Trainable
from utils.utilities import ConfigsParser
from env.community import Community
from env.state import StateVars
from env.environment import FlexEnv
from env.environment_v1 import FlexEnvV1
from analyze.plots import Plots

from opti.optimize import CommunityOptiModel


#paths

cwd=Path.cwd()
datafolder=cwd.parent  / 'Data'
raylog=cwd.parent  / 'raylog'
prof_folder=raylog / 'profiles'
resultsfolder=cwd.parent  / 'Results'
storage_path='/home/omega/Downloads/ShareIST'

configs_folder=cwd.parent  / 'configs'

#paths


#%% exp_name + get configs for experiment
exp_name=YAMLParser().load_yaml(configs_folder / 'exp_name.yaml')['exp_name']
file_experiment, algo_config=ConfigsParser(configs_folder, exp_name).get_configs() # only a testing environment

#%% get configs for testing environment
test_config_file=configs_folder / 'exp_name.yaml'
test_name=YAMLParser().load_yaml(test_config_file)['test_env']
file_ag_conf, file_apps_conf, file_scene_conf, file_prob_conf ,file_vars=ConfigsParser(configs_folder, test_name).get_configs()

#%%Make test env
#dataset
file=YAMLParser().load_yaml(file_prob_conf)['dataset_file']
gecad_dataset=datafolder / file
        
#%%  Test environment   
  

test_com=Community(file_ag_conf,
              file_apps_conf,
              file_scene_conf,
              file_prob_conf,
              gecad_dataset)


com_vars=StateVars(file_vars)

test_env_config={'name': test_name,
                 'community': test_com,
                 'com_vars': com_vars,
                 'num_agents': test_com.num_agents}
   

# folder_name='baseline_fixed_' + test_name
# folder_name='Train'
env_cls_name=YAMLParser().load_yaml(file_prob_conf)['env_cls']
tenvi=globals().get(env_cls_name)(test_env_config)

# tenvi=FlexEnv(test_env_config)
dummy_tester=DummyTester(tenvi)
opti_actions_folder=resultsfolder / YAMLParser().load_yaml(test_config_file)['actions_from_folder']

optitest=TestEnvOpti(tenvi, dummy_tester, file_experiment, test_config_file, opti_actions_folder)
# optitest=TestEnvOpti(tenvi, dummy_tester, exp_name, test_config_file, opti_actions_folder)
full_state, env_state_conc, episode_metrics, filename = optitest.test(results_path=resultsfolder)

# baselinefixedtest= BaselineFixedTest(tenvi, dummy_tester,file_experiment,test_config_file)
# full_state, env_state_conc, episode_metrics, filename = baselinefixedtest.test(results_path=resultsfolder)

#%%


