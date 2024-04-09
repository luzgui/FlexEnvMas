#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 15:01:52 2024

@author: omega
"""

import os
from os import path
from pathlib import Path
import sys


cwd=Path.cwd()
datafolder=cwd / 'Data'
raylog=cwd / 'raylog'
configs=cwd / 'configs'
algos_config = configs / 'algos_configs'


from community import Community
from dataprocessor import *

#env classes
from environment import FlexEnv
from state import StateVars
from experiment import Experiment
from experiment_test import SimpleTests

#ray + gym
from ray.rllib.env.wrappers.multi_agent_env_compatibility import MultiAgentEnvCompatibility

import gymnasium as gym

import ray #ray2.0 implementation
from ray import tune, air, train
from ray.tune import analysis, ExperimentAnalysis, TuneConfig
from ray.tune.experiment import trial

#PPO algorithm
from ray.rllib.algorithms.ppo import PPO, PPOConfig #trainer and config
from ray.rllib.env.env_context import EnvContext
#models
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.pre_checks import env

# Custom Model
from models2 import ActionMaskModel, CCActionMaskModel
ModelCatalog.register_custom_model('shift_mask', ActionMaskModel)
ModelCatalog.register_custom_model("cc_shift_mask", CCActionMaskModel)

#System
import os
from os import path
from pathlib import Path
import sys
import time
import datetime
from datetime import datetime
import json

#Custom functions
# from shiftenvRLlib import ShiftEnv

from models2 import ActionMaskModel, CCActionMaskModel



import random

from trainable import *
from obs_wrapper import *


from auxfunctions_CC import *

import time
start_time = time.time()


#%% get configs
file_ag_conf= configs / 'agents_config.yaml'
file_apps_conf= configs / 'apps_config.yaml'
file_scene_conf = configs / 'scenario_config.yaml'
file_prob_conf = configs / 'problem_config.yaml'
file_vars = configs / 'state_vars.yaml'
# file_vars = configs / 'state_vars_fixed.yaml'
file_experiment = configs / 'experiment_config.yaml'

#algos configs
ppo_config=algos_config / 'ppo_config.yaml'

#%% import datafiles and agent dataprocessor
# gecad_dataset=datafolder / 'Dataset_gecad_changed.xlsx'
gecad_dataset=datafolder / 'dataset_gecad_clean.csv'
# gecad_processor=GecadDataProcessor(file_prob_conf,file_ag_conf,gecad_dataset)
# data=gecad_processor.data
#%% Make community            
com=Community(file_ag_conf,
              file_apps_conf,
              file_scene_conf,
              file_prob_conf,
              gecad_dataset)


com_vars=StateVars(file_vars)



#%%  Make environment   
env_config={'community': com,
            'com_vars': com_vars,
            'num_agents': com.num_agents}
   
envi=FlexEnv(env_config)

#%%
menvi=MultiAgentEnvCompatibility(envi)
menvi._agent_ids=['ag1', 'ag2', 'ag3']
from ray.tune.registry import register_env
def env_creator(env_config):
    # return NormalizeObs(menv_base)  # return an env instance
    new_env=MultiAgentEnvCompatibility(envi)
    new_env._agents_ids=envi._agent_ids
    return new_env
    # return MultiAgentEnvCompatibility(envi)
    # return menv_base

register_env("flexenv", env_creator)

#%% experiment
# from experiment import *


experiment=Experiment(envi, file_experiment)
config=experiment.make_algo_config(ppo_config)
config_tune=experiment.make_tune_config()
config_run=experiment.make_run_config(raylog.as_posix())


resources=tune.PlacementGroupFactory([{'CPU': 1.0}] + [{'CPU': 1.0}] * 6)

trainable_resources = tune.with_resources(trainable_mas, resources)

spill_1=raylog / 'spill1'
spill_2=raylog / 'spill2'

ray.init(_system_config={"local_fs_capacity_threshold": 0.99,
                         "object_spilling_config": json.dumps({"type": "filesystem",
                                                               "params": {"directory_path":[spill_1.as_posix(),
                                                                                            spill_2.as_posix()],}},)},)

tuner = tune.Tuner(
      trainable_resources,
      param_space=config,
      tune_config=config_tune,
      run_config=config_run)

#%% Simple agent test
# from experiment_test import SimpleTests
# from testenv import *
# dummy_tester=DummyTester(envi)
# simpletest=SimpleTestEnv(envi, dummy_tester)
# full_state, env_state_conc, episode_metrics, filename = simpletest.test(1,[],plot=True)

# simpletest.test_full_state(full_state)


#%% Train
results=tuner.fit()
print(results.errors)
# 

#%% Time
end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time:", elapsed_time, "seconds")


