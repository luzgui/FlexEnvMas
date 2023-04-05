#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 10:48:58 2022

@author: omega
"""

import gymnasium as gym

import ray #ray2.0 implementation

from ray import tune

from ray import air
# from ray.tune import Analysis
from ray.tune import analysis
from ray.tune import ExperimentAnalysis
from ray.tune import TuneConfig
# from ray.tune.execution.trial_runner import _load_trial_from_checkpoint
from ray.tune.experiment import trial

from pathlib import Path

#PPO algorithm
from ray.rllib.algorithms.ppo import PPO, PPOConfig #trainer and config

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
from pathlib import Path
import sys
import time

import datetime
from datetime import datetime

#Custom functions
# from shiftenvRLlib import ShiftEnv
from auxfunctions_shiftenv import *
from plotutils import *
from models2 import ActionMaskModel, CCActionMaskModel

from ray.rllib.utils.pre_checks import env

import random

from trainable import *
from obs_wrapper import *

# from ray.tune.registry import register_env
# from ray.rllib.examples.env.multi_agent import MultiAgentCartPole, make_multi_agent
from shiftenvRLlib_mas import ShiftEnvMas

from auxfunctions_CC import *

# Custom Model
ModelCatalog.register_custom_model('shift_mask', ActionMaskModel)
ModelCatalog.register_custom_model("cc_shift_mask", CCActionMaskModel)

from ray.rllib.env.wrappers.multi_agent_env_compatibility import MultiAgentEnvCompatibility

#profiling
import cProfile
import pstats
from pstats import SortKey

#paths

cwd=Path.cwd()
datafolder=cwd / 'Data'
raylog=cwd / 'raylog'
prof_folder=raylog / 'profiles'


##############################################################################



#%% Make environment config / Data
import data_process
env_config=data_process.make_env_config(datafolder.as_posix())

# filename=prof_folder+'/make_env_profile'
# conf = cProfile.run('data_process.make_env_config(datafolder)',filename)


#%% Make environment instance

import environment_build
# 
menv_base=environment_build.make_env(env_config)
#menv_base.normalization=False

# filename=prof_folder+'/env_profile_' + exp_name
# menv_base=cProfile.run('environment_build.make_env(env_config)',filename)



# menv=environment_build.make_env(env_config)

menv=MultiAgentEnvCompatibility(menv_base)
# menv=NormalizeObs(menv_base)
# menv=menv_base
menv_data=menv.env.data


from ray.tune.registry import register_env
def env_creator(env_config):
    # return NormalizeObs(menv_base)  # return an env instance
    return MultiAgentEnvCompatibility(menv_base)
    # return menv_base

register_env("shiftenv", env_creator)

# register_env("shiftenv", menv)




#%% Make experiment/train Tune config
import experiment_build

exp_name='test-CC-Normal'

# pol_type='shared_pol'
pol_type='agent_pol'

config, config_tune=experiment_build.make_train_config(menv,pol_type)
# config.observation_filter='MeanStdFilter'

#configs for FCUL-PC
# config.num_rollout_workers=25


#%% Train

# stop = {"training_iteration": 1}


# trainer=CentralizedCritic(config)
# filename=prof_folder+'/trainer_profile_' + exp_name
# results = cProfile.run('trainer.train()',filename)

# trainer.train()




run_config=air.RunConfig(verbose=3, 
                         name=exp_name,
                         local_dir=raylog.as_posix())


# # resources=tune.PlacementGroupFactory([{'CPU': 1.0}] + [{'CPU': 1.0}] * 27 +  [{'GPU': 1.0}]) #reosurces FCUL
resources=tune.PlacementGroupFactory([{'CPU': 1.0}] + [{'CPU': 1.0}] * 4)
trainable_resources = tune.with_resources(trainable_mas, resources)

tuner = tune.Tuner(
      trainable_resources,
      param_space=config,
      tune_config=config_tune,
      run_config=run_config)


# tuner.fit()

filename=prof_folder / ('tuner_profile_' + exp_name)
results = cProfile.run('tuner.fit()',filename)


#%%
# # TESTING
# #########################################################
# #%% Test
# import test_build

# # test_exp_name='test_ist_2ag_gs'
# # test_exp_name='test-3000-2g-FCUL-comp'
# # test_exp_name='test-3000-2g-FCUL'

# test_exp_name=exp_name

# # Good ones
# # test_exp_name='test-Feb13'
# # test_exp_name='test-shared-2ag-FCUL'
# # test_exp_name='test-shared-collective-reward-FCUL'


# tenv, tester, best_checkpoint = test_build.make_tester(test_exp_name,raylog,datafolder)

# # tenv=NormalizeObs(tenv)

# tenv_data=tenv.data
# #%% Plot
# import test_agents
# full_state, env_state, metrics=test_agents.test(tenv, 
#                                                 tester, 
#                                                 n_episodes=1,
#                                                 plot=True)
# # print(metrics)
# from plotutils import *
# make_boxplot(metrics,tenv)

# m=metrics.loc['com']

# print(metrics.loc['ag1']['selfsuf'].mean())

# # metrics.to_csv('metrics_competitive_365_sequential.csv')





#%% Profilling

p = pstats.Stats(filename.as_posix())
p.sort_stats(SortKey.CUMULATIVE).print_stats(20)





