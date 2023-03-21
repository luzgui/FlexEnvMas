#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 10:48:58 2022

@author: omega
"""

import gym

import ray #ray2.0 implementation

from ray import tune, air
# from ray.tune import Analysis
from ray.tune import analysis
from ray.tune import ExperimentAnalysis
from ray.tune import TuneConfig
from ray.tune.execution.trial_runner import _load_trial_from_checkpoint
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
import sys
import time

import datetime
from datetime import datetime

#Custom functions
from shiftenvRLlib import ShiftEnv
from auxfunctions_shiftenv import *
from plotutils import *
from models2 import ActionMaskModel

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


#
cwd=os.getcwd()
datafolder=cwd + '/Data'
raylog=cwd + '/raylog'


##############################################################################



#%% Make environment config / Data
import data_process
env_config=data_process.make_env_config(datafolder)


#%% Make environment instance

import environment_build
menv_base=environment_build.make_env(env_config)
# menv=environment_build.make_env(env_config)


# menv=NormalizeObs(menv_base)
menv=menv_base
menv_data=menv.data


from ray.tune.registry import register_env
def env_creator(env_config):
    # return NormalizeObs(menv_base)  # return an env instance
    return menv_base

register_env("shiftenv", env_creator)




#%% Make experiment/train Tune config
import experiment_build

# pol_type='shared_pol'
pol_type='agent_pol'
config=experiment_build.make_train_config(menv,pol_type)
# config.observation_filter='MeanStdFilter'

#configs for FCUL-PC
# config.num_rollout_workers=25

# stop = {"training_iteration": 1}


# tuner = tune.Tuner(
#      CentralizedCritic,
#      param_space=config.to_dict(),
#      run_config=air.RunConfig(stop=stop, verbose=3),)

# results = tuner.fit()

#%% Train
exp_name='test-CC-2'

from trainable import *


# resources=tune.PlacementGroupFactory([{'CPU': 1.0}] + [{'CPU': 1.0}] * 27 +  [{'GPU': 1.0}]) #reosurces FCUL
resources=tune.PlacementGroupFactory([{'CPU': 1.0}] + [{'CPU': 1.0}] * 4)


tuneResults=tune.run(trainable_mas,
          config=config.to_dict(),
           resources_per_trial=resources,
          local_dir=raylog,
          name=exp_name,
          verbose=3)

# Results=tuneResult

######################################################
# TESTING
#########################################################
#%% Test
import test_build

# test_exp_name='test_ist_2ag_gs'
# test_exp_name='test-3000-2g-FCUL-comp'
# test_exp_name='test-3000-2g-FCUL'

test_exp_name=exp_name

# Good ones
# test_exp_name='test-Feb13'
# test_exp_name='test-shared-2ag-FCUL'
# test_exp_name='test-shared-collective-reward-FCUL'


tenv, tester, best_checkpoint = test_build.make_tester(test_exp_name,raylog,datafolder)

# tenv=NormalizeObs(tenv)

tenv_data=tenv.data
#%% Plot
import test_agents
full_state, env_state, metrics=test_agents.test(tenv, 
                                                tester, 
                                                n_episodes=1,
                                                plot=True)
# print(metrics)
from plotutils import *
make_boxplot(metrics,tenv)

m=metrics.loc['com']

print(metrics.loc['ag1']['selfsuf'].mean())

# metrics.to_csv('metrics_competitive_365_sequential.csv')











