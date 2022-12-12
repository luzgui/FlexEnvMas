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

from trainable import *

# from ray.tune.registry import register_env
from ray.rllib.examples.env.multi_agent import MultiAgentCartPole, make_multi_agent
from shiftenvRLlib_mas import ShiftEnvMas
# Custom Model
ModelCatalog.register_custom_model('shift_mask', ActionMaskModel)


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
menv=environment_build.make_env(env_config)

#%% Make experiment/train Tune config
import experiment_build
config=experiment_build.make_train_config(menv)

#%% Train
exp_name='script_test'

from trainable import *


tuneResults=tune.run(trainable_mas,
         config=config.to_dict(),
         resources_per_trial=tune.PlacementGroupFactory([{'CPU': 1.0}] + [{'CPU': 1.0}] * 4),
         local_dir=raylog,
         name=exp_name,
         verbose=3)

# Results=tuneResult


#%% Test
import test_build
tenv, tester = test_build.make_tester(exp_name,raylog,datafolder)


#%% Plot
import test_agents
full_state, env_state=test_agents.test(tenv, tester, n_episodes=1)



