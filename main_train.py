#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 09:18:23 2023

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
# from shiftenvRLlib import ShiftEnv
from auxfunctions_shiftenv import *
from plotutils import *
from models2 import ActionMaskModel, CCActionMaskModel



import random

from trainable import *
from obs_wrapper import *

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
resultsfolder=cwd / 'Results'


##############################################################################


#%% Make environment config / Data
import data_process
env_config=data_process.make_env_config(datafolder.as_posix())
env_data=env_config['data']
# filename=prof_folder+'/make_env_profile'
# conf = cProfile.run('data_process.make_env_config(datafolder)',filename)


#%% Make environment instance

import environment_build
# 
menv_base=environment_build.make_env(env_config)
menv_base.reset()
#menv_base.normalization=False

# filename=prof_folder+'/env_profile_' + exp_name
# menv_base=cProfile.run('environment_build.make_env(env_config)',filename)



# menv=environment_build.make_env(env_config)

menv=MultiAgentEnvCompatibility(menv_base)
# menv._agent_ids=menv_base.agents_id
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

exp_name='ccenas-tune-cp'

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


from ray.train import RunConfig, CheckpointConfig

run_config=air.RunConfig(verbose=3, 
                         name=exp_name,
                         local_dir=raylog.as_posix())
                         # checkpoint_config=CheckpointConfig(
        # *Best* checkpoints are determined by these params:
        # checkpoint_score_attribute="episode_reward_mean",
        # checkpoint_score_order="max",
        # num_to_keep=5,
        # checkpoint_at_end=True))

#resources FCUL
# b={'CPU':3,'GPU':0.1}
# resources=tune.PlacementGroupFactory([b]*10)
# config.num_rollout_workers=10


#reosurces local
resources=tune.PlacementGroupFactory([{'CPU': 1.0}] + [{'CPU': 1.0}] * 4)


trainable_resources = tune.with_resources(trainable_mas, resources)

tuner = tune.Tuner(
      trainable_resources,
      param_space=config,
      tune_config=config_tune,
      run_config=run_config)


tuner.fit()

# filename=prof_folder / ('tuner_profile_' + exp_name)
# results = cProfile.run('tuner.fit()',filename)





#%% Profilling

# p = pstats.Stats(filename.as_posix())
# p.sort_stats(SortKey.CUMULATIVE).print_stats(20)


