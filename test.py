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
from icecream import ic

#pyomo
from auxfunctions_opti import *
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

#paths

cwd=Path.cwd()
datafolder=cwd / 'Data'
raylog=cwd / 'raylog'
prof_folder=raylog / 'profiles'
resultsfolder=cwd / 'Results'
storage_path='/home/omega/Downloads/ShareIST'


#%%



test=ExperimentTest(envi,
          config_run.name, 
          raylog,
          experiment.config,
          trainable_mas,
          PPO)

test_agent=test.get_tester()

#%%

from testenv import TestEnv
tester=TestEnv(envi, test_agent)

full_state, env_state, metrics, results_filename_path=tester.test(
                                                    n_episodes=1,
                                                    plot=True,
                                                    results_path=None)

 # results_path=Path(cp.path).parent



