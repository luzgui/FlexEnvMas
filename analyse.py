#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 18:13:37 2024

@author: omega
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 11:05:42 2024

@author: omega
"""



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
from plotutils import *



import random


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
from trainable import Trainable
from utilities import ConfigsParser, FolderUtils
from dataprocessor import YAMLParser
from community import Community
from state import StateVars
from environment import FlexEnv
from analyzer import Analyzer
#paths

cwd=Path.cwd()
datafolder=cwd / 'Data'
raylog=cwd / 'raylog'
prof_folder=raylog / 'profiles'
resultsfolder=cwd / 'Results'
storage_path='/home/omega/Downloads/ShareIST'

configs_folder=cwd / 'configs'

#%% 
from analyzer import Analyzer, AnalyzerMulti
from plots import Plots
exp_group_defs=YAMLParser().load_yaml(configs_folder / 'results_config.yaml')

#%% Comparison experiment
results_config=configs_folder / 'results_config.yaml'
anal_test_name='exp_group1'
analyse_multi=AnalyzerMulti(results_config, anal_test_name)

#%% plots (multi experiment)
analyse_multi.plot_year_cost(save=True)
analyse_multi.plot_multi_joint(x='x_ratio',y='save_rate',save=True)
analyse_multi.plot_per_agent_cost_multi_hist(save=True)


#%% dataframes

baseline_costs=analyse.get_baseline_costs()
costs=analyse.get_cost_compare()
costs_per_agent=analyse.get_per_agent_costs()

data_opti=analyse.get_one_day_data(184, 'opti')
data_rl=analyse.get_one_day_data(184, 'rl')


costs_multi=analyse_multi.get_multi_cost_compare()


# analyse.plot_joint(x='x_ratio',y='save_rate',save=False)
# analyse.plot_cost_hist(save=False)

analyse.plot_one_day(184, 'opti')
analyse.plot_one_day(184, 'rl')


#%%


