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
anal_test_name='exp_group2'
analyse_multi=AnalyzerMulti(results_config, anal_test_name)

#%% plots (multi experiment)
analyse_multi.plot_year_cost(save=False)
analyse_multi.plot_multi_joint(x='x_ratio',y='save_rate',save=False)
analyse_multi.plot_per_agent_cost_multi_hist(save=False)

analyse_multi.plot_multi_metrics(save=False)

analyse_multi.analyser_objs['IL'].plot_one_day(11, 'rl')
one_day=analyse_multi.analyser_objs['IL'].get_one_day_data(11, 'rl')

#%% dataframes

bl=analyse_multi.analyser_objs['Random_baseline'].get_baseline_costs()
bl_IL=analyse_multi.analyser_objs['IL'].get_baseline_costs()

costs=analyse_multi.get_multi_cost_compare()

analyse_multi.plot_per_agent_year_cost(save=True)

costs_multi=analyse_multi.get_multi_cost_compare()

base_costs=analyse_multi.analyser_objs['IL'].get_baseline_costs()

algo='CC'
per_agent=analyse_multi.analyser_objs[algo].get_per_agent_costs()
per_agent_m=per_agent.describe()
per_agent_sum=per_agent.sum()
per_agent_mean=per_agent.mean()

analyse_multi.analyser_objs[algo].plot_one_day(11,'opti')


#%%
baseline_analyse=Analyzer(analyse_multi.baseline, analyse_multi.baseline)
base_per_agent_cost=baseline_analyse.get_per_agent_costs()
baseline_analyse.plot_one_day(0, 'rl')
