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
# from plotutils import *



import random


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

# test_names=['exp_group_fc1',
#             'exp_group_train']
            # 'exp_group_sub',
            
# test_names=['exp_group_0','exp_group_sub','exp_group_fc1']

# test_names=['exp_group_0_1','exp_group_train']


# test_names=['exp_group_train']

# test_names=['exp21']
# test_names=['exp2']
# test_names=['exp2','exp21']

# test_names=['exp21','exp21r']
test_names=['exp2r','exp21r']

# exp='exp_group_fc1'
# exp='exp_group_train'
save_file=False
exp_obs={}
for exp in test_names:
    analyse_multi=AnalyzerMulti(results_config, exp)
    exp_obs[exp]=analyse_multi
    analyse_multi.plot_year_cost(save=save_file)
    # analyse_multi.plot_multi_joint(x='x_ratio',y='dif_simple',save=save_file)
    analyse_multi.plot_multi_joint(x='x_ratio',y='dif',save=save_file)
    
    
#%%  
algo='IL'
for exp in test_names:
    obj=exp_obs[exp].get_analyser_objects()
    d=11
    obj[algo].plot_one_day(d,'rl',save_file)
    obj[algo].plot_one_day(d,'opti',save_file)
    

# exp_obj=obj[algo]
# cost=exp_obj.get_cost_compare()

# cost_sum=cost.sum()

# opti_test=exp_obs['exp_group_fc1'].get_analyser_objects()['CC'].opti_objective['objective'].mean()    
# opti_train=exp_obs['exp_group_train'].get_analyser_objects()['IL'].opti_objective['objective'].mean()   

cost_test=exp_obs['exp2'].get_analyser_objects()['IL'].get_cost_compare()   
cost_train=exp_obs['exp_group_train'].get_analyser_objects()['IL'].get_cost_compare() 


cost_test['new']=((cost_test['cost']+1)-(cost_test['objective']+1))/(cost_test['objective']+1)
#%%
# fc1=analyse_multi.analyser_objs['IL']
# data=fc1.get_one_day_data(11, 'rl')

#%% plot days
# d=10
# fc1.plot_one_day(d, 'rl')
# fc1.plot_one_day(d, 'opti')
# analyse_multi.analyser_objs['Random_baseline'].plot_one_day(d, 'rl')
# cost=fc1.get_cost_compare()

#%% plots (multi experiment)




# analyse_multi.plot_per_agent_cost_multi_hist(save=False)

# analyse_multi.plot_multi_metrics(save=False)

# analyse_multi.analyser_objs['CC'].plot_one_day(17, 'rl')
# one_day=analyse_multi.analyser_objs['IL'].get_one_day_data(80, 'rl')

# #%% dataframes

# bl=analyse_multi.analyser_objs['Random_baseline'].get_baseline_costs()
# bl_IL=analyse_multi.analyser_objs['IL'].get_baseline_costs()

# costs=analyse_multi.get_multi_cost_compare()

# analyse_multi.plot_per_agent_year_cost(save=True)

# costs_multi=analyse_multi.get_multi_cost_compare()

# base_costs=analyse_multi.analyser_objs['IL'].get_baseline_costs()

# algo='CC'
# per_agent=analyse_multi.analyser_objs[algo].get_per_agent_costs()
# per_agent_m=per_agent.describe()
# per_agent_sum=per_agent.sum()
# per_agent_mean=per_agent.mean()

# analyse_multi.analyser_objs[algo].plot_one_day(11,'opti')


# #%%
# baseline_analyse=Analyzer(analyse_multi.baseline, analyse_multi.baseline)
# base_per_agent_cost=baseline_analyse.get_per_agent_costs()
# baseline_analyse.plot_one_day(0, 'rl')
