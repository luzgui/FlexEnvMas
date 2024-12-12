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

#%% Create analyzer objects

experiment_name=exp_group_defs['exps']
days_to_plot=exp_group_defs['days_to_plot']
save_file=exp_group_defs['save_file']

analysis_objs=dict.fromkeys(experiment_name)



for exp in experiment_name:
    print(exp)
    analysis_objs[exp]={}
    for k in exp_group_defs[exp]['experiments']:
        env_name=exp_group_defs[exp]['experiments'][k]['test_env']
        configs=ConfigsParser(configs_folder, env_name)
        
        file_agents_conf,file_apps_conf,file_scene_conf,file_prob_conf,file_vars,file_experiment,ppo_config=configs.get_configs()
        
        gecad_dataset=datafolder / 'dataset_gecad_clean.csv'
        test_com=Community(file_agents_conf,
                      file_apps_conf,
                      file_scene_conf,
                      file_prob_conf,
                      gecad_dataset)
        
        com_vars=StateVars(file_vars)
        test_env_config={'community': test_com,
                    'com_vars': com_vars,
                    'num_agents': test_com.num_agents}
           
        tenv=FlexEnv(test_env_config)
        
        folder=resultsfolder / exp_group_defs[exp]['experiments'][k]['folder']
        folder_bl=resultsfolder / exp_group_defs[exp]['experiments'][k]['baseline_folder']
        folder_opti=resultsfolder / exp_group_defs[exp]['experiments'][k]['opti_folder']
        analysis_objs[exp][k]=Analyzer(folder, folder_bl, folder_opti,tenv)
        
        # plot some days
        for day in days_to_plot:
            analysis_objs[exp][k].plot_one_day(day, 'rl', save=save_file)
            analysis_objs[exp][k].plot_one_day(day, 'opti', save=save_file)
            
    
    analyse_multi=AnalyzerMulti(analysis_objs[exp],exp)
    analyse_multi.plot_multi_joint(x='x_ratio',y='dif',save=save_file)
    # analyse_multi.plot_multi_joint(x='x_ratio',y='dif_simple',save=save_file)
    # analyse_multi.plot_multi_joint(x='x_ratio',y='gamma',save=save_file)
    data=analyse_multi.get_multi_cost_compare()


#%%
obj=analysis_objs['expdouble']['exp0001']
s=obj.state
t=2880
s1=s.loc[t:t+96]
d=345
obj.plot_one_day(d,'rl')
obj.plot_one_day(d,'opti')
