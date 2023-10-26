    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 09:22:27 2023

@author: omega
"""

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
storage_path='/home/omega/Downloads/ShareIST'


    #%% Test

import test_build

#runs in FCUL after normalization if inputs
#centralized Critic
# exp_name='test-CC-Normal'
# exp_name='test-CC-Normal-shared-F2'

#independent learning
# exp_name='PPO-IL-Normal-F1'
# exp_name='PPO-IL-Normal-shared-F2'


#LowPV agents
#centralized Critic
# exp_name='CC-Normal-ag_pol'
# exp_name='CC-Normal-shared-PVLow-F1'

#Independent agents
# exp_name='PPO-IL-ag_pol-PVLow'
# exp_name='PPO-IL-Shared-PVLow'





# test_exp_name='cluster'

# Good ones
# test_exp_name='test-Feb13'
# test_exp_name='test-shared-2ag-FCUL'
test_exp_name='test-shared-collective-reward-FCUL'

#DimRedux
# exp_name ='agpol_cc_cluster'
# exp_name ='agpol_cc_cluster_with_delta'
# exp_name ='agpol_cc_cluster_with_no_delta_no_excess'

# exp_name=os.path.join('Experiment-dimRedux', exp_name)

# exp_name='bableas'
# exp_name='ccenas-tune-cp'
# exp_name='post-tstep-bug'
# exp_name='storage-test'
# exp_name='storage_test-win2'
# exp_name='deb-test'
exp_name='deb0'

test_exp_name=exp_name

# raylog=raylog / 'Experiment-dimRedux'


tenv, tester, best_checkpoint = test_build.make_tester(test_exp_name,raylog,datafolder)
            
# tenv=NormalizeObs(tenv)

tenv_data=tenv.data
trainable_path = Path(best_checkpoint.path).parent #path of the trainbale folder to store results
# trainable_path=''

#%% Plot
import test_agents

full_state, env_state, metrics, results_filename_path=test_agents.test(tenv, 
                                                    tester, 
                                                    n_episodes=1,
                                                    plot=True,
                                                    results_path=trainable_path)


#%% extract the kth day from env_state
k=1
w=96
one_day=env_state2.iloc[k*w:(k+1)*w]

#%% make cost plots
from plotutils import *
make_boxplot(metrics,tenv)
make_costplot(None,None,results_filename_path,save_fig=False)
              
#%%              



# #import metrics from file
# csv_list=get_files(resultsfolder / 'Train-PVHigh','.csv','metrics')

# # metrics=pd.read_csv(csv_list[0], index_col=0)
# # metrics_com=metrics.loc['com']
# # make_costplot(metrics,'plot-metrics-PPO-IL-Shared-TrainPVHigh-TestPVHigh.png', False)


# # filename=resultsfolder / 'PicsSample' / 'joint_plot.png'

# TotalCosts={}
# for f in csv_list:
#     print(f)
#     file_path=Path(f)
#     metrics=pd.read_csv(f, index_col=0)
#     new_filename='plot-' + file_path.name
#     new_file_path=file_path.with_name(new_filename)
#     new_file_path=new_file_path.with_suffix('.png')
    
#     make_costplot(metrics,new_file_path,True)
#     print(new_file_path.name)
#     TotalCosts[new_file_path.name]=metrics.loc['com']['cost'].sum()

# # m=metrics.loc['com']

# # print('self-suf mean:', metrics.loc['ag1']['selfsuf'].mean())

# # metrics.to_csv('metrics_competitive_365_sequential.csv')

# penguins = sns.load_dataset("penguins")
# sns.JointGrid(data=penguins, x="bill_length_mm", y="bill_depth_mm")
# g = sns.JointGrid(data=penguins, x="bill_length_mm", y="bill_depth_mm")
# g.plot(sns.scatterplot, sns.histplot)