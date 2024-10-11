#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 15:01:52 2024

@author: omega
"""


from pathlib import Path
import time
import json


cwd=Path.cwd()
datafolder=cwd / 'Data'
raylog=cwd / 'raylog'
configs_folder=cwd / 'configs'
algos_config = configs_folder / 'algos_configs'


from community import Community
from environment import FlexEnv
from state import StateVars
from experiment import Experiment
from experiment_test import SimpleTests
from trainable import Trainable
from dataprocessor import YAMLParser
from utilities import ConfigsParser

#ray + gym
import ray #ray2.0 implementation
from ray.rllib.env.wrappers.multi_agent_env_compatibility import MultiAgentEnvCompatibility
from ray.tune.registry import register_env
from ray import tune
#models
from ray.rllib.models import ModelCatalog

# Custom Model
from models2 import ActionMaskModel, CCActionMaskModel
from models_rnn import LSTMActionMaskModel
ModelCatalog.register_custom_model('shift_mask', ActionMaskModel)
ModelCatalog.register_custom_model("cc_shift_mask", CCActionMaskModel)
ModelCatalog.register_custom_model("lstm_model", LSTMActionMaskModel)

#Custom functions
from models2 import ActionMaskModel, CCActionMaskModel

start_time = time.time()






#%% exp_name + get configs
exp_name=YAMLParser().load_yaml(configs_folder / 'exp_name.yaml')['exp_name']
configs=ConfigsParser(configs_folder, exp_name)

file_ag_conf, file_apps_conf, file_scene_conf, file_prob_conf,file_vars,file_experiment, ppo_config=configs.get_configs()

#%% import datafiles and agent dataprocessor
# gecad_dataset=datafolder / 'Dataset_gecad_changed.xlsx'
gecad_dataset=datafolder / 'dataset_gecad_clean.csv'
# gecad_processor=GecadDataProcessor(file_prob_conf,file_ag_conf,gecad_dataset)
# data=gecad_processor.data

#%% Trainning or debugging
train=YAMLParser().load_yaml(file_experiment)['train']

#%% Make community            
com=Community(file_ag_conf,
              file_apps_conf,
              file_scene_conf,
              file_prob_conf,
              gecad_dataset)


com_vars=StateVars(file_vars)


#%%  Make environment   
env_config={'community': com,
            'com_vars': com_vars,
            'num_agents': com.num_agents}
   
envi=FlexEnv(env_config)

df_list=envi.env_processor.get_daily_stats()
# merged=envi.env_processor.merge_df_list_on_agents(df_list)
#%%
menvi=MultiAgentEnvCompatibility(envi)
menvi._agent_ids=['ag1', 'ag2']

def env_creator(env_config):
    # return NormalizeObs(menv_base)  # return an env instance
    new_env=MultiAgentEnvCompatibility(envi)
    new_env._agents_ids=envi._agent_ids
    return new_env
    # return MultiAgentEnvCompatibility(envi)
    # return menv_base

register_env("flexenv", env_creator)

#%% experiment
if train:
    print('trainning')



    # from experiment import *
    
    
    experiment=Experiment(envi, file_experiment)
    config=experiment.make_algo_config(ppo_config)
    config_tune=experiment.make_tune_config()
    config_run=experiment.make_run_config(raylog.as_posix())
    
    resources=experiment.get_resources()
    trainable_obj=Trainable(file_experiment)
    trainable_func=trainable_obj.trainable
    trainable_resources = tune.with_resources(trainable_func, resources)
    
    spill_1=raylog / 'spill1'
    spill_2=raylog / 'spill2'
    
    ray.init(_system_config={"local_fs_capacity_threshold": 0.99,
                             "object_spilling_config": json.dumps({"type": "filesystem",
                                                                   "params": {"directory_path":[spill_1.as_posix(),
                                                                                                spill_2.as_posix()],}},)},)
    
    tuner = tune.Tuner(
          trainable_resources,
          param_space=config,
          tune_config=config_tune,
          run_config=config_run)
    
    
    #%% Train
    results=tuner.fit()
    print(results.errors)
    # # 

else:
    print('Debugging Mode')
    #%% Simple agent test
    import matplotlib.pyplot as plt
    import numpy as np
    from experiment_test import SimpleTests
    from testenv import SimpleTestEnv, DummyTester, BaselineTest, SimpleTestCycle
    dummy_tester=DummyTester(envi)
    import pandas as pd
    
    # simpletest=SimpleTestEnv(envi, dummy_tester)
    # baselinetest=BaselineTest(envi, dummy_tester)
    
    t=2
    m=pd.DataFrame()
    for k in range(0,t):
        start=[k,0]
        simpletestcycle=SimpleTestCycle(envi, dummy_tester, start)
        full_state, env_state_conc, episode_metrics, filename = simpletestcycle.test(1,[],plot=False)
        m=pd.concat([m, episode_metrics])
    
    
    m_com=m.loc['com']
    x = np.arange(t)
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(x, m_com['cost'], color='blue', label='Cost')
    ax1.set_xlabel('Time step')
    ax1.set_ylabel('Cost', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax2 = ax1.twinx()
    ax2.plot(x, m_com['reward_episode'], color='orange', label='Reward')
    ax2.set_ylabel('Reward Episode', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')
    plt.title('Cost and Reward Episode Over Time')
    plt.show()

    # full_state, env_state_conc, episode_metrics, filename = simpletest.test(1,[],plot=True)
    # full_state, env_state_conc, episode_metrics, filename = baselinetest.test(1,[],plot=True)



#%% Time
end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time:", elapsed_time, "seconds")


