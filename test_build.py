#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 16:18:40 2022

@author: omega
"""

from auxfunctions_shiftenv import *
from ray.rllib.algorithms.ppo import PPO #trainer
from shiftenvRLlib_mas import ShiftEnvMas


def make_tester(exp_name, raylog, datafolder):

    #%% Recover and load checkpoints
    #define the metric and the mode criteria for identifying the best checkpoint
    metric="_metric/episode_reward_mean"
    mode="max"
    
    # exp_name='1Ag_0512'
    # exp_name='1_Ag_new'
    # exp_name='3Oct'
    # exp_name='1Ag'
    # log_dir=raylog
    
    #get best checkpoint info
    best_checkpoint, df, best_trial = get_checkpoint(raylog, exp_name, metric, mode)
    
    #config for best trial
    best_config=best_trial.config
    
    
    #%% Generate data for testing
    
    #indezx for load and for agent is also 'ag'
    test_load_id=['ag2'] #selct new loads for testing 
    test_agents_id=['ag1'] #choose which agents are in play
    num_agents=len(test_agents_id)
    
    
    data = get_raw_data('Dataset_gecad.xlsx', datafolder)
    
    test_env_data=make_env_data_mas(data, 
                                    len(data)-1, 
                                    test_load_id, 
                                    4, 
                                    num_agents,
                                    test_agents_id)
    
    #%% Update config with test data
    # !BUG!
    #we can only update the data. not the environment
    # bug - ned to come back here and figure out how to make two different environments with different data 
    best_config['env_config']['data']=test_env_data #update data
    best_config['env_config']['env_info']='testing environment' 
    #Make the testing environment
    tenv=ShiftEnvMas(best_config['env_config'])
    
    #Instantiate and restore agent
    tester=PPO(best_config,env=best_config['env'])
    tester.restore(best_checkpoint)
    
    
    # tester_config=tester.config
    # policy=tester.get_policy()
    # policy.model.internal_model.base_model.summary()
    
    # p0=tester.get_policy('pol_ag0')
    # p1=tester.get_policy('pol_ag1')
    
    # # w0=p0.get_weights()
    # w1=p1.get_weights()
    
    # weights_file=

# m1=p1.model.internal_model.base_model.summary()
    return tenv, tester