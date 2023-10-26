#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 16:18:40 2022

@author: omega
"""

from auxfunctions_shiftenv import *
from ray.rllib.algorithms.ppo import PPO #trainer
from ray.rllib.algorithms.ppo import PPOConfig #config
from shiftenvRLlib_mas import ShiftEnvMas
from ray.tune.registry import register_env
from ray.train import Checkpoint


def make_tester(exp_name, raylog, datafolder):

    #%% Recover and load checkpoints
    #define the metric and the mode criteria for identifying the best checkpoint
    
    ## Max reward
    # metric="_metric/episode_reward_mean"
    # mode="max"
    
    ## last checkpoint (may be equivalent to min entropy)
    # metric = '_metric/training_iteration'
    
    # metric = 'training_iteration'
    metric = 'episode_reward_mean'
    # metric='entropy'
    # mode='min'    
    mode='max'
    
    # log_dir=raylog
    
    #get best checkpoint info
    best_checkpoint, df, best_trial = get_checkpoint(raylog, exp_name, metric, mode)
    
    # best_checkpoint, df, best_trial = get_checkpoint(log_dir, exp_name, metric, mode)
    
    #config for best trial
    best_config=best_trial.config
    
    env_config=best_config['env_config']
    #%% Generate data for testing
    
    #index for load and for agent is also 'ag'
    test_load_id=['ag10','ag13'] #selct new loads for testing
    #10 and 13 are from the same cluster
    num_agents=len(test_load_id)
    test_agents_id=['ag'+str(k+1) for k in range(num_agents)] #choose which agents are in play

    
    
    # data = get_raw_data('Dataset_gecad.xlsx', datafolder)
    data = get_raw_data('Dataset_gecad_changed.xlsx', datafolder.as_posix(),unit='kwh')
    
    
    # define the length of the dataset
    dt=env_config['tstep_per_day']
    t_init=0
    # t_end=t_init+(96*1)+1
    t_end=len(data) #10 days
    # H=len(data)-1
    
    # t_init=15552
    # t_end=15936
    # t_end=t_end+1
    
    

    test_env_data=make_env_data_mas(data,
                                    t_init,
                                    t_end, 
                                    test_load_id, 
                                    1, 
                                    num_agents,
                                    test_agents_id,
                                    [])
    
    #%% Update config with test data
    # !BUG!
    #we can only update the data. not the environment
    # bug - ned to come back here and figure out how to make two different environments with different data 
    best_config['env_config']['data']=test_env_data #update data
    best_config['env_config']['env_info']='testing environment'
    #save the exp_name for further use in plot and metric functions
    best_config['env_config']['exp_name']=exp_name
    
    ### try new initialization for testing purposes
    # best_config['env_config']['init_condition']='mode_window_seq'
    # best_config['env_config']['init_condition']='mode_window'
    best_config['env_config']['init_condition']='mode_window_no-repeat'
    
    
    #make config object to build
    best_config_obj=PPOConfig().from_dict(best_config)
    
    
    #this is needed because there wasd an error 
    
    # ValueError: Your desired `train_batch_size` (8000) or a value 10% off of that cannot be achieved with your other settings (num_rollout_workers=12; num_envs_per_worker=1; rollout_fragment_length=200)! Try setting `rollout_fragment_length` to 'auto' OR 666.
    best_config_obj.training(train_batch_size=1000)\
                    .rollouts(num_rollout_workers=1) 
    
    
    #Make the testing environment
    tenv=ShiftEnvMas(best_config['env_config'])
    
    def env_creator(env_config): # return an env instance
        return tenv
    register_env("shiftenv", env_creator)
    
    
    
    #Instantiate and restore agent
    tester=best_config_obj.build()
    # tester.from_checkpoint(checkpoint=best_checkpoint)
    
    # tester.from_checkpoint(checkpoint=checkpoint_dir)

    
    # tester=PPO(best_config,env=best_config['env'])
    tester.restore(best_checkpoint)
    
    
    # tester_config=tester.config
    # policy=tester.get_policy()
    # policy.model.internal_model.base_model.summary()
    
    # p0=tester.get_policy('pol_ag0')
    # p1=tester.get_policy('pol_ag1')
    
    # w1=p1.get_weights()
    
    
    # # w0=p0.get_weights()
    # w1=p1.get_weights()
    
    # weights_file=

# m1=p1.model.internal_model.base_model.summary()

    print(tenv.env_config)
    # print(tester.config.poltype)
    return tenv, tester, best_checkpoint


