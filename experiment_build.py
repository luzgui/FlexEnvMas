#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 15:46:19 2022

@author: omega
"""
from models2 import ActionMaskModel, CCActionMaskModel
from ray.rllib.models import ModelCatalog
ModelCatalog.register_custom_model('shift_mask', ActionMaskModel)
ModelCatalog.register_custom_model("cc_shift_mask", CCActionMaskModel)

#PPO algorithm
from ray.rllib.algorithms.ppo import PPO #trainer
from ray.rllib.algorithms.ppo import PPOConfig #config

from shiftenvRLlib_mas import ShiftEnvMas

from ray import tune, air
# from ray.tune import Analysis
from ray.tune import analysis
from ray.tune import ExperimentAnalysis
from ray.tune import TuneConfig
# from ray.tune.execution.trial_runner import _load_trial_from_checkpoint
from ray.tune.experiment import trial
from obs_wrapper import *

from auxfunctions_shiftenv import *
from termcolor import colored


def make_train_config(menv,pol_type):
    """
    Pol_type argument determines if there will be shared policy or each agent will have its own policy
    
    """
    
    config_pol={}
    
    if not isinstance(menv, ShiftEnvMas):
        menv=menv.env
    
    #which  policy
    if pol_type=='agent_pol':
    
        policies={'pol_'+aid:(None,
                            menv.observation_space,
                            menv.action_space,
                            config_pol,) for aid in menv.agents_id }
        
        policy_function=policy_mapping_fn
        
        
    elif pol_type=='shared_pol':

        policies={'shared_pol': (None,menv.observation_space,menv.action_space,config_pol,)}
        
        policy_function=policy_mapping_fn_shared
     
    print('Policy Type:', colored(pol_type,'red'))
        
        
    
        
    #Config
    config = PPOConfig()\
                    .training(lr=1e-5,
                              num_sgd_iter=1,
                              train_batch_size=128,
                              _enable_learner_api=False,
                              model={'custom_model':'cc_shift_mask',
                                    'fcnet_hiddens': [128,128],
                                    'fcnet_activation':'relu',
                                    'custom_model_config': 
                                        {'fcnet_hiddens': [128,128]}})\
                    .environment(
                        env='shiftenv',
                        # env=ShiftEnvMas,
                        observation_space=menv.observation_space,
                        action_space=menv.action_space,
                        env_config=menv.env_config,
                        disable_env_checking=True)\
                    .debugging(seed=1024, log_level='DEBUG')\
                    .rollouts(num_rollout_workers=0)\
                    .multi_agent(policies=policies,
                                  policy_mapping_fn=policy_function)\
                    .framework(framework='tf2',
                               eager_tracing=True)\
                    .rl_module(_enable_rl_module_api=False)    
                    # .resources(num_cpus_per_worker=1,
                    #            num_cpus_per_trainer_worker=1,
                    #            num_trainer_workers=1)\
                    # .evaluation(evaluation_interval=1,
                    #             evaluation_num_workers=1,
                    #             evaluation_num_episodes=10,) 

    # config['poltype']=pol_type #store the value in the config 
    
    
    config_tune=TuneConfig(mode='max',
                           metric='episode_reward_mean',)
    
    
    
    return config, config_tune


