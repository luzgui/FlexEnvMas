#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 15:46:19 2022

@author: omega
"""
from models2 import ActionMaskModel
from ray.rllib.models import ModelCatalog
ModelCatalog.register_custom_model('shift_mask', ActionMaskModel)

#PPO algorithm
from ray.rllib.algorithms.ppo import PPO #trainer
from ray.rllib.algorithms.ppo import PPOConfig #config

from shiftenvRLlib_mas import ShiftEnvMas

from ray import tune, air
# from ray.tune import Analysis
from ray.tune import analysis
from ray.tune import ExperimentAnalysis
from ray.tune import TuneConfig
from ray.tune.execution.trial_runner import _load_trial_from_checkpoint
from ray.tune.experiment import trial
from obs_wrapper import *


def make_train_config(menv):
    
    config_pol={}
    
    policies={'pol_'+aid:(None,
                       menv.observation_space,
                       menv.action_space,
                       config_pol,) for aid in menv.agents_id }
    

    
    
    #Config
    config = PPOConfig()\
                    .training(lr=1e-5,
                              num_sgd_iter=100,
                              train_batch_size=8000,
                              model={'custom_model':ActionMaskModel,
                                    'fcnet_hiddens': [128,128],
                                    'fcnet_activation':'relu',
                                    'custom_model_config': 
                                        {'fcnet_hiddens': [128,128]}})\
                    .environment(
                        env='shiftenv',           
                        observation_space=menv.observation_space,
                        action_space=menv.action_space,
                        env_config=menv.env_config)\
                    .debugging(seed=1024,log_level='WARN')\
                    .rollouts(num_rollout_workers=1)\
                    .multi_agent(policies=policies,
                                  policy_mapping_fn=policy_mapping_fn)
    
                    # .evaluation(evaluation_interval=1,
                    #             evaluation_num_workers=1,
                    #             evaluation_num_episodes=10,) 
                    # .resources(placement_strategy=tune.PlacementGroupFactory([{'CPU': 1.0}] + [{'CPU': 1.0}] * 1))
                    
    return config



def policy_mapping_fn(agent_id):
    'Policy mapping function'
    return 'pol_' + agent_id