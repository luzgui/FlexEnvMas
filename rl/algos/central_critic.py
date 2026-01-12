#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  9 12:57:24 2026

@author: omega


"""

import argparse
import numpy as np
from gymnasium.spaces import Discrete
import os

import ray
from ray import air, tune
from ray.rllib.algorithms.ppo.ppo import PPO, PPOConfig
from ray.rllib.algorithms.ppo.ppo_tf_policy import (
    PPOTF1Policy,
    PPOTF2Policy,
)
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.evaluation.postprocessing import compute_advantages, Postprocessing

from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.utils.tf_utils import explained_variance, make_tf_callable
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
from termcolor import colored

from rl.models.models2 import *
#debug
from icecream import ic
import pdb

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()

OPPONENT_OBS = "opponent_obs"
OPPONENT_ACTION = "opponent_action"



def cc_postprocessing(policy, 
                      sample_batch, 
                      other_agent_batches=None, 
                      episode=None):

    
    n_agents=policy.config['env_config']['num_agents']
    n_agents_other=n_agents-1 #NUmber of other agents
    
    # the observation dimension that is input to the policy model is the sum of the 'action_mask' and 'observation' (49+2=51). 
    # This is a trick to get that value that we cannot get from the config dictionary
    
    obs_space=policy.config['observation_space']
    obs_dim=0
    for key in obs_space.keys():
        obs_dim+=obs_space[key].shape[0]
        
        
    if policy.loss_initialized():


        assert other_agent_batches is not None
        opponent_batch_list = list(other_agent_batches.values())
        
        other_agents_id=list(other_agent_batches.keys())
        
        global_obs_batch = np.stack(
        [other_agent_batches[aid][2]["obs"] for aid in other_agents_id],axis=1)
        global_obs_batch=global_obs_batch.reshape((len(global_obs_batch),n_agents_other*obs_dim))
        

        ##
        # try:
        #     global_action_batch = np.stack(
        #     [other_agent_batches[aid][2]['actions'] for aid in other_agents_id],axis=1)

        # except Exception as e:
        #     print("An error occurred:", e)
        #     from ray.util import pdb
        #     pdb.set_trace() 
    
        
        # global_action_batch = np.stack(
        # [other_agent_batches[aid][2]['actions'] for aid in other_agents_id],axis=1)
        # from ray.util import pdb
        # pdb.set_trace() 
    
        sample_batch["opponent_obs"] = global_obs_batch
        # sample_batch["opponent_action"] = global_action_batch
        

        # sample_batch['vf_preds'] = convert_to_numpy(
        #     policy.compute_central_vf(
        #         sample_batch['obs'],
        #         sample_batch['opponent_obs'],
        #         sample_batch['opponent_action'],))
        sample_batch['vf_preds'] = convert_to_numpy(
            policy.compute_central_vf(
                sample_batch['obs'],
                sample_batch['opponent_obs'],))


        
    else:
        # Policy hasn't been initialized yet, use zeros.
        
        len_batch=len(sample_batch)
        zero_obs=np.zeros((len_batch,obs_dim))
        zero_opp_obs=np.zeros((len_batch,obs_dim*n_agents_other))
        # zero_opp_actions=np.zeros((len_batch,n_agents_other))
        zero_vf_preds=np.zeros((len_batch,))
        

        
        sample_batch[OPPONENT_OBS] = zero_opp_obs
        # sample_batch[OPPONENT_ACTION] = zero_opp_actions
        sample_batch[SampleBatch.VF_PREDS] = np.zeros((len_batch,))
        

        
        # sample_batch[OPPONENT_OBS] = np.zeros_like(sample_batch[SampleBatch.CUR_OBS])
        # sample_batch[OPPONENT_ACTION] = np.zeros_like(sample_batch[SampleBatch.ACTIONS])
        # sample_batch[SampleBatch.VF_PREDS] = np.zeros_like(
        #     sample_batch[SampleBatch.REWARDS], dtype=np.float32
        # )
        
        # sample_batch[[OPPONENT_ACTION]
        

    completed = sample_batch["dones"][-1]
    if completed:
        last_r = 0.0
    else:
        last_r = sample_batch[SampleBatch.VF_PREDS][-1]
    
    # print(colored('Computing advantages...','red'))
    train_batch = compute_advantages(
        sample_batch,
        last_r,
        policy.config["gamma"],
        policy.config["lambda"],
        use_gae=policy.config["use_gae"],
    )
    # print(colored('Finished postprocess...','green'))
   
    # from ray.util import pdb
    # pdb.set_trace() 
    return train_batch


def loss_with_central_critic(policy, base_policy, model, dist_class, train_batch):
    # Save original value function.
            
    vf_saved = model.value_function
    
    # Calculate loss with a custom value function.
    model.value_function = lambda: policy.model.central_value_function(
        train_batch[SampleBatch.CUR_OBS],
        train_batch[OPPONENT_OBS],)
    policy._central_value_out = model.value_function()
    loss = base_policy.loss(model, dist_class, train_batch)

    # Restore original value function.
    model.value_function = vf_saved

    return loss


def central_vf_stats(policy, train_batch):
    # Report the explained variance of the central value function.
    return {
        "vf_explained_var": explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS], policy._central_value_out
        )
    }


class CentralizedValueMixin:
    """Add method to evaluate the central value function from the model."""
    def __init__(self):
            self.compute_central_vf = self.model.central_value_function



def get_ccppo_policy(base):
    class CCPPOTFPolicy(CentralizedValueMixin, base):
        def __init__(self, observation_space, action_space, config):
            base.__init__(self, observation_space, action_space, config)
            CentralizedValueMixin.__init__(self)

        
        @override(base)
        def loss(self, model, dist_class, train_batch):
            # Use super() to get to the base PPO policy.
            # This special loss function utilizes a shared
            # value function defined on self, and the loss function
            # defined on PPO policies.


            return loss_with_central_critic(
                self, super(), model, dist_class, train_batch
            )
        
        @override(base)
        def postprocess_trajectory(
            self, sample_batch, other_agent_batches=None, episode=None
        ):
            return cc_postprocessing(
                self, sample_batch, other_agent_batches, episode
            )
        
        @override(base)
        def stats_fn(self, train_batch: SampleBatch):
            stats = super().stats_fn(train_batch)
            stats.update(central_vf_stats(self, train_batch))
            return stats

    return CCPPOTFPolicy




#------
CCPPOStaticGraphTFPolicy = get_ccppo_policy(PPOTF1Policy)
CCPPOEagerTFPolicy = get_ccppo_policy(PPOTF2Policy)

#------



class CentralizedCritic(PPO):
    @classmethod
    @override(PPO)
    def get_default_policy_class(cls, config):
        # if config["framework"] == "torch":
        #     return CCPPOTorchPolicy
        if config["framework"] == "tf":
            return CCPPOStaticGraphTFPolicy
        else:
            return CCPPOEagerTFPolicy



