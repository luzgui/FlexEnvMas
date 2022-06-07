#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 14:07:43 2022

@author: omega
"""

import ray
from ray.rllib import agents
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.utils import try_import_tf
from gym import spaces
# from or_gym.utils import create_env

tf = try_import_tf()
tf0=tf[0]
tf1=tf[1]


class ActionMaskModel(TFModelV2):
    
    def __init__(self, 
                 obs_space, 
                 action_space, 
                 num_outputs,
                 model_config, 
                 name, 
                 true_obs_shape=(32,),
                 action_embed_size=2, *args, **kwargs):
        
        super(ActionMaskModel, self).__init__(obs_space,
            action_space, num_outputs, model_config, name, 
            *args, **kwargs)
        
        self.action_embed_model = FullyConnectedNetwork(
            spaces.Box(0, 1, shape=true_obs_shape), 
                action_space, action_embed_size,
            model_config, name + "_action_embedding")
        
        self.register_variables(self.action_embed_model.variables())    
        
    def forward(self, input_dict, state, seq_lens):
        avail_actions = input_dict["obs"]["avail_actions"]
        action_mask = input_dict["obs"]["action_mask"]
        
        action_embedding, _ = self.action_embed_model({
            "obs": input_dict["obs"]["state"]})
        intent_vector = tf1.expand_dims(action_embedding, 1)
        action_logits = tf1.reduce_sum(avail_actions * intent_vector,
            axis=1)
        inf_mask = tf1.maximum(tf0.log(action_mask), tf1.float32.min)
        return action_logits + inf_mask, state   
    
    def value_function(self):
        return self.action_embed_model.value_function()
    
    


