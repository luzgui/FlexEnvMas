#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 16:22:15 2026

@author: omega
"""

import gymnasium  as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import random as rnd
import re
import math
from termcolor import colored
from icecream import ic


from ray.rllib.env.multi_agent_env import MultiAgentEnv, make_multi_agent

# from state_vars import *
from termcolor import colored

from utils.dataprocessor import DataProcessor, EnvDataProcessor
from utils.utilities import *

from env.reward import Reward
from env.state_update import StateUpdate 

from env.environment import FlexEnv


class FlexEnvV1(FlexEnv):
    def __init__(self,env_config):
        super().__init__(env_config)
        
    def update_all_masks(self):
        """loops over agents and updates individual masks. Returns all masks
    
        Models interruptible shiftable devices 
        """
        
        for aid in self.agents_id:

            if self.state.loc[aid]['y_s'] < self.agents_params.loc[aid]['T_prof']:
                self.mask.loc[aid] = np.ones(self.action_space.n)
    
            elif self.state.loc[aid]['y_s'] >= self.agents_params.loc[aid]['T_prof']:
                self.mask.loc[aid]=[1.0,0.0]
                
            else:
                self.mask.loc[aid] = np.ones(self.action_space.n)
            
        
        return self.mask
        
    
    
    
    
    

