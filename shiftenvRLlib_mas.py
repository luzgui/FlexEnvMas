import gym
from gym import spaces
import numpy as np
import random as rnd
import re


from ray.rllib.env.multi_agent_env import MultiAgentEnv, make_multi_agent

from shiftenvRLlib import ShiftEnv


class ShiftEnvMas(ShiftEnv,MultiAgentEnv):
    def __init__(self,config):
        super().__init__(config)
        self.num_agents=config['num_agents']
        
        self.agents_id=['ag'+str(k) for k in range(config['num_agents'])]
    
    def reset(self):
        print('implememtar reset')
        
    def step():
        print('implememtar step')
        
        


