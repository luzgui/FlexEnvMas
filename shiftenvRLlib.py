#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 11:31:14 2022

@author: omega
"""
# import math
# from numpy import Inf
import gym
from gym import spaces
import numpy as np
import random as rnd
import re
# import time
# from collections import OrderedDict


class ShiftEnv(gym.Env):
    
    
    """
    A custom OpenAI Gym environment for managing shiftable loads
    
    The objective is to schedulle a cyclic shiftable appliance.
    The environment has the follwoing dynamics:
        - the episode starts at a given timestep
        - Tprsdhe agent observes the state and takes an action (whether to turn ON or not the appliance IN THE PRESENT STATE)
            - Due to the using of action masking we are able to guarantee that once the appliance is connected it maintains connceted for the required number of timesteps
        - It recieves a reward based on the current state 
        - Evolves into next state
        
    
    
    """

    def __init__(self, config):

        self.reward_type=config["reward_type"]
        self.tar_type=config['tar_type']
        self.data=config["data"] # We need to import the unflexible load and PV production.
        # data is an array with load in column 0 and pv production in column 1
        
        setattr(self,'coisa',4)     
       
        
        self.tstep_size=config["step_size"]
        self.T=len(self.data) # Time horizon
        self.Tw=config["window_size"] #window horizon
        self.dh=self.tstep_size*(1/60.0) # Conversion factor energy-power
        self.tstep_init=0 #initial timestep in each episode
        self.t_ahead=4 #number of timeslots that each actual timeslot is loloking ahead (used in update_forecast) 
        
        #Appliance profile
        
        self.profile=config["profile"]
        self.T_prof=len(self.profile)
        self.E_prof=self.profile.sum()*self.dh #energy needed for the appliance
        self.R=0
        self.R_Total=[] # A way to see the evolution of the rewards as the model is being trained
        self.n_episodes=0
        self.hist=[]
        
        #new vars
        self.L_s=np.zeros(self.T) # we make a vector of zeros to store the shiftable load profile
        # self.l_s=self.L_s[0] 
        
        #initialize with the first element in 
        self.t_shift=0
        
        
        #this must be set as a moving variable that updates the deliver time according to the day
        self.t_deliver=config["time_deliver"]
        
        self.min_max=self.data['minutes'].max()
        

        # self.tar_buy=0.17 # import tariff in €/kWh
        # self.tar_sell=0.0 # remuneration for excess
                
        
        # defining the state variables
        self.state_vars={'tstep':
                                {'max':10000,'min':0},
                        'minutes':
                                {'max':1440,'min':0},
                        'sin':
                                {'max':1.0,'min':-1.0},
                        'cos':
                                {'max':1.0,'min':-1.0},
                        'gen0':# g : PV generation at timeslot
                                {'max':10.0,'min':0},
                        'gen1':# g : PV generation forecast next timeslot
                                {'max':10.0,'min':0},
                        'gen2':# g : PV generation forecast 1h ahead
                                {'max':10.0,'min':0},
                        'gen3':# g : PV generation forecast 6h ahead
                                {'max':10.0,'min':0},
                        'gen4':# g : PV generation forecast 12h ahead
                                {'max':10.0,'min':0},
                        'gen5':# g : PV generation forecast 24h ahead
                                {'max':10.0,'min':0},
                        'gen6':# g : PV generation forecast next timeslot
                                {'max':10.0,'min':0},
                        'gen7':# g : PV generation forecast 1h ahead
                                {'max':10.0,'min':0},
                        'gen8':# g : PV generation forecast 6h ahead
                                {'max':10.0,'min':0},
                        'gen9':# g : PV generation forecast 12h ahead
                                {'max':10.0,'min':0},
                        'gen10':# g : PV generation forecast 24h ahead
                                {'max':10.0,'min':0},
                        'gen11':# g : PV generation forecast 12h ahead
                                {'max':10.0,'min':0},
                        'gen12':# g : PV generation forecast 24h ahead
                                {'max':10.0,'min':0},
                        'load0':# g : PV generation at timeslot
                                {'max':10.0,'min':0},
                        'load1':# g : PV generation forecast next timeslot
                                {'max':10.0,'min':0},
                        'load2':# g : PV loaderation forecast 1h ahead
                                {'max':10.0,'min':0},
                        'load3':# g : PV loaderation forecast 6h ahead
                                {'max':10.0,'min':0},
                        'load4':# g : PV loaderation forecast 12h ahead
                                {'max':10.0,'min':0},
                        'load5':# g : PV loaderation forecast 24h ahead
                                {'max':10.0,'min':0},
                        'load6':# g : PV loaderation forecast next timeslot
                                {'max':10.0,'min':0},
                        'load7':# g : PV loaderation forecast 1h ahead
                                {'max':10.0,'min':0},
                        'load8':# g : PV loaderation forecast 6h ahead
                                {'max':10.0,'min':0},
                        'load9':# g : PV loaderation forecast 12h ahead
                                {'max':10.0,'min':0},
                        'load10':# g : PV loaderation forecast 24h ahead
                                {'max':10.0,'min':0},
                        'load11':# g : PV loaderation forecast 12h ahead
                                {'max':10.0,'min':0},
                        'load12':# g : PV generation forecast 24h ahead
                                {'max':10.0,'min':0},
                        'delta0':# g : PV generation at timeslot
                                {'max':10.0,'min':-10.0},
                        'delta1':# g : PV generation forecast next timeslot
                                {'max':10.0,'min':-10.0},
                        'delta2':# g : PV deltaeration forecast 1h ahead
                                {'max':10.0,'min':-10.0},
                        'delta3':# g : PV deltaeration forecast 6h ahead
                                {'max':10.0,'min':-10.0},
                        'delta4':# g : PV deltaeration forecast 12h ahead
                                {'max':10.0,'min':-10.0},
                        'delta5':# g : PV deltaeration forecast 24h ahead
                                {'max':10.0,'min':-10.0},
                        'delta6':# g : PV deltaeration forecast next timeslot
                                {'max':10.0,'min':-10.0},
                        'delta7':# g : PV deltaeration forecast 1h ahead
                                {'max':10.0,'min':-10.0},
                        'delta8':# g : PV deltaeration forecast 6h ahead
                                {'max':10.0,'min':-10.0},
                        'delta9':# g : PV deltaeration forecast 12h ahead
                                {'max':10.0,'min':-10.0},
                        'delta10':# g : PV deltaeration forecast 24h ahead
                                {'max':10.0,'min':-10.0},
                        'delta11':# g : PV deltaeration forecast 12h ahead
                                {'max':10.0,'min':-10.0},
                        'delta12':# g : PV generation forecast 24h ahead
                                {'max':10.0,'min':-10.0},
                        'excess0':# g : PV generation at timeslot
                                {'max':10.0,'min':-10.0},
                        'excess1':# g : PV generation forecast next timeslot
                                {'max':10.0,'min':-10.0},
                        'excess2':# g : PV excesseration forecast 1h ahead
                                {'max':10.0,'min':-10.0},
                        'excess3':# g : PV excesseration forecast 6h ahead
                                {'max':10.0,'min':-10.0},
                        'excess4':# g : PV excesseration forecast 12h ahead
                                {'max':10.0,'min':-10.0},
                        'excess5':# g : PV excesseration forecast 24h ahead
                                {'max':10.0,'min':-10.0},
                        'excess6':# g : PV excesseration forecast next timeslot
                                {'max':10.0,'min':-10.0},
                        'excess7':# g : PV excesseration forecast 1h ahead
                                {'max':10.0,'min':-10.0},
                        'excess8':# g : PV excesseration forecast 6h ahead
                                {'max':10.0,'min':-10.0},
                        'excess9':# g : PV excesseration forecast 12h ahead
                                {'max':10.0,'min':-10.0},
                        'excess10':# g : PV excesseration forecast 24h ahead
                                {'max':10.0,'min':-10.0},
                        'excess11':# g : PV excesseration forecast 12h ahead
                                {'max':10.0,'min':-10.0},
                        'excess12':# g : PV generation forecast 24h ahead
                                {'max':10.0,'min':-10.0},
                
                        
                                

                        # 'load_s': #l_s: shiftable load 
                        #     {'max':max(self.profile),'min':0},
                        # 'delta_s': #The differential betwee gen and l_s
                        #     {'max':10,'min':-10.0},
                        # 'delta_c': #The differential betwee gen and load + l_s
                        #         {'max':10,'min':-10.0},
                        'y': # =1 if ON at t, 0 OTW
                            {'max':1.0,'min':0.0},
                        'y_1': # =1 if ON in t-1
                            {'max':1.0,'min':0.0},
                        'y_s':  # +1 if app is schedulled at t (incremental) 
                                #(how many times it was connected)
                            {'max':self.T,'min':0.0},
                        # 'cost':
                        #     {'max':100.0,'min':-100.0},
                        # 'cost_s':
                        #     {'max':100.0,'min':-100.0},
                        # 'cost_s_x':#cost of supply shiftable load using PV excess
                        #     {'max':100.0,'min':-100.0},
                        'tar_buy':
                            {'max':1,'min':0},
                        'tar_buy0': #Tariff at the next timestep
                            {'max':1,'min':0},
                        'E_prof': # reamining energy to supply appliance energy need
                            {'max':2*self.E_prof,'min':-2*self.E_prof}}
                        # 'excess': #PV excess affter supplying baseload (self.load)}
                        #     {'max':10,'min':-10.0},}
                            
                    
        
            
        
        self.var_class={'gen','load','delta','excess'}
        
        self.obs=None   
            
        #Number of variables to be used
        self.var_dim=len(self.state_vars.keys())
                
        
                
        self.highlim=np.array([value['max'] for key, value in self.state_vars.items()])
        self.lowlim=np.array([value['min'] for key, value in self.state_vars.items()])
            
        
        #Actions
        self.action_space = spaces.Discrete(2) # ON/OFF
        
        # Observation space      
        self.observation_space = spaces.Dict({
            "action_mask": spaces.Box(0.0, 1.0, shape=(self.action_space.n,)),
            "observations": spaces.Box(low=np.float32(self.lowlim), high=np.float32(self.highlim), shape=(self.var_dim,))})
        
                
        #Training/testing termination condition
        
        self.done_cond=config['done_condition'] # window or horizon mode
        self.init_cond=config['init_condition'] 
        
        
        # app conection counter
        self.count=0


    def step(self, action):
        # print('action', action)
        # self.done=False
        """
        Runs one time-step of the environment's dynamics. The reset() method is called at the end of every episode
        :param action: The action to be executed in the environment
        :return: (observation, reward, done, info)
            observation (object):
                Observation from the environment at the current time-step
            reward (float):
                Reward from the environment due to the previous action performed
            done (bool):
                a boolean, indicating whether the episode has ended
            info (dict):
                a dictionary containing additional information about the previous action
        """

        #Qual o significado de action:
                # - ação no timeslot atual?: nesse caso necessito de corrigir o state e so depois avançar com o tempo
                # - ação para o timeslot seguint
        self.action=float(action)
        
        reward=self.get_reward() #get reward value
        
        #accumulated total cost
        self.c_T+=self.cost_s

        #A function to populate 
        
        
        
        self.check_term() #check weather to end or not the episode
        
        self.tstep+=1 # update timestep
        self.state_update() #update state variables  
        
        
        
        # print(reward)

        self.R+=reward
        self.r=reward
        

        
    
        self.obs={"action_mask": self.get_mask(),
              "observations": self.get_obs()
              }
        


        return self.obs, reward, self.done, {}
    




    def reset(self):
        """
        Reset the environment state and returns an initial observation
        
        Returns:
        -------
        observation (object): The initial observation for the new episode after reset
        :return:
        """
        
    
        self.done=False
        
        
        self.tstep=self.get_init_tstep()
        
        
        self.tstep_init=self.tstep # initial timestep
        # print(self.tstep)
        
        # self.tar_buy=0.17
        self.L_s=np.zeros(self.T)

        # self.gen=self.data[self.tstep,0]
        # self.load=self.data[self.tstep,1]
        # self.minutes=self.data[self.tstep,2]
        self.minutes=self.data.iloc[self.tstep]['minutes']
        
        
       
        
        
        self.sin=np.sin(2*np.pi*(self.minutes/self.min_max))
        self.cos=np.cos(2*np.pi*(self.minutes/self.min_max))
        
        self.load_s=0   
        
        
        #update forecasts
        self.update_forecast()
        
        
        self.get_tariffs() #update tarrifs
        self.get_tariffs0() #update tarifs from next timestep
        
        
        self.R=0
        self.r=0
        
        self.c_T=0
        
        self.delta_c=(self.load0+self.load_s)-self.gen0
        self.delta_s=self.load_s-self.gen0
        
        # self.load_s=self.L_s[0] #initialize with the first element in L_s
        self.load_s=0 #machine starts diconected
        
        
        # if the random initial time step is after the delivery time it must not turn on
        # if self.minutes >= self.t_deliver-self.T_prof*self.tstep_size:
        #     self.E_prof=0 #means that there is no more energy to consume 
        #     self.y_s=self.T_prof #means taht it connected allready the machine T_prof times
        # else:
        #     self.E_prof=self.profile.sum()*self.dh #energy needed for the appliance
        #     self.y_s=0 # means that it has never connected the machine
        
        self.E_prof=self.profile.sum()
        self.y_s=0
        
        
        #inititialize binary variables
        self.y=0
        self.y_1=0
        # self.y_s=self.y
        self.t_shift=0
        
        self.hist=[]
        self.hist.append(self.y)
        

        
        
        (self.load0+self.load_s)-self.gen0
        self.cost=max(0,self.delta_c)*self.tar_buy + min(0,self.delta_c)*self.tar_sell
        self.cost_s=max(0,self.delta_s)*self.tar_buy + min(0,self.delta_s)*self.tar_sell
        
        self.excess=max(0,-self.delta0)   
        self.cost_s_x=max(0,self.load_s-self.excess0)*self.tar_buy + min(0,self.load_s-self.excess0)*self.tar_sell
        
        
        
        self.obs={"action_mask": np.array([1,1]), # we concede full freedom for the appliance 
              "observations": self.get_obs()
              }
    
        return self.obs




    def get_reward(self):
        
            if self.reward_type == 'simple_cost':

                # reward=np.exp(-(self.cost_s**2)/0.01)+np.exp(-(((self.y_s-self.T_prof)**2)/0.001))
                # The reward should be function of the action
                if self.minutes == self.min_max-self.T_prof*self.tstep_size and self.y_s!=self.T_prof:
                    reward=-1
                
                else:
                    # reward=np.exp(-(self.cost**2)/0.001)-0.5                                           
                    # reward=-self.cost*self.delta
                    # reward=-10*self.cost_s*self.delta
                    
                    reward=-10*max(0,((self.load+self.action*0.3)-self.gen))*self.tar_buy
                    
            
                    
            # :::::::::::::::::::::::
            if self.reward_type == 'excess_cost':

                # reward=np.exp(-(self.cost_s**2)/0.01)+np.exp(-(((self.y_s-self.T_prof)**2)/0.001))
                # The reward should be function of the action
                if self.minutes == self.min_max-self.T_prof*self.tstep_size and self.y_s!=self.T_prof:
                    reward=-1
                
                else:
                    # reward=np.exp(-(self.cost**2)/0.001)-0.5                                           
                    # reward=-self.cost*self.delta
                    # reward=(-10*max(0,((self.action*0.3)-self.excess))*self.tar_buy + 0.1*self.excess)*self.action
                    
                    reward=-((self.action*self.profile[0]-self.excess0)*self.tar_buy)*self.action
                    
                    
            # :::::::::::::::::::::::
            if self.reward_type == 'excess_cost_2':

                # reward=np.exp(-(self.cost_s**2)/0.01)+np.exp(-(((self.y_s-self.T_prof)**2)/0.001))
                # The reward should be function of the action
                if self.minutes == self.min_max-self.T_prof*self.tstep_size and self.y_s!=self.T_prof:
                    reward=-1
                
                else:
                    # reward=np.exp(-(self.cost**2)/0.001)-0.5                                           
                    # reward=-self.cost*self.delta
                    # reward=(-10*max(0,((self.action*0.3)-self.excess))*self.tar_buy + 0.1*self.excess)*self.action
                    
                    reward=-(self.action*self.profile[0]-self.excess0)*self.tar_buy
                       
                        
            # :::::::::::::::::::::::
            if self.reward_type == 'excess_cost_3':
                if self.minutes == self.min_max-self.T_prof*self.tstep_size and self.y_s!=self.T_prof:
                    reward=-1
                else:
                    forcast_sum=self.excess1+self.excess2+self.excess3+self.excess4+self.excess5+self.excess6+self.excess7+self.excess8+self.excess9+self.excess10+self.excess11+self.excess12
                    print(forcast_sum)
                    if forcast_sum < self.E_prof: #there is a forecasted excess smaller that the appliance needs
                        reward=-(((self.action*self.profile[0]-self.excess0)*self.tar_buy))*self.action + 1 
                    else:
                        reward=-(((self.action*self.profile[0]-self.excess0)*self.tar_buy))*self.action
                        
 
            
            
            # :::::::::::::::::::::::
            elif self.reward_type == 'next_time_cost':
                
                if self.action ==1:
                    load_shiftable=0.3
                    #If the agent decides to turn ON at the next timestep then it will pay the cost of being  ON at the next timestep
                    
                if self.action == 0: #The agent decides to be turned OFF
                    load_shiftable=0
                    
                
                reward=-(max(0,self.delta0+load_shiftable)*self.tar_buy0*self.dh + min(0,self.delta0+load_shiftable)*self.tar_sell*self.dh)
                    
                
                # :::::::::::::::::::::::
                    
            elif self.reward_type== 'shift_cost':
                
                if self.minutes == self.min_max-self.T_prof*self.tstep_size and self.y_s!=self.T_prof:
                    reward=-1
                else:
                
                    reward=-self.cost_s_x


                
          # :::::::::::::::::::::::
            elif self.reward_type== 'next_shift_cost':
                
                if self.minutes == self.min_max-self.T_prof*self.tstep_size and self.y_s!=self.T_prof:
                    reward=-10
                else:
                
                    if self.action==1:
                        load_shiftable=0.3
                    elif self.action == 0: #The agent decides to be turned OFF
                        load_shiftable=0
                    
                        
                    excess=max(0,-self.delta0)
                    reward=-(max(0,load_shiftable-excess)*self.tar_buy*self.dh + min(0,load_shiftable-excess)*self.tar_sell*self.dh)
                   
                    
                   
                    
          # :::::::::::::::::::::::::::::::::
              
            elif self.reward_type== 'gauss_shift_cost':
                
                if self.minutes == self.min_max-self.T_prof*self.tstep_size and self.y_s!=self.T_prof:
                    reward=-1
                else:
                
                    reward=np.exp(-(self.cost_s_x**2)/0.01)
                    reward=reward/self.Tw


            # self.delta_c=(self.load+self.load_s)-self.gen
            # self.cost=max(0,self.delta_c)*self.tar_buy*self.dh + min(0,self.delta_c)*self.tar_sell*self.dh

                                           
            # reward=np.exp(-(self.cost_s**2)/0.001)-0.5                                           
        
            # if self.g!=0 and self.y=1 :
                
            # a=0.7
            # b=0.3
            # reward=a*np.exp(-(self.cost_s**2)/0.001)+b*np.exp(-((self.y_s-1)**2)/0.001)
            
            # if self.y==self.y_1:
            # reward=np.exp(-(self.cost_s**2)/0.001)+np.exp(-((self.y_s-self.T_prof)**2)/0.001)+np.exp(-((self.y-self.y_1)**2)/0.001)
                
                
            # else:
                # reward=-1
            # reward=-self.cost_s-(-np.exp(-((self.y_s-1)**2)/0.001)
            
            # reward=-self.cost_s-10*self.E_prof
            # if self.tstep == self.T-1 and self.y_s!=1: # if we arrived at end without turning on
            #     reward=-100
            # else:
            # if     
            
            # a=1
            # b=0.5
            # c=0.01
            # # reward=-(a*self.cost_s)-b*(self.y-self.y_1)-c*(abs(self.y_s-self.T_prof))
            # reward= 10*self.cost_s*self.gen*action
            
            
            
            # if (self.tstep >= self.t_deliver-self.T_prof and self.y_s < self.T_prof) or (self.y_s > self.T_prof) or (self.y_s > 0 and self.y_s < self.T_prof and self.y==0):
                    
            # if (self.minutes >= self.t_deliver-self.T_prof*self.tstep_size and self.minutes <= self.min_max and self.y_s < self.T_prof) or (self.y_s > self.T_prof) or (self.y_s > 0 and self.y_s < self.T_prof and self.y==0):
            #     # reward=-1/self.T
            #     reward=-1
            # else:
            #     reward= -self.cost_s*self.y-0.1/self.Tw*(abs(self.y-self.y_1))
                
            # if (self.minutes >= self.t_deliver-self.T_prof*self.tstep_size and self.minutes <= self.min_max and self.y_s < self.T_prof) or (self.y_s > self.T_prof) or (self.y_s > 0 and self.y_s < self.T_prof and self.y==0):
            #     # reward=-1/self.T
            #     reward=-1
            # else:
            #     reward= -self.cost_s

            
            # if self.tstep == self.T-1:
            #     reward=-self.c_T-((self.y_s-self.T_prof)**2)
            #     print('r', reward)
            # else:
            #     reward=0
            
            # print('tstep',self.tstep)
            # if self.tstep < self.T:
            #     reward=0
            # elif self.tstep == self.T:
            #     reward=-self.c_T-(self.ys-self.T_prof)
            #     print('aqui')
            
            
            # if self.y_s-self.T_prof!=0:
            #     reward=-1
            # else:    
            #     reward=-self.c_T
            
            
            
            # if self.y_s <= 1: #we only want to connect the device once
                    # reward=-self.c # reward is the cost
                    # reward=np.exp(-(self.c**2)/0.001)
            # else: 
                # reward=-1
                
                
            
            return reward


    def render(self, mode='human', close=False):
        """
        :param mode:
        :return:
        """
        return
    

    def get_obs(self):
        return np.array((self.tstep,
                         self.minutes,
                         self.sin,
                         self.cos,
                         self.gen0,
                         self.gen1,
                         self.gen2,
                         self.gen3,
                         self.gen4,
                         self.gen5,
                         self.gen6,
                         self.gen7,
                         self.gen8,
                         self.gen9,
                         self.gen10,
                         self.gen11,
                         self.gen12,
                         self.load0,
                         self.load1,
                         self.load2,
                         self.load3,
                         self.load4,
                         self.load5,
                         self.load6,
                         self.load7,
                         self.load8,
                         self.load9,
                         self.load10,
                         self.load11,
                         self.load12,
                         self.delta0,
                         self.delta1,
                         self.delta2,
                         self.delta3,
                         self.delta4,
                         self.delta5,
                         self.delta6,
                         self.delta7,
                         self.delta8,
                         self.delta9,
                         self.delta10,
                         self.delta11,
                         self.delta12,
                         self.excess0,
                         self.excess1,
                         self.excess2,
                         self.excess3,
                         self.excess4,
                         self.excess5,
                         self.excess6,
                         self.excess7,
                         self.excess8,
                         self.excess9,
                         self.excess10,
                         self.excess11,
                         self.excess12,
                         self.y,
                         self.y_1,
                         self.y_s,
                         self.tar_buy,
                         self.tar_buy0,
                         self.E_prof), 
                         dtype=np.float)
    
    def get_full_obs(self):
        return np.array((self.tstep,
                         self.minutes,
                         self.sin,
                         self.cos,
                         self.gen,
                         self.gen0,
                         self.gen1,
                         self.gen6,
                         self.gen12,
                         self.gen24,
                         self.load,
                         self.load0,
                         self.load1,
                         self.load6,
                         self.load12,
                         self.load24,
                         self.delta,
                         self.delta0,
                         self.delta1,
                         self.delta6,
                         self.delta12,
                         self.delta24,
                         self.load_s,
                         self.delta_s,
                         self.delta_c,
                         self.y,
                         self.y_1,
                         self.y_s,
                         self.cost,
                         self.cost_s,
                         self.cost_s_x,
                         self.tar_buy,
                         self.tar_buy0,
                         self.E_prof,
                         self.excess,), 
                         dtype=np.float)
    
      
    def get_term_cond(self):
        "A function to get the correct episiode termination condition according to weather it is in window or horizon mode defined in the self.done_cond variable"
        
        if self.done_cond == 'mode_window': # episode ends when when Tw timeslots passed
            return self.tstep_init+self.Tw-1
        elif self.done_cond == 'mode_horizon': #episode ends in the end of the data length
            return self.T
     
           
    def get_init_tstep(self):
        "A function that returns the initial tstep of the episode"
        if self.init_cond == 'mode_window':
            
            # t=rnd.randrange(0, self.T-self.Tw-1) # a random initial state in the whole year
            t=rnd.choice([k*self.Tw for k in range(int((self.T/self.Tw)-1))]) # we allways start at the beggining of the day and advance Tw timesteps but choose randomly what day we start
            assert self.data.iloc[t]['minutes']==0, 'initial timeslot not 0'
            
            return t
            
        elif self.init_cond=='mode_random':
            t=rnd.randrange(0, self.T-self.Tw-1)
            return t
        
        
        elif self.init_cond == 'mode_horizon': 
            #episode starts at t=0
            return 0
        
        
    def get_mask(self):
        
        # mask = np.ones(self.action_space.n)
        # obs=self.get_obs()
        
        if self.action==1 and self.y_s < self.T_prof:
            mask=np.array([0,1])
        # else:
        #     mask = np.ones(self.action_space.n)
            
        elif self.y_s >= self.T_prof:
            mask=np.array([1,0])
            
        else:
            mask = np.ones(self.action_space.n)
        
        # if self.action==1 and self.count <= self.T_prof :
        #     self.count+=1
        #     mask=np.array([0,1])
        # # else:
        # #     mask = np.ones(self.action_space.n)
            
        # elif self.action==0:
        #     self.count=0
        #     mask = np.ones(self.action_space.n)
            

        return mask
        
        
        
    def check_term(self):
        if self.tstep==self.get_term_cond(): 
            self.R_Total.append(self.R)
            # print('sparse', self.R)
            print(self.R)

            # print('tstep_init',self.tstep_init)
            # print('tstep',self.tstep)

            self.n_episodes+=1
            
            
            self.obs={"action_mask": self.get_mask(),
                  "observations": self.get_obs()
                  }
        

        
            reward=self.get_reward()
            
            self.done = True
            
            return self.obs, reward, self.done, {'episode has ended'}
        
        else:
            self.done = False
        



    def state_update(self):
        
        #Variables update
        
        self.y=self.action # what is the ON/OFF state of appliance
        

        self.get_tariffs() #update tariffs
        self.get_tariffs0() #update tarifs from next timestep
        
        #update forecasts
        self.update_forecast()
        
        
        

        # ##
        self.minutes=self.data.iloc[self.tstep]['minutes']
        self.sin=np.sin(2*np.pi*(self.minutes/self.min_max))
        self.cos=np.cos(2*np.pi*(self.minutes/self.min_max))
        
        
        # note to consider: shiftable load is allways zero unless the appliance is activated
        
        
        # Binary variables
        
        
         # how many times the machine have been turned on

        self.hist.append(self.action)
        if self.tstep > self.tstep_init+1:
            # print(self.step)
            self.y_1=self.hist[-2] #penultiumate element to get the previous action
        else:
            self.y_1=0
            
        #Load the shiftable profile [H1]
        
        # if self.y_s == 1:
        #     self.t_shift=0 # a counter for the shiftable load profile
        #     if self.t_shift < self.T_prof: #while counter is less than the duration of the cycle
        #         self.l_s=self.profile[self.t_shift]
        #         self.t_shift=+1
        
        # elif self.y_s != 1:
            
        
        
        #Load control [H3]
        
        
        # if self.tstep >= self.t_deliver-self.T_prof and self.y_s < self.T_prof: # We need to impose that it  must start
        #     self.y=1
        
        # if self.y_s > self.T_prof and action == 1:
        #     self.y=0
        
        
        # if self.tstep <= self.T-self.T_prof:
        #     if self.y == 1: #if it must be turned ON on the present tslot
        #         if self.y_s==0: #if its never been ON
        #             # self.t_shift=0
        #             self.load_s=self.y*self.profile[self.t_shift]
                
        #         if self.y_s!=0 and self.t_shift < self.T_prof-1:
        #             self.t_shift+=1
        #             self.load_s=self.y*self.profile[self.t_shift]
                
        #         # if self.y_s > self.T_prof:
                    
                
        #     elif self.y == 0:
        #         self.load_s = 0
                
                # self.E_prof == self.profile.sum()*self.dh
        
        
        #garantee that it follows the machine profile
        # if self.tstep <= self.T-self.T_prof:
        #     if self.y == 1: #if it must be turned ON on the present tslot
        #         if self.y_s==0: #if its never been ON
        #             # self.t_shift=0
        #             self.load_s=self.y*self.profile[self.t_shift]
                
        #         if self.y_s!=0 and self.t_shift < self.T_prof-1:
        #             self.t_shift+=1
        #             self.load_s=self.y*self.profile[self.t_shift]
                
        #         if self.y_s == self.T_prof:
        #             self.t_shift=0
        #             self.load_s=self.y*self.profile[self.t_shift]
                    
                    
                
        #     elif self.y == 0:
        #         self.load_s = 0
         
        
        
        #convert action to power (constant profile)
        self.load_s=self.action*self.profile[0]
        
        
        self.y_s+=self.action
        if self.minutes ==0: #the number of connected timeslots resets when a new day starts
            self.t_shift=0
            self.y_s=0
            
            
            # print('L_s',self.L_s)
            # print('L_s2', self.L_s[self.t:self.t+self.T_prof])
            # print('prof', self.profile)
            # self.L_s[self.t:self.t+self.T_prof]=self.profile # we load the shiftable profile from t onwards
            
            # self.L_s=np.zeros(self.T)
            # if self.tstep <= self.T-self.T_prof:
            #     for k in range(self.T_prof):
            #         self.L_s[self.tstep+k]=self.profile[k]
            #         self.load_s=self.L_s[self.tstep] 
            
        # if action == 0:
        #     self.load_s=self.L_s[self.tstep] 
            
            # self.L_s[self.t:][:self.T_prof]=self.profile
            # #what if the appliance is turned on several times??
        
        #update remaining energy that needs to be consumed
        
        # if self.E_prof < self.profile[0]: #correct numerical imprecisions
        #     self.E_prof=0.0   
        if self.minutes==0: #It restarts in the beggining of the day 
            self.E_prof=self.profile.sum()

        self.E_prof-=self.action*self.profile[0]

        # print('Eprof',self.E_prof)
        
        #restart t_shift for each new day so that appliances may be schedulled again

        
        
        
        #deficit vs excess
        self.delta_c = (self.load0+self.load_s)-self.gen0 # if positive there is imports from the grid. If negative there are exports to the grid 
        
        self.delta_s=self.load_s-self.gen0
        

        #energy cost
        
        # cost considering the full load
        self.cost=max(0,self.delta_c)*self.tar_buy + min(0,self.delta_c)*self.tar_sell
        #cost considering only the appliance
        self.cost_s=max(0,self.delta_s)*self.tar_buy + min(0,self.delta_s)*self.tar_sell
        
        
        self.excess=max(0,-self.delta0)   
        self.cost_s_x=max(0,self.load_s-self.excess0)*self.tar_buy + min(0,self.load_s-self.excess0)*self.tar_sell
        
        

    def update_forecast(self):
        for var in self.var_class:
            var_keys=[key for key in self.state_vars.keys() if var in key]
            
            for k in var_keys:
                t=re.findall(r"\d+", k)
                # print(var)
                # print(t)
                t=int(t[0])
                
                if self.tstep+self.t_ahead*t >= self.T: #if forecast values fall outside horizon
                    setattr(self,k, 0)
                else:
                    # print('OLHA AQUI SOCIO!!!!',self.tstep+t,var)
                    # What is happening: the "number" in the variable name (example: gen2, gen3 ) is used as the time (t) that is incremented in the current timestep (self.tstep). We can multiply t*4 to make it span 24 hours with 2h intervals
                    setattr(self,k, self.data.iloc[self.tstep+self.t_ahead*t][var] )
                
                
            # s=shiftenv
            # for var in s.var_class:
            #     var_keys=[key for key in s.state_vars.keys() if var in key]
            #     print(var_keys)
                
            #     for k in var_keys:
            #         print(k)
            #         t=re.findall(r"\d+", k)
            #         print(t)
            #         t=int(t[0])
            #         setattr(s,k, s.data.iloc[s.tstep+2*t][var])
        
        
        
    def get_tariffs(self):
        "Define tarrifs in €/kWh -- define here any function for defining tarrifs"
        # Tarifa bi-horaria
        # if self.minutes >= 240 and self.minutes <=540:
        #Tarifa bi-horaria
        
        if  self.tar_type=='bi':
        
            if self.minutes >= self.tstep_size*2*8 and self.minutes <=self.tstep_size*2*22:
                self.tar_buy=0.1393
            else:
                self.tar_buy=0.0615
            # self.tar_buy=0.17*(1-(self.gen/1.764)) #PV indexed tariff 
                
            self.tar_sell=0.0 # remuneration for excess production
        
        elif self.tar_type=='flat':
            self.tar_buy=0.10
            self.tar_sell=0.0
        
        
    def get_tariffs0(self):
        "Define tarrifs in €/kWh for the next timestep-- define here any function for defining tarrifs"
        if self.minutes+self.tstep_size>= self.tstep_size*2*8 and self.minutes+self.tstep_size <=self.tstep_size*2*22:
            self.tar_buy0=0.1393
        else:
            self.tar_buy0=0.0615
        # self.tar_buy=0.17*(1-(self.gen/1.764)) #PV indexed tariff 
            
        self.tar_sell0=0.0 # remuneration for excess production
    
        
        
        
        
        