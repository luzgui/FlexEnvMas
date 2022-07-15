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
# import time
# from collections import OrderedDict


class ShiftEnv(gym.Env):
    
    
    """
    A custom OpenAI Gym environment for managing shiftable loads
    
    The objective is to schedulle a cyclic shiftable appliance.
    The environment has the follwoing dynamics:
        - the episode starts at a given timestep
        - The agent observes the state and takes an action (whether to turn ON or not the appliance IN THE NEXT STATE)
            - Due to the using of action masking we are able to guarantee that once the appliance is connected it maintains connceted for the required number of timesteps
        - It recieves a reward based on the current state 
        - Evolves into next state
        
    
    
    """

    def __init__(self, config):

        self.reward_type=config["reward_type"]
        self.data=config["data"] # We need to import the unflexible load and PV production.
        # data is an array with load in column 0 and pv production in column 1
        
       
        
        self.tstep_size=config["step_size"]
        self.T=len(self.data) # Time horizon
        self.Tw=config["window_size"] #window horizon
        self.dh=self.tstep_size*(1/60.0) # Conversion factor energy-power
        self.tstep_init=0 #initial timestep in each episode
        
        
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
        
        self.min_max=max(self.data[:,2])
        

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
                        'gen':# g : PV generation at timeslot
                            {'max':10,'min':0},
                        'gen0':# g : PV generation forecast next timeslot
                                {'max':10,'min':0},
                        'gen1':# g : PV generation forecast 1h ahead
                                 {'max':10,'min':0},
                        'gen6':# g : PV generation forecast 6h ahead
                                 {'max':10,'min':0},
                        'gen12':# g : PV generation forecast 12h ahead
                                 {'max':10,'min':0},
                        'gen24':# g : PV generation forecast 24h ahead
                                 {'max':10,'min':0},  
                        'load':# l : Load at timeslot t
                            {'max':10,'min':0},
                        'load0':# g : PV load forecast next timeslot
                                {'max':10,'min':0},
                        'load1':# g : PV load forecast 1h ahead
                                 {'max':10,'min':0},
                        'load6':# g : PV load forecast 6h ahead
                                 {'max':10,'min':0},
                        'load12':# g : PV load forecast 12h ahead
                                 {'max':10,'min':0},
                        'load24':# g : PV load forecast 24h ahead
                                 {'max':10,'min':0},
                        'delta': #The differential between the gen and load
                            {'max':10,'min':-10.0},
                        'delta0': #The differential between the gen and load next timeslot
                            {'max':10,'min':-10.0},
                        'delta1': #The differential between the gen and load 1h ahead
                            {'max':10,'min':-10.0},
                        'delta6': #The differential between the gen and load 6h ahead
                            {'max':10,'min':-10.0},
                        'delta12': #The differential between the gen and load 12h ahead
                            {'max':10,'min':-10.0},
                        'delta24': #The differential between the gen and load 24h ahead
                            {'max':10,'min':-10.0},
                        'load_s': #l_s: shiftable load 
                            {'max':max(self.profile),'min':0},
                        'delta_s': #The differential betwee gen and l_s
                            {'max':10,'min':-10.0},
                        'delta_c': #The differential betwee gen and load + l_s
                                {'max':10,'min':-10.0},
                        'y': # =1 if ON at t, 0 OTW
                            {'max':1.0,'min':0.0},
                        'y_1': # =1 if ON in t-1
                            {'max':1.0,'min':0.0},
                        'y_s':  # +1 if app is schedulled at t (incremental) 
                                #(how many times it was connected)
                            {'max':self.T,'min':0.0},
                        'cost':
                            {'max':100.0,'min':-100.0},
                        'cost_s':
                            {'max':100.0,'min':-100.0},
                        'cost_s_x':#cost of supply shiftable load using PV excess
                            {'max':100.0,'min':-100.0},

                        'tar_buy':
                            {'max':1,'min':0},
                        'tar_buy0': #Tariff at the next timestep
                            {'max':1,'min':0},
                        'E_prof': # reamining energy to supply appliance energy need
                            {'max':self.E_prof,'min':0.0},
                        'excess': #PV excess affter supplying baseload (self.load)}
                            {'max':10,'min':-10.0},}
                            
                    
        
            
        
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
        self.state_update(self.action) #update state variables  
        
        
        
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
        
        # We can choose to reset to a random state or to t=0
        # self.tstep=0 # start at t=0
        
        # self.tstep = rnd.randrange(0, self.T-47-1) # a random initial state in the whole year   
        # print('chamou resert')
        
        self.tstep=self.get_init_tstep()
        
        
        self.tstep_init=self.tstep # initial timestep
        # print(self.tstep)
        
        # self.tar_buy=0.17
        self.L_s=np.zeros(self.T)

        # self.gen=self.data[self.tstep,0]
        # self.load=self.data[self.tstep,1]
        self.minutes=self.data[self.tstep,2]
        
        
       
        
        
        self.sin=np.sin(2*np.pi*(self.minutes/self.min_max))
        self.cos=np.cos(2*np.pi*(self.minutes/self.min_max))
        
        self.load_s=0   
        

        self.gen=self.data[self.tstep][0] # valor da generation para cada instante
        
        if self.tstep+1 >= self.T:
            self.gen0 = 0  
        else:
            self.gen0=self.data[self.tstep+1][0] #next tstep
                
                
        if self.tstep+2 >= self.T:
            self.gen1 = 0  
        else:
            self.gen1=self.data[self.tstep+2][0] #1 hour ahead      
                
        
        if self.tstep+6*2 >= self.T:
            self.gen6 = 0  
        else:
            self.gen6=self.data[self.tstep+6*2][0] #6 hours ahead
                

        if self.tstep+12*2 >= self.T:
            self.gen12 = 0  
        else:
            self.gen12=self.data[self.tstep+12*2][0] #12 hours ahead
            
        if self.tstep+24*2 >= self.T:
            self.gen24 = 0  
        else:
            self.gen24=self.data[self.tstep+24*2][0] #12 hours ahead
        
        
        #Load and load forecast
        self.load=self.data[self.tstep][1] # valor da load para cada instante
        
        if self.tstep+1 >= self.T:
            self.load0 = 0  
        else:
            self.load0=self.data[self.tstep+1][1] #next tstep
                
                
        if self.tstep+2 >= self.T:
            self.load1 = 0  
        else:
            self.load1=self.data[self.tstep+2][1] #1 hour ahead      
                
        
        if self.tstep+6*2 >= self.T:
            self.load6 = 0  
        else:
            self.load6=self.data[self.tstep+6*2][1] #6 hours ahead
                

        if self.tstep+12*2 >= self.T:
            self.load12 = 0  
        else:
            self.load12=self.data[self.tstep+12*2][1] #12 hours ahead
            
        if self.tstep+24*2 >= self.T:
            self.load24 = 0  
        else:
            self.load24=self.data[self.tstep+24*2][1] #24 hours ahead
        
        
        
        self.get_tariffs() #update tarrifs
        self.get_tariffs0() #update tarifs from next timestep
        
        #deltas
        self.delta = self.load-self.gen
        self.delta0 = self.load0-self.gen0
        self.delta1 = self.load1-self.gen1
        self.delta6 = self.load6-self.gen6
        self.delta12 = self.load12-self.gen12
        self.delta24 = self.load24-self.gen24
        
        
        self.R=0
        self.r=0
        
        self.c_T=0
        
        self.delta_c=(self.load+self.load_s)-self.gen
        self.delta_s=self.load_s-self.gen
        
        # self.load_s=self.L_s[0] #initialize with the first element in L_s
        self.load_s=0 #machine starts diconected
        
        
        # if the random initial time step is after the delivery time it must not turn on
        # if self.minutes >= self.t_deliver-self.T_prof*self.tstep_size:
        #     self.E_prof=0 #means that there is no more energy to consume 
        #     self.y_s=self.T_prof #means taht it connected allready the machine T_prof times
        # else:
        #     self.E_prof=self.profile.sum()*self.dh #energy needed for the appliance
        #     self.y_s=0 # means that it has never connected the machine
        
        self.E_prof=self.profile.sum()*self.dh
        self.y_s=0
        
        
        #inititialize binary variables
        self.y=0
        self.y_1=0
        # self.y_s=self.y
        self.t_shift=0
        
        self.hist=[]
        self.hist.append(self.y)

        # self.tstep=0
        # self.grid=0.0
        # self.I_E = 0.0
        
        self.cost=max(0,self.delta_c)*self.tar_buy + min(0,self.delta_c)*self.tar_sell
        self.cost_s=max(0,self.delta_s)*self.tar_buy + min(0,self.delta_s)*self.tar_sell
        
        self.excess=max(0,-self.delta)   
        self.cost_s_x=max(0,self.load_s-self.excess)*self.tar_buy + min(0,self.load_s-self.excess)*self.tar_sell
        
        
        
        self.obs={"action_mask": self.get_mask(),  
              "observations": self.get_obs()
              }
    
        return self.obs




    def get_reward(self):
        
            if self.reward_type == 'simple_cost':

                # reward=np.exp(-(self.cost_s**2)/0.01)+np.exp(-(((self.y_s-self.T_prof)**2)/0.001))
                # The reward should be function of the action
                if self.minutes == self.min_max-self.T_prof*self.tstep_size and self.y_s!=self.T_prof:
                    reward=-10
                
                else:
                    # reward=np.exp(-(self.cost**2)/0.001)-0.5                                           
                    # reward=-self.cost*self.delta
                    reward=-10*self.cost_s*self.delta
                    
                    
                    
                    
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
        if self.done_cond == 'mode_window':
            
            # t=rnd.randrange(0, self.T-self.Tw-1) # a random initial state in the whole year
            t=rnd.choice([k*self.Tw for k in range(int((self.T/self.Tw)-1))]) # we allways start at the beggining of the day and advance Tw timesteps but choose randomly what day we start
            
            return t
            
           
        
        
        elif self.done_cond == 'mode_horizon': 
            #episode starts at t=0
            return 0
        
        
    def get_mask(self):
        
        mask = np.ones(self.action_space.n)
        # obs=self.get_obs()
        
        if self.y==1 and self.y_s < self.T_prof:
            mask=np.array([0,1])
        # else:
        #     mask = np.ones(self.action_space.n)
            
        elif self.y_s >= self.T_prof:
            mask=np.array([1,0])

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
            
            return self.obs, reward, self.done, {}
        
        else:
            self.done = False
        



    def state_update(self, action):
        
        #Variables update
        
        self.y=action # what is the ON/OFF state of appliance
        
        
        # print('step', self.tstep)
        # print('episodes', self.n_episodes)
        
        
        self.get_tariffs() #update tariffs
        self.get_tariffs0() #update tarifs from next timestep
        
            
        # O self.data é o data das duas colunas verticais do auxfunctions
        self.gen=self.data[self.tstep][0] # valor da generation para cada instante
        
        if self.tstep+1 >= self.T:
            self.gen0 = 0  
        else:
            self.gen0=self.data[self.tstep+1][0] #next tstep
                
                
        if self.tstep+2 >= self.T:
            self.gen1 = 0  
        else:
            self.gen1=self.data[self.tstep+2][0] #1 hour ahead      
                
        
        if self.tstep+6*2 >= self.T:
            self.gen6 = 0  
        else:
            self.gen6=self.data[self.tstep+6*2][0] #6 hours ahead
                

        if self.tstep+12*2 >= self.T:
            self.gen12 = 0  
        else:
            self.gen12=self.data[self.tstep+12*2][0] #12 hours ahead
            
        if self.tstep+24*2 >= self.T:
            self.gen24 = 0  
        else:
            self.gen24=self.data[self.tstep+24*2][0] #12 hours ahead
        
        
        #Load and load forecast
        self.load=self.data[self.tstep][1] # valor da load para cada instante
        
        if self.tstep+1 >= self.T:
            self.load0 = 0  
        else:
            self.load0=self.data[self.tstep+1][1] #next tstep
                
                
        if self.tstep+2 >= self.T:
            self.load1 = 0  
        else:
            self.load1=self.data[self.tstep+2][1] #1 hour ahead      
                
        
        if self.tstep+6*2 >= self.T:
            self.load6 = 0  
        else:
            self.load6=self.data[self.tstep+6*2][1] #6 hours ahead
                

        if self.tstep+12*2 >= self.T:
            self.load12 = 0  
        else:
            self.load12=self.data[self.tstep+12*2][1] #12 hours ahead
            
        if self.tstep+24*2 >= self.T:
            self.load24 = 0  
        else:
            self.load24=self.data[self.tstep+24*2][1] #12 hours ahead
        
        
        #deltas
        self.delta = self.load-self.gen
        self.delta0 = self.load0-self.gen0
        self.delta1 = self.load1-self.gen1
        self.delta6 = self.load6-self.gen6
        self.delta12 = self.load12-self.gen12
        self.delta24 = self.load24-self.gen24
        

        ##
        self.minutes=self.data[self.tstep][2]
        self.sin=np.sin(2*np.pi*(self.minutes/self.min_max))
        self.cos=np.cos(2*np.pi*(self.minutes/self.min_max))
        
        
        # note to consider: shiftable load is allways zero unless the appliance is activated
        
        
        # Binary variables
        
        
         # how many times the machine have been turned on

        
        
        self.hist.append(self.y)
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
        self.load_s=self.y*self.profile[0]
        self.y_s+=self.y
            
            
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
        if self.E_prof < self.load_s*self.dh:
            self.E_prof=0.0
        elif self.minutes==self.min_max:
            self.E_prof=self.profile.sum()*self.dh
        else:
            self.E_prof-=self.load_s*self.dh
        
        
        #restart t_shift for each new day so that appliances may be schedulled again
        if self.minutes ==self.min_max:
            self.t_shift=0
            self.y_s=0
        
        
        
        #deficit vs excess
        self.delta_c = (self.load+self.load_s)-self.gen # if positive there is imports from the grid. If negative there are exports to the grid 
        
        self.delta_s=self.load_s-self.gen
        

        #energy cost
        
        # cost considering the full load
        self.cost=max(0,self.delta_c)*self.tar_buy + min(0,self.delta_c)*self.tar_sell
        #cost considering only the appliance
        self.cost_s=max(0,self.delta_s)*self.tar_buy + min(0,self.delta_s)*self.tar_sell
        
        
        self.excess=max(0,-self.delta)   
        self.cost_s_x=max(0,self.load_s-self.excess)*self.tar_buy + min(0,self.load_s-self.excess)*self.tar_sell
        
        

        
        
        
        
    def get_tariffs(self):
        "Define tarrifs in €/kWh -- define here any function for defining tarrifs"
        # Tarifa bi-horaria
        # if self.minutes >= 240 and self.minutes <=540:
        #Tarifa bi-horaria
        if self.minutes >= self.tstep_size*2*8 and self.minutes <=self.tstep_size*2*22:
            self.tar_buy=0.1393
        else:
            self.tar_buy=0.0615
        # self.tar_buy=0.17*(1-(self.gen/1.764)) #PV indexed tariff 
            
        self.tar_sell=0.0 # remuneration for excess production
    
        
        
    def get_tariffs0(self):
        "Define tarrifs in €/kWh for the next timestep-- define here any function for defining tarrifs"
        if self.minutes+self.tstep_size>= self.tstep_size*2*8 and self.minutes+self.tstep_size <=self.tstep_size*2*22:
            self.tar_buy0=0.1393
        else:
            self.tar_buy0=0.0615
        # self.tar_buy=0.17*(1-(self.gen/1.764)) #PV indexed tariff 
            
        self.tar_sell0=0.0 # remuneration for excess production
    
        
        
        
        
        