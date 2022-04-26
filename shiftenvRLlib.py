#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 11:31:14 2022

@author: omega
"""
import gym
import gym
import numpy as np
import random as rnd
import time



class ShiftEnv(gym.Env):
    
    
    """
    A custom OpenAI Gym environment for managing shiftable loads
    """

    def __init__(self, config):
        """
        data: a pandas dataframe with number of columns N=# of houses or loads and time-horizon T=Total number of timesteps
        soc_max: maximum battery state-of-charge
        eta: Charging efficiêncy
        charge_lim: maximum charging power
        """

        self.reward_type=config["reward_type"]
        self.data=config["data"] # We need to import the unflexible load and PV production.
        # data is an array with load in column 0 and pv production in column 1
        
       
        
        
        self.T=len(self.data) # Time horizon
        self.dh=30.0*(1/60.0) # Conversion factor energy-power
        
        #Appliance profile
        
        self.profile=config["profile"]
        self.T_prof=len(self.profile)
        self.E_prof=self.profile.sum()*self.dh #energy needed for the appliance
        
        self.R=0
        self.R_Total=[] # A way to see the evolution of the rewards as the model is being trained
        self.n_episodes=0

        #new vars
        self.L_s=np.zeros(self.T) # we make a vector of zeros to store the shiftable load profile
        # self.l_s=self.L_s[0] #initialize with the first element in 
        
        # self.y=0
        # self.y_s=self.y
        # self.y_x=0
        # self.y_target=1 # target number of times the machine should connect per defined period
        
        # self.tstep=0
        # self.y_tsteps=
        # self.y_period=1 # Number of days  

        # self.t=0
        # self.grid=0.0
        # self.I_E = 0.0
        # self.delta=0.0
        
        self.tar_buy=0.17 # import tariff in €/kWh
        self.tar_sell=0.0 # remuneration for excess


        # defining the state variables
        self.state_vars={'tstep':
                        {'max':2000,'min':0},
                        'minutes':
                            {'max':1440,'min':0}, 
                        'gen':# g : PV generation at timeslot
                            {'max':10,'min':0},
                        'gen0':# g : PV generation forecast next timeslot
                                {'max':10,'min':0},
                        'gen1':# g : PV generation forecast 1h ahead
                                 {'max':10,'min':0},
                        'gen3':# g : PV generation forecast 3h ahead
                                 {'max':10,'min':0},
                        'gen6':# g : PV generation forecast 6h ahead
                                 {'max':10,'min':0},
                        'load':# l : Load at timeslot t
                            {'max':10,'min':0},
                        'load_s': #l_s: shiftable load 
                            {'max':max(self.profile),'min':0},
                        'delta': #The differential between the gen and load
                            {'max':10,'min':-10.0},
                        'delta_s': #The differential betwee gen and l_s
                            {'max':10,'min':-10.0},
                        'y': # binary 1 if app is schedulled at t, 0 otw
                            # When it was connected
                            {'max':1.0,'min':0.0}, 
                        'y_s':  # +1 if app is schedulled at t (incremental) 
                                #(how many times it was connected)
                            {'max':self.T,'min':0.0},
                        'y_x':  # +1 if app is ON at t (incremental)
                                # How many timesteps it was connected
                            {'max':10*self.T_prof,'min':0.0},
                        'cost':
                            {'max':100.0,'min':-100.0},
                        'cost_s':
                                {'max':100.0,'min':-100.0},
                        'E_prof': # reamining energy to supply appliance energy need
                            {'max':self.E_prof,'min':0.0}}
                    
        
    
        
        #Number of variables to be used

        self.var_dim=len(self.state_vars.keys())
        

        
        self.highlim=np.array([value['max'] for key, value in self.state_vars.items()])
        self.lowlim=np.array([value['min'] for key, value in self.state_vars.items()])
            
        # Observation space
        self.observation_space = gym.spaces.Box(low=self.lowlim, high=self.highlim, shape=(self.var_dim,),dtype=np.dtype('float32'))



        #Actions

        self.action_space = gym.spaces.Discrete(2) # ON/OFF
        
    


    def step(self, action):
        # print('action', action)
        done=False
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
        #get charge load form action
        #Mofification António
        # action_charge=self.get_charge(action)
        # action_discharge=self.get_discharge(action)
        
        
        action=float(action)
        
        reward=self.get_reward(action, reward_type=self.reward_type)
        

        if self.tstep==len(self.data)-1:
            # done=True
            self.R_Total.append(self.R)
            print(self.R)
            # print('timestep',self.t)

            self.n_episodes+=1
            done = True

            return np.array((self.tstep,self.minutes,self.gen,self.gen0,self.gen1,self.gen3,self.gen6, self.load,self.load_s,self.delta,self.delta_s,self.y,self.y_s,self.y_x,self.cost,self.cost_s,self.E_prof), dtype=np.dtype('float32')),0,done, {}
        
        else:
            done = False

        #Variables update
        
        self.tstep+=1
        
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
                
        
        if self.tstep+3*2 >= self.T:
            self.gen3 = 0  
        else:
            self.gen3=self.data[self.tstep+3*2][0] #3 hours ahead
                

        if self.tstep+6*2 >= self.T:
            self.gen6 = 0  
        else:
            self.gen6=self.data[self.tstep+6*2][0] #6 hours ahead
            
        
          
        self.load=self.data[self.tstep][1] # valor da load para cada instante
        self.minutes=self.data[self.tstep][2]
        
        
        # note to consider: shiftable load is allways zero unless the appliance is activated
        
        
        # Binary variables
        self.y=action #what action have been taken
        self.y_s+=self.y # how many times the machine have been turned on
        # self.y_x+=self.y
        
        
        if self.load_s != 0:
            self.y_x+=1
        #Load the shiftable profile [H1]
        
        # if self.y_s == 1:
        #     self.t_shift=0 # a counter for the shiftable load profile
        #     if self.t_shift < self.T_prof: #while counter is less than the duration of the cycle
        #         self.l_s=self.profile[self.t_shift]
        #         self.t_shift=+1
        
        # elif self.y_s != 1:
            
        
        
        #Load the shiftable profile [H2]
        
        if action == 1:
            # print('L_s',self.L_s)
            # print('L_s2', self.L_s[self.t:self.t+self.T_prof])
            # print('prof', self.profile)
            # self.L_s[self.t:self.t+self.T_prof]=self.profile # we load the shiftable profile from t onwards
            
            self.L_s=np.zeros(self.T)
            if self.tstep <= self.T-self.T_prof:
                for k in range(self.T_prof):
                    self.L_s[self.tstep+k]=self.profile[k]
                    self.load_s=self.L_s[self.tstep] 
            
        if action == 0:
            self.load_s=self.L_s[self.tstep] 
            
            # self.L_s[self.t:][:self.T_prof]=self.profile
            # #what if the appliance is turned on several times??
        
        #update remaining energy that needs to be consumed
        if self.E_prof < self.load_s:
            self.E_prof=0.0
        else:
            self.E_prof-=self.load_s*self.dh
        
        
        #deficit vs excess
        self.delta = (self.load+self.load_s)-self.gen # if positive there is imports from the grid. If negative there are exports to the grid 
        
        self.delta_s=self.load_s-self.gen

        
        # reward=self.get_reward(action, reward_type=self.reward_type)
        self.R+=reward
        self.r=reward
        
        
        #energy cost
        
        self.cost=max(0,self.delta)*self.tar_buy*self.dh + min(0,self.delta)*self.tar_sell*self.dh
        self.cost_s=max(0,self.delta_s)*self.tar_buy*self.dh + min(0,self.delta_s)*self.tar_sell*self.dh
        # self.c_s=
        
        # if self.delta > 0: # there is import from the grid
        #     self.c=self.tar_buy*self.delta*self.dh
        # elif self.delta <= 0: #There is export to the grid
        #     self.c=self.tar_sell*self.delta*self.dh
            

        info={}

        observation=np.array((self.tstep,self.minutes,self.gen,self.gen0,self.gen1,self.gen3,self.gen6, self.load,self.load_s,self.delta,self.delta_s,self.y,self.y_s,self.y_x,self.cost,self.cost_s,self.E_prof), dtype=np.dtype('float32'))
    

        return observation, reward, done, info
    


    def get_reward(self,action, reward_type):

        # reward=np.exp(-(self.cost**2)/0.001)
        # if self.g!=0 and self.y=1 :
            
        # a=0.7
        # b=0.3
        # reward=a*np.exp(-(self.cost_s**2)/0.001)+b*np.exp(-((self.y_s-1)**2)/0.001)
        
        # reward=np.exp(-(self.cost_s**2)/0.001)+np.exp(-((self.y_s-1)**2)/0.001)

        # reward=-self.cost_s-(-np.exp(-((self.y_s-1)**2)/0.001)
        
        # reward=-self.cost_s-10*self.E_prof
        # if self.tstep == self.T-1 and self.y_s!=1: # if we arrived at end without turning on
        #     reward=-100
        # else:
        reward=-self.cost_s
        
        
        
        
        # if self.y_s <= 1: #we only want to connect the device once
                # reward=-self.c # reward is the cost
                # reward=np.exp(-(self.c**2)/0.001)
        # else: 
            # reward=-1
            

        
        return reward


    def reset(self):
        """
        Reset the environment state and returns an initial observation

        Returns:
        -------
        observation (object): The initial observation for the new episode after reset
        :return:
        """

        done=False
        
        # We can choose to reset to a random state or to t=0
        # self.tstep=0 # start at t=0
        self.tstep = rnd.randrange(0, self.T-self.T_prof-1, 2) # a random initial state
        
        self.L_s=np.zeros(self.T)

        self.gen=self.data[self.tstep,0]
        self.load=self.data[self.tstep,1]
        self.minutes=self.data[self.tstep,2]
        self.load_s=0   
        
        # self.gen0=self.data[1][0] #next tstep
        # self.gen1=self.data[2][0] #1 hour ahead
        # self.gen3=self.data[3*2][0] #3 hours ahead
        # self.gen6=self.data[6*2][0] #6 hours ahead
        
        if self.tstep+1 >= self.T:
            self.gen0 = 0  
        else:
            self.gen0=self.data[self.tstep+1][0] #next tstep
                
                
        if self.tstep+2 >= self.T:
            self.gen1 = 0  
        else:
            self.gen1=self.data[self.tstep+2][0] #1 hour ahead      
                
        
        if self.tstep+3*2 >= self.T:
            self.gen3 = 0  
        else:
            self.gen3=self.data[self.tstep+3*2][0] #3 hours ahead
                

        if self.tstep+6*2 >= self.T:
            self.gen6 = 0  
        else:
            self.gen6=self.data[self.tstep+6*2][0] #6 hours ahead
       
   
        self.R=0
        self.r=0


        self.delta=(self.load+self.load_s)-self.gen
        self.delta_s=self.load_s-self.gen
        
        self.load_s=self.L_s[0] #initialize with the first element in L_s
        self.E_prof=self.profile.sum()*self.dh #energy needed for the appliance
        
        self.y=0
        self.y_s=self.y
        self.y_x=0

        # self.tstep=0
        # self.grid=0.0
        # self.I_E = 0.0
        
        self.cost=max(0,self.delta)*self.tar_buy*self.dh + min(0,self.delta)*self.tar_sell*self.dh
        self.cost_s=max(0,self.delta_s)*self.tar_buy*self.dh + min(0,self.delta_s)*self.tar_sell*self.dh
    
        observation=np.array((self.tstep,self.minutes,self.gen,self.gen0,self.gen1,self.gen3,self.gen6, self.load,self.load_s,self.delta,self.delta_s,self.y,self.y_s,self.y_x,self.cost,self.cost_s,self.E_prof), dtype=np.dtype('float32'))
    
        return observation


    def render(self, mode='human', close=False):
        """
        :param mode:
        :return:
        """
        return
    
    
    
    def resetzero(self):
        """
        Reset the environment state and returns an initial observation

        Returns:
        -------
        observation (object): The initial observation for the new episode after reset
        :return:
        """

        done=False
        
        # We can choose to reset to a random state or to t=0
        self.tstep=0 # start at t=0

        self.L_s=np.zeros(self.T)

        self.gen=self.data[self.tstep,0]
        self.load=self.data[self.tstep,1]
        self.minutes=self.data[self.tstep,2]
        self.load_s=0   
        
        # self.gen0=self.data[1][0] #next tstep
        # self.gen1=self.data[2][0] #1 hour ahead
        # self.gen3=self.data[3*2][0] #3 hours ahead
        # self.gen6=self.data[6*2][0] #6 hours ahead
        
        if self.tstep+1 >= self.T:
            self.gen0 = 0  
        else:
            self.gen0=self.data[self.tstep+1][0] #next tstep
                
                
        if self.tstep+2 >= self.T:
            self.gen1 = 0  
        else:
            self.gen1=self.data[self.tstep+2][0] #1 hour ahead      
                
        
        if self.tstep+3*2 >= self.T:
            self.gen3 = 0  
        else:
            self.gen3=self.data[self.tstep+3*2][0] #3 hours ahead
                

        if self.tstep+6*2 >= self.T:
            self.gen6 = 0  
        else:
            self.gen6=self.data[self.tstep+6*2][0] #6 hours ahead
       
   
        self.R=0
        self.r=0


        self.delta=(self.load+self.load_s)-self.gen
        self.delta_s=self.load_s-self.gen
        
        self.load_s=self.L_s[0] #initialize with the first element in L_s
        self.E_prof=self.profile.sum()*self.dh #energy needed for the appliance
        
        self.y=0
        self.y_s=self.y
        self.y_x=0

        # self.tstep=0
        # self.grid=0.0
        # self.I_E = 0.0
        
        self.cost=max(0,self.delta)*self.tar_buy*self.dh + min(0,self.delta)*self.tar_sell*self.dh
        self.cost_s=max(0,self.delta_s)*self.tar_buy*self.dh + min(0,self.delta_s)*self.tar_sell*self.dh
    
        observation=np.array((self.tstep,self.minutes,self.gen,self.gen0,self.gen1,self.gen3,self.gen6, self.load,self.load_s,self.delta,self.delta_s,self.y,self.y_s,self.y_x,self.cost,self.cost_s,self.E_prof), dtype=np.dtype('float32'))
    
        return observation