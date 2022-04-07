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



class FlexEnv(gym.Env):
    
    
    
    """
    A custom OpenAI Gym environment for managing a electrical battery
    """

    def __init__(self, config):
        """
        data: a pandas dataframe with number of columns N=# of houses or loads and time-horizon T=Total number of timesteps
        soc_max: maximum battery state-of-charge
        eta: Charging efficiêncy
        charge_lim: maximum charging power
        """

        self.reward_type=config["reward_type"]

        self.soc_max=config["soc_max"]  # Defined by the user (Try real values)
        self.eta=config["eta"]
        self.data=config["data"] # We need to import the unflexible load and PV production.
        # data is an array with load in column 0 and pv production in column 1
        self.T=len(self.data) # Time horizon
        self.dh=30.0*(1/60.0) # Conversion factor energy-power
        self.R_Total=[] # A way to see the evolution of the rewards as the model is being trained
        self.n_episodes=0
        # Modificação António
        self.soc=0.0
        self.t=0
        self.grid=0.0
        self.I_E = 0.0
        self.bat_used=0.0
        self.delta=0.0

        self.__version__ = "0.0.1"
        
        # Modify the observation space, low, high and shape values according to your custom environment's needs
        # highlim=np.array([100.0,100.0,10*self.soc_max,10*self.soc_max,100.0,100.0,100.0])
        # lowlim=np.array([-100.0,-100.0,-100.0,-100.0,-100.0,-100.0,-100.0])
        
        highlim=np.array([1000.0,1000.0,1000*self.soc_max,1000*self.soc_max,1000.0,1000.0,1000.0])
        lowlim=np.array([-1000.0,-1000.0,-1000.0,-1000.0,-1000.0,-1000.0,-1000.0])
        
        #Names of variables
        self.varnames=('gen','load','soc','soc_1','delta','grid','I/E')
        
        #Number of variables
        self.var_dim=len(self.varnames)
        

    
        # Observation space
        # self.observation_space = gym.spaces.Box(low=np.float32(lowlim), high=np.float32(highlim), shape=(self.var_dim,))
        
        self.observation_space = gym.spaces.Box(low=lowlim, high=highlim, shape=(self.var_dim,))

        """
        Variables definitions (not all are being used)
        
        State definition
        # t : time slot
        # gen : PV generation at timeslot t
        # load : Load at timeslot t
        # SOC : State of charge
        # SOC_1 : State of charge t-1
        # tar : Electricity tariff at timeslot t
        # R  : total reward per episode
        # sc: self-consumption
        # r :reward
        
        # grid: Energy that comes from the grid
        # I/E - grid import/export: 
        # bat_used: Total Energy used from the battery
        # delta: gen-load. The differential between the PV generation and the load at each instant
        # Energy Balance 
        # energy cost
        # Total energy cost
        """

        #Actions
        # The battery actions are the selection of charging power but these are discretized between zero and charge_lim
        
        # The charge steps are defined: 0 is the min, self.charge_lim is the max and there's 
        # (self.charge_lim/minimum_charge_step)+1) numbers in the array. Basically it increases the minimum charge step each number
        # limits on batery
        self.charge_lim=config["charge_lim"] # The battery has a limit of charging of charge_lim
        self.discharge_lim=-self.charge_lim # The battery has a limit of discharging of charge_lim
        self.minimum_charge_step =config["min_charge_step"]


        self.action_space = gym.spaces.Box(low=self.discharge_lim, high=self.charge_lim,shape=(1,))
        
    


    def step(self, action):
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
        
        # if self.t>=len(self.data)-1 or self.soc >= self.soc_max:
        if self.t==len(self.data)-1:
            # done=True
            self.R_Total.append(self.R)
            print(self.R)

            self.n_episodes+=1
            done = True

            return np.array(([self.g,self.l,self.soc,self.soc1,self.delta,self.grid,self.I_E])),0,done, {}
        else:
            done = False

        
        self.t+=1
        
        # O self.data é o data das duas colunas verticais do auxfunctions
        self.g=self.data[self.t][0] # valor da generation para cada instante
        self.l=self.data[self.t][1] # valor da load para cada instante
        
        
        self.delta = self.g-self.l
        self.soc1 = self.soc
        
        
        # Charging discharging dynamics
        if action >= 0: #charging
            self.soc+=self.eta*(action)*self.dh # New State of Charge
        elif action < 0: #discharging
            self.soc+=((action)/self.eta)*self.dh
        # elif action_ == 0: #do nothing
        #     self.soc = self.soc
        
        
        self.tar=0.17 # grid tariff in €/kWh
        
        # If the action is to discharge, then the self.grid is the difference between the discharge amount and the load needed
        # If the action is to charge the battery or do nothing to the battery, then the self.grid is the load needed for that instant 
        if action<0: # Discharge the battery
            if abs(action)<=abs(self.delta):
                self.grid =abs(self.delta)-abs(action)
            else:
                self.grid=0
        else: # Charge the battery
            if self.delta < 0:
                self.grid = abs(self.delta)
            else:
                self.grid = 0
        
                
        # IMport/Export
        self.I_E = action-self.delta
            
        # If the action is to discharge, the amount the battery discharges is the amount of energy used   
        if action<0:
            self.bat_used+=abs(action)

        # self consumption (it should be 0<SC<1 but it is possible to have SC>1)
        #TODO: Think about how to maintain 0<SC<1
        
        if self.g!=0:
            self.sc=(self.l+action)/self.g
        else:
            self.sc=0

        
        # print(reward)
        # reward=self.get_reward(action, reward_type=self.reward_type)
        self.R+=reward
        self.r=reward
        
        
        #energy cost
        self.c=self.tar*self.grid*self.dh

        info={}

        observation=np.array(([self.g,self.l,self.soc,self.soc1,self.delta,self.grid,self.I_E]))
        # print(observation)
        # print(observation,reward,done)

        return observation, reward, done, info
    


    def get_reward(self,action, reward_type):

        
        #hipothesis 1
        if reward_type==1:
        
        
        
        # Rewards mais geral possível (soc, g, soc_max, l, minimum_charge_step,action_,grid,delta)
        # Charge
        
            if action > 0:
                if self.soc_max>= self.soc >= 0 and self.delta>0 and action<=self.delta:
                    reward_charge = abs(action)*2
                else:
                    reward_charge =-abs(action)*2
            else:
                reward_charge =0
                
            # Discharge
            if action< 0:
                if self.soc_max>= self.soc >=0 and self.delta<0:
                    
                    if abs(self.delta)<abs(action): 
                        # Este se calahr era melhor multiplicar para os rewards positivos serem mais positivos
                        reward_discharge =abs(self.delta)-(abs(action)-abs(self.delta)) # Tentar perceber até onde é que faz sentido se ele descarregar bue se isso é realmente mau, pode ser só indiferente
                        # if reward_discharge < 0:
                        #     reward_discharge = 0
                        
                    else:
                        reward_discharge = abs(action)
    
                else:
                    reward_discharge = -abs(action)*2
            else:
                reward_discharge= 0
                
            
            # Do nothing
            if action==0 :
                if self.g==0 and 0<= self.soc <=self.minimum_charge_step: # Aumentar o reward positivo
                    reward_stay=self.charge_lim/2
                else:
                    reward_stay=0
            else:
                reward_stay=0
                
            # Grid
            reward_grid = -self.grid
            
            reward = reward_charge+reward_discharge+reward_stay +reward_grid
        
        
        #hipothesis 2
        
        elif reward_type==2:
            
            if self.soc <= self.soc_max and self.soc >= 0:
                reward=float(np.exp(-(action-self.delta)**2/0.01)) # Gaussian function that approximates a tep function
            
            else:
                reward=0
        
        
        else:
        
            print('No reward')
        
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
        self.t=0 # start at t=0
        # self.t = rnd.randrange(0, 46, 2) # a random initial state

        self.g=self.data[0,0]
        self.l=self.data[0,1]
        
        self.soc=0.4*self.soc_max
        self.soc1=0
        self.tar=0.17
        self.R=0
        self.r=0
        # self consumption
        #we are starting at t=0 with sc=0 because there is no sun. the initial state depends on the action and we cannot have actions on reset()
        self.sc=0
        self.grid=0
        self.bat_used=0
        self.delta=self.g-self.l
        
        self.I_E=0
        

    

        observation=np.array(([self.g,self.l,self.soc,self.soc1,self.delta,self.grid,self.I_E]))

        return observation


    def render(self, mode='human', close=False):
        """
        :param mode:
        :return:
        """
        return