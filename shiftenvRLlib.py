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
        
       
        
        self.tstep_size=config["step_size"]
        self.T=len(self.data) # Time horizon
        self.Tw=24*2 #window horizon
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
                        'tar_buy':
                            {'max':1,'min':0},
                                
                        'E_prof': # reamining energy to supply appliance energy need
                            {'max':self.E_prof,'min':0.0}}
                    
        
            
        
        #Number of variables to be used
        self.var_dim=len(self.state_vars.keys())
                
        
                
        self.highlim=np.array([value['max'] for key, value in self.state_vars.items()])
        self.lowlim=np.array([value['min'] for key, value in self.state_vars.items()])
            
        # Observation space
        self.observation_space = gym.spaces.Box(low=np.float32(self.lowlim), high=np.float32(self.highlim), shape=(self.var_dim,),dtype=np.dtype('float32'))



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

        
        
        action=float(action)
        
        reward=self.get_reward(action, reward_type=self.reward_type)
        

        if self.tstep==self.tstep_init+47: # episode ends when when 24 hours passed
            # done=True
            self.R_Total.append(self.R)
            print(self.R)
            # print('tstep_init',self.tstep_init)
            # print('tstep',self.tstep)

            self.n_episodes+=1
            done = True

            # return np.array((self.tstep,self.minutes,self.sin,self.cos,self.gen,self.gen0,self.gen1,self.gen3,self.gen6, self.load,self.load_s,self.delta,self.delta_s,self.y,self.y_1,self.y_s,self.cost,self.cost_s,self.tar_buy,self.E_prof), dtype=np.dtype('float32')),0,done, {}
        
            return self.get_obs(), 0, done, {}
        
        else:
            done = False
            
        #Variables update
        
        self.tstep+=1
        
        
        
        if self.tstep >= 8 and self.tstep <=16:
            self.tar_buy=0.09 
        else:
            self.tar_buy=0.17
            
            
        
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
        self.sin=np.sin(2*np.pi*(self.minutes/self.min_max))
        self.cos=np.cos(2*np.pi*(self.minutes/self.min_max))
        
        
        # note to consider: shiftable load is allways zero unless the appliance is activated
        
        
        # Binary variables
        
        self.y=action # what is the ON/OFF state of appliance
        
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
        if self.tstep <= self.T-self.T_prof:
            if self.y == 1: #if it must be turned ON on the present tslot
                if self.y_s==0: #if its never been ON
                    # self.t_shift=0
                    self.load_s=self.y*self.profile[self.t_shift]
                
                if self.y_s!=0 and self.t_shift < self.T_prof-1:
                    self.t_shift+=1
                    self.load_s=self.y*self.profile[self.t_shift]
                
                if self.y_s == self.T_prof:
                    self.t_shift=0
                    self.load_s=self.y*self.profile[self.t_shift]
                    
                    
                
            elif self.y == 0:
                self.load_s = 0
                

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
        
        #accumulated total cost
        self.c_T+=self.cost_s

        info={}

        # observation=np.array((self.tstep,self.minutes,self.sin,self.cos,self.gen,self.gen0,self.gen1,self.gen3,self.gen6, self.load,self.load_s,self.delta,self.delta_s,self.y,self.y_1,self.y_s,self.cost,self.cost_s,self.tar_buy,self.E_prof))
        
        observation=self.get_obs()
    

        return observation, reward, done, info
    


    def get_reward(self,action, reward_type):

        # reward=np.exp(-(self.cost**2)/0.001)
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
                
        if (self.minutes >= self.t_deliver-self.T_prof*self.tstep_size and self.minutes <= self.min_max and self.y_s < self.T_prof) or (self.y_s > self.T_prof) or (self.y_s > 0 and self.y_s < self.T_prof and self.y==0):
            reward=-1/self.T
        else:
            reward= -self.cost**2*self.y-0.1/self.T*(abs(self.y-self.y_1))
            
        
        
        
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
        self.tstep = rnd.randrange(0, self.T-47-1) # a random initial state in the whole year   
        self.tstep_init=self.tstep # initial timestep
        # print(self.tstep)
        
        self.tar_buy=0.17
        self.L_s=np.zeros(self.T)

        self.gen=self.data[self.tstep,0]
        self.load=self.data[self.tstep,1]
        self.minutes=self.data[self.tstep,2]
        self.sin=np.sin(2*np.pi*(self.minutes/self.min_max))
        self.cos=np.cos(2*np.pi*(self.minutes/self.min_max))
        
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
        
        self.c_T=0
        
        self.delta=(self.load+self.load_s)-self.gen
        self.delta_s=self.load_s-self.gen
        
        self.load_s=self.L_s[0] #initialize with the first element in L_s
        
        
        # if the random initial time step is after the delivery time it must not turn on
        if self.minutes >= self.t_deliver-self.T_prof*self.tstep_size:
            self.E_prof=0 #means that there is no more energy to consume 
            self.y_s=self.T_prof #means taht it connected allready the machine T_prof times
        else:
            self.E_prof=self.profile.sum()*self.dh #energy needed for the appliance
            self.y_s=0 # means that it has never connected the machine
        
        
        
        
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
        
        self.cost=max(0,self.delta)*self.tar_buy*self.dh + min(0,self.delta)*self.tar_sell*self.dh
        self.cost_s=max(0,self.delta_s)*self.tar_buy*self.dh + min(0,self.delta_s)*self.tar_sell*self.dh
    
        # observation=np.array((self.tstep,self.minutes,self.sin,self.cos,self.gen,self.gen0,self.gen1,self.gen3,self.gen6, self.load,self.load_s,self.delta,self.delta_s,self.y,self.y_1,self.y_s,self.cost,self.cost_s,self.tar_buy,self.E_prof))
    
        return self.get_obs()


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
                         self.gen3,
                         self.gen6,
                         self.load,
                         self.load_s,
                         self.delta,
                         self.delta_s,
                         self.y,
                         self.y_1,
                         self.y_s,
                         self.cost,
                         self.cost_s,
                         self.tar_buy,
                         self.E_prof), dtype=np.dtype('float32'))
    