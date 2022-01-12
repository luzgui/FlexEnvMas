import gym
import numpy as np
import random as rnd

import matplotlib.pyplot as plt
import time

class FlexEnv(gym.Env):
    """
    A custom OpenAI Gym environment for managing a electrical battery
    """

    def __init__(self, data,soc_max,eta,charge_lim):
        """
        data: a pandas dataframe with number of columns N=# of houses or loads and time-horizon T=Total number of timesteps
        soc_max: maximum battery state-of-charge
        eta: Charging efficiêncy
        charge_lim: maximum charging power
        """

        self.soc_max=soc_max  # Defined by the user (Try real values)
        self.eta=eta
        self.data=data # We need to import the unflexible load and PV production.
        # data is an array with load in column 0 and pv production in column 1
        self.T=len(self.data) # Time horizon
        self.dh=30*(1/60) # Conversion factor energy-power
        self.R_Total=[] # A way to see the evolution of the rewards as the model is being trained
        self.n_episodes=0
        # Modificação António
        self.soc=0
        self.t=0

        self.__version__ = "0.0.1"
        # Modify the observation space, low, high and shape values according to your custom environment's needs

        #limits on states
        highlim=np.array([self.T, 10,10,self.soc_max,self.soc_max,1,100,1,100])
        lowlim=np.array([0,0,0,0,0,0,-100,0,-100])
        
#         high = np.array([self.x_threshold * 2,
#                          np.finfo(np.float32).max, # Infinito
#                          self.theta_threshold_radians * 2,
#                          np.finfo(np.float32).max,],dtype=np.float32) # Infinito
        
#          self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # limits on batery
        self.charge_lim=charge_lim # The battery charges charge_lim when it charges
        self.discharge_lim=-charge_lim # The battery discharges charge_lim when it discharges

        # Observation space
        self.observation_space = gym.spaces.Box(low=np.float32(lowlim), high=np.float32(highlim), shape=(9,), dtype='float32')

        """
        State definition
        # 0-->t : time slot
        # 1-->gen : PV generation at timeslot t
        # 2-->load : Load at timeslot t
        # 3--> SOC : State of charge
        # 4-->SOC_1 : State of charge t-1
        # 5-->tar : Electricity tariff at timeslot t
        # 6-->R  : total reward per episode
        # 7 --> sc: self-consumption
        # 8 --> r :reward

        """

        #Actions
        # The battery actions are the selection of charging power but these are discretized between zero and charge_lim
        
        # The charge steps are defined: 0 is the min, self.charge_lim is the max and there's 
        # (self.charge_lim/0.2)+1) numbers in the array. Basically it increases 0.2 each number
        self.charge_steps=np.linspace(0,self.charge_lim,int((self.charge_lim/0.2)+1)) #definition of actions
        self.action_space = gym.spaces.Discrete(len(self.charge_steps)-1)

        #TODO: Battery can only charge but we want it to be abble to also discharge. that way we can increase the number of days considered. Define actions so that discharging to feed the load is possible


    def get_load(self,action):
        #translates the action number into a charging power in charge_steps
        return self.charge_steps[action]


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
        action_=self.get_load(action)
        
        ## Modification António
        # if self.g == 0:
        #     action_ = 0

        # self.t=self.t

        # if self.t>=len(self.data)-1 or self.soc >= self.soc_max:
        if self.t==len(self.data)-1:
            done=True
            self.R_Total.append(self.R)
            print(self.R)

            self.n_episodes+=1
            # print(self.n_episodes)
            return self.reset(), 0,done, {}
        
        ## Modification (António) - If needed self.g1 represents the previous generation
        # if self.t > 0:
        #     self.g1 = self.data[self.t-1][0]
        # else:
        #     self.g1 = 0
        
        # if self.t == 0:
        #     self.g1 = 0 
        # else:
        #     self.g1 = self.data[self.t-1][0] 

        
        self.t+=1
        # print(self.t)
        # print(action_)
        
        # O self.data é o data das duas colunas verticais do auxfunctions
        self.g=self.data[self.t][0] # valor da generation para cada instante
        self.l=self.data[self.t][1] # valor da load para cada instante
        
        self.soc1 = self.soc
        # print(action)
        
        self.soc+=self.eta*action_*self.dh
        
        ## Modification Antonio
        # (It goes bad as the actions can go in the wrong way but as self.g is 0, it doesn't affect the SOC)
        # self.soc += self.eta*action_*self.dh*self.g #TODO: Em vez de concentrar nos SOC's concentro-me nas actions
        
        ## Modification (Antonio) - Trying to maintain 0<SC<SOC_max
        # if self.soc < self.soc_max:
        #     self.soc = self.soc
        # else:
        #     self.soc = self.soc_max
        
        self.tar=0.17 # grid tariff in €/kWh

        # self consumption (it should be 0<SC<1 but it is possible to have SC>1)
        #TODO: Think about how to maintain 0<SC<1
        
        if self.g!=0:
            self.sc=(self.l+action_)/self.g
        else:
            self.sc=0

        reward=self.get_reward(action)
        # print(reward)
        self.R+=reward
        self.r=reward

        info={}

        observation=np.array((self.t,self.g,self.l,self.soc,self.soc1,self.tar,self.R, self.sc,self.r),dtype='float32')
        # print(observation)
        # print(observation,reward,done)

        return observation, reward, done, info
    
    
    def get_reward(self,action):

        action_=self.get_load(action)
        """
        The objective of the battery is to maximize self-consumption, 
        i.e charge the battery when there is PV production available.
        """

        # TODO: define a new reward based on total energy cost

        # reward=0
        if self.g!=0:
            eps_sc=0.3 #allowed variation percentage
            eps_soc=1
            sc_opt=1

            #If charging translates into a greater SC then r=1
            if (self.sc <= (1+eps_sc)*sc_opt and self.sc >= (1-eps_sc)*sc_opt) and (self.soc <=self.soc_max):
                reward1=np.float(1)
            else:
                reward1=-action_
        else:
            reward1=-action_
        
        # Modification (António)
        if self.soc > self.soc_max: # If the SOC becomes bigger then the SOC_max the reward is really negative (Probably more useful when the battery can discharge)
            # reward = -300    
            reward2 = -300*(self.soc-self.soc_max) # The bigger the difference from the SOC_max, the worse
        else:
            reward2 =0
        # if self.soc == self.soc_max:
        #     reward = 60
        
        # if self.soc >= self.soc_max and action>=0: # If the battery charges but the SOC is already at its max (or bigger, which is not feasible in real life), the reward is really negative
        #     # reward = -300*action                  # Doesn't make much sense to punish accordingly to the action, it should be awful to charge when the SOC is at SOC_max
        #     reward = -600
        # Tested and it went well with 5e5 steps
        
        if self.soc >= self.soc_max and self.soc > self.soc1: # If the battery charges but the SOC is already at its max (or bigger, which is not feasible in real life), the reward is really negative
            # reward = -300*action                  # Doesn't make much sense to punish accordingly to the action, it should be awful to charge when the SOC is at SOC_max
            reward3 = -800
        else:
            reward3 = 0
        # Tested but probably not going well
        
        # If the battery charges whwn there is no generation, the reward is negative.
        # The battery charges when the current SOC is bigger then the previous one 
        if self.g <=0 and self.soc > self.soc1: 
            reward4 = -4000
        else:
            reward4= 0 
        # Tested and it went well
            
        # If the battery charges when there is generation, the reward is positive.
        # The battery charges when the current SOC is bigger then the previous one 
        # It can't be too positive otherwise it will charge every instant there is generation, which is not good
        # (This doesn't make much sense)
        # if self.g > 0 and self.soc > self.soc1: 
        #     reward = 200*self.g # Trying to put the emphasis where there is more generation
        
        # The objective is to go as fast as possible to the self.soc_max when there is generation.
        # As such, for each instant that generation is available, the bigger the difference between SOC and SOC_max, the worse it is
        if self.g > 0 and self.soc < self.soc_max: 
            reward5 = -500*(self.soc_max-self.soc)
        else:
            reward5 =0
            
        # Every instant the SOC is between 0 and the SOC, it gets a positive reward
        if 0 <= self.soc <= self.soc_max:
            reward6 = 300
        else:
            reward6 =0
        # Tested and it went well
            
        # If the SOC is already at SOC_max and there is generation in that instant but the battery doesn't charge, it gets a positive reward     
        # if self.soc == self.soc_max and self.soc == self.soc1 and self.g>0:
        #     reward7 = 1000
        # else:
        #     reward7 = 0
            
        # If the SOC is already close to the SOC_max and there is generation in that instant but the battery doesn't charge, it gets a positive reward
        # 0.2 is the minimum steo it can increase charging, so if it is less than the minimal charge away from the optimal value, it gets a positive reward
        if self.soc_max-0.2 <=self.soc < self.soc_max and self.soc == self.soc1 and self.g>0:
            reward7 = 1000
        else:
            reward7 = 0
        
        reward = reward1+reward2+reward3+reward4+reward5+reward6+reward7 
            
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
        self.soc=0
        self.soc1=0
        self.tar=0.17
        self.R=0
        self.r=0
        # self consumption
        #we are starting at t=0 with sc=0 because there is no sun. the initial state depends on the action and we cannot have actions on reset()
        self.sc=0

        observation=np.array((self.t,self.g,self.l,self.soc,self.soc1,self.tar,self.R,self.sc,self.r),dtype='float32')

        return observation


    def render(self, mode='human', close=False):
        """
        :param mode:
        :return:
        """
        return


def makeplot(T,soc,sol,gen,load,env):

    fig, ax = plt.subplots(figsize=(10,7))
    # t=np.arange(0,T,1)
    ax.plot(load,label='load')
    ax.plot(sol+load,label='load+bat_charge')
    ax.plot(sol,label='bat_charge')
    ax.plot(soc,label='soc')
    ax.plot(gen,label='gen')
    
    ax.grid()
    ax.legend()
    ax.set(xlabel='Time of the day', ylabel='kW/kWh',
           title='Schedulling battery solution')
#     plt.show()
#     time.sleep(0.1)
#     return(ax)


def reward_plot(R):
    """Creates a plot with the evolution of rewards along episodes"""
#     plt.figure(figsize=(12,7))
    fig1, ax1 = plt.subplots(figsize=(10,7))
    ax1.plot(R)

    ax1.set(xlabel='episodes', ylabel='reward',
           title='evolution of rewards with the number of episodes')
    ax1.grid()

    # fig.savefig("test.png") # uncomment to save figures
#     plt.show()


def get_actions(action_track,env):
    actions=env.actions
    action_vector=[]
    action_vector=[actions[k] for k in action_track]
    return action_vector
