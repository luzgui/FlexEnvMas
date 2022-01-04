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
        
        self.soc_max=soc_max
        self.eta=eta
        self.data=data #we need to import the unflexible load and PV production. 
        # data is an array with load in column 0 and pv profucition in column 1
        self.T=len(self.data) #Time horizon 
        self.dh=30*(1/60) #Convertion factor energy-power
        self.R_Total=[]
        self.n_episodes=0
        
        
        self.__version__ = "0.0.1"
        # Modify the observation space, low, high and shape values according to your custom environment's needs
        
        
        #limits on states
        
        highlim=np.array([self.T, 10,10,self.soc_max,self.soc_max,1,100,1,100])
        lowlim=np.array([0,0,0,0,0,0,-100,0,-100])
        
        
        # limits on batery 
        self.charge_lim=charge_lim
        self.discharge_lim=-charge_lim
        
        
        #State/observation definition
        self.observation_space = gym.spaces.Box(low=np.float32(lowlim), high=np.float32(highlim), shape=(9,),dtype='float32')
        
        """
        State definition
        # 0-->t : time slot
        # 1-->gen : PV generation at timeslot t
        # 2-->load : Load at timeslot t
        # 3--> SOC : State of charge
        # 4-->SOC_1 : State of cherge t-1
        # 5-->tar : Electricity tariff at timeslot t
        # 6-->R  : total reward per episode
        # 7 --> sc: self-consumption
        # 8 --> r :reward
        
        """
        
        #Actions
        # The battery actions are the selection of charging power but these are discretized between zero and charge_lim
        self.charge_steps=np.linspace(0,self.charge_lim,int((self.charge_lim/0.2)+1)) #definition of actions
        self.action_space = gym.spaces.Discrete(len(self.charge_steps)-1)

        #TODO: Battery can only charge but we want it to be abble to also discharge. that way we can increase the number of days considered. Define actions so that discharging to feed the load is possible 


    def get_load(self,action):
        #translates the actionm number into a charging power in charge_steps
        return self.charge_steps[action]


    def get_reward(self,action):
        
        action_=self.get_load(action)
        """
        The objective of the battery is to maximize self-consumption, i.e charge the battery when there is PV production available. 
        """
        
        # TODO: define a new reward based on total energy cost
    
        # reward=0
        if self.g!=0:
            eps_sc=0.3 #allowed variation percentage 
            eps_soc=1
            sc_opt=1
            
            #If charging translates into a greater SC then r=1
            if (self.sc <= (1+eps_sc)*sc_opt and self.sc >= (1-eps_sc)*sc_opt) and (self.soc <=self.soc_max):
                reward=np.float(1)
            
            else:
                reward=-action_
        else:
            reward=-action_
        
        return reward
            

        
        
    
    
    
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
        
        # self.t=self.t
        
        # if self.t>=len(self.data)-1 or self.soc >= self.soc_max:
        if self.t==len(self.data)-1:  
            done=True
            self.R_Total.append(self.R)
            print(self.R)
            
            self.n_episodes+=1
            # print(self.n_episodes)
            return self.reset(), 0,done, {}
        
       
        
        self.t+=1
        # print(self.t)
        # print(action_)
        self.g=self.data[self.t][0]
        self.l=self.data[self.t][1]
        self.soc1=self.soc
        # print(action)
        self.soc+=self.eta*action_*self.dh #Next SOC is multiplied by energy-power convert and efficiency (action in kW, soc in kWh) 
        # print(self.soc)
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


    

    def reset(self):
        """
        Reset the environment state and returns an initial observation

        Returns
        -------
        observation (object): The initial observation for the new episode after reset
        :return:
        """
        
        done=False
        
        
        # We can choose to reset to a ransom state or to t=0
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
        
    fig, ax = plt.subplots()
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
    plt.show()    
    time.sleep(0.1)
      
def reward_plot(R):
    """Creates a plot with the evolution of rewards along episodes"""
    fig, ax = plt.subplots()
    ax.plot(R)
    
    ax.set(xlabel='episodes', ylabel='reward',
           title='evolution of rewards with the number of episodes')
    ax.grid()
    
    # fig.savefig("test.png") # uncomment to save figures
    plt.show()

def get_actions(action_track,env):
    actions=env.actions
    action_vector=[]
    action_vector=[actions[k] for k in action_track]
    return action_vector
                
