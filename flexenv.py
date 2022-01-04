import gym
import numpy as np
import random as rnd

import matplotlib.pyplot as plt
import time

class FlexEnv(gym.Env):
    """
    A template to implement custom OpenAI Gym environments

    """

    metadata = {'render.modes': ['human']}
    
    def __init__(self, data,soc_max,eta,charge_lim):
        
        self.soc_max=soc_max
        self.eta=eta
        self.data=data #we need to import the unflexible load and PV production. 
        # data is an array with load in column 0 and pv profucition in column 1
        self.T=len(self.data)
        self.dh=30*(1/60)
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
        
        self.charge_steps=np.linspace(0,self.charge_lim,int((self.charge_lim/0.2)+1))
        
        
        self.observation_space = gym.spaces.Box(low=np.float32(lowlim), high=np.float32(highlim), shape=(9,),dtype='float32')
        
        # 0-->t : time slot
        # 1-->gen : PV generation at timeslot t
        # 2-->load : Load at timeslot t
        # 3--> SOC : State of charge
        # 4-->SOC_1 : State of cherge t-1
        # 5-->tar : Electricity tariff at timeslot t
        # 6-->R  : total reward per episode
        # 7 --> sc: self-consumption
        # 8 --> r :reward
        
        # Modify the action space, and dimension according to your custom environment's needs
        self.action_space = gym.spaces.Discrete(len(self.charge_steps)-1)

    def get_load(self,action):
        #translates the actionm number into a charging power in charge_steps
        return self.charge_steps[action]


    def get_reward(self,action):
        
        action_=self.get_load(action)
    
        # if self.Etc < self.E_max:
        #     reward=-1
        
        # if self.Ec==self.E_max and self.l+self.load < self.g:
        #     reward=1
            
        # else:
        #     reward=0
            
        #reward
        # reward=-np.float((self.l+action_-self.g)*self.dh*self.tar)
        # reward=float((action+self.l-self.g)*self.tar)
    
        # if self.l+action_-self.g == 0:
        #     reward=1
        # else:
        #     reward=-1*action_
        
        
        # else:
        #     reward=0
        # reward=self.g*action
        
        
        ## best so far
        # if self.g>self.l:
        #     reward=float(np.minimum((self.l+action_),self.g)/self.g)
        
        # elif self.g<=self.l:
        #     reward=-1*action_
        
        
        
        # ## best so far
        # if self.g>=self.l:
        #     sc=(self.l+action_)/self.g
        #     eps_max=1.1
        #     eps_min=0.9
        #     # print(sc)
        #     if (sc <= eps_max and sc >= eps_min):
        #         reward=np.float(1)
        #     elif (sc > eps_max or sc < eps_min):
        #         # reward=-np.float(action_)
        #         reward=-action_
        # else:
        #         # reward=-np.float(action_)
        #         reward=-action_
           
        # return reward
    
        # reward=0
        if self.g!=0:
            eps_sc=0.3 #allowed variation percentage 
            eps_soc=1
            sc_opt=1
            
        
            # if ((self.sc <= (1+eps_sc)*sc_opt and self.sc >= (1-eps_sc)*sc_opt) and (self.soc <= ((1+eps_soc)*self.soc_max and self.sc >= (1-eps_soc)*self.soc_max))) :
            #     reward=np.float(1)
            
            if (self.sc <= (1+eps_sc)*sc_opt and self.sc >= (1-eps_sc)*sc_opt) and (self.soc <=self.soc_max):
                reward=np.float(1)
                # reward=action_
                
            # if (self.sc <= (1+eps_sc)*sc_opt and self.sc >= (1-eps_sc)*sc_opt):
                    # reward=np.float(1)
                
            # elif self.soc <=self.soc_max:
            #     reward=np.float(2)
                
                
            # if (self.sc <= (1+eps_sc)*sc_opt and self.sc >= (1-eps_sc)*sc_opt):
            #         reward=np.float(10)
            
            # if (self.sc <= (1+eps_sc)*sc_opt and self.sc >= (1-eps_sc)*sc_opt):
            #     reward=np.float(1)
                
            # elif (self.sc > ((1+eps_sc)*sc_opt and self.sc < (1-eps_sc)*sc_opt)):
            #     reward=-np.float(action_)
                # reward=-action_
            else:
            # reward=-np.float(action_)
                reward=-action_
        else:
            reward=-action_
        
        return reward
            
            
        # if self.Ec==0 and self.g > 0:
        #     reward=self.g
        # else:
        #     reward=0
        
        
    
    
    
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
        self.soc+=self.eta*action_*self.dh #multiplied by energy-power convert and efficiency 
        # print(self.soc)
        self.tar=0.17
        
        # self consumption
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
        
        # observation=[self.t,self.g,self.l,self.Ec,self.Etc,self.tar]
        
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
    
    ax.plot(sol+load,label='load+sol')
    ax.plot(sol,label='sol')
    ax.plot(soc,label='soc')
    
    ax.plot(gen,label='gen')
    ax.grid()
    ax.legend()
    ax.set(xlabel='Time of the day', ylabel='kW/kWh',
           title='Schedulling battery solution')
    plt.show()    
    time.sleep(0.1)

    
def reward_plot(R):
    
    fig, ax = plt.subplots()
    ax.plot(R)
    
    ax.set(xlabel='episodes', ylabel='reward',
           title='evolution of rewards with the number of episodes')
    ax.grid()
    
    # fig.savefig("test.png")
    plt.show()

    
def get_actions(action_track,env):
    actions=env.actions
    action_vector=[]
    action_vector=[actions[k] for k in action_track]
    return action_vector
                
