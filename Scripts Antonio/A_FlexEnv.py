import gym
import numpy as np

import matplotlib.pyplot as plt
import time

class FlexEnv(gym.Env):
    """
    A custom OpenAI Gym environment for managing a electrical battery
    """

    def __init__(self, data,soc_max,eta,charge_lim,min_charge_step):
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
        self.c_Total=[]
        self.n_episodes=0
        # Modificação António
        self.soc=0
        self.t=0
        self.grid=0
        self.PV=0

        self.__version__ = "0.0.1"
        # Modify the observation space, low, high and shape values according to your custom environment's needs

        #limits on states
        
        highlim=np.array([10,10,self.soc_max,self.soc_max,10,10,10])
        lowlim=np.array([0,0,0,0,-10,-10,-10])
        
        #Names of variables
        self.varnames=('gen','load','soc','soc_1','r','delta','I/E')
        
        #Number of variables
        self.var_dim=len(self.varnames)
        
        """
        Variables definitions (not all are being used)
        
        # t : time slot
        # gen : PV generation at timeslot t
        # load : Load at timeslot t
        # SOC : State of charge
        # SOC_1 : State of charge t-1
        # tar : Electricity tariff at timeslot t
        # R  : total reward per episode
        # sc: self-consumption
        # r :reward
        
        # delta: energy defict/super-avit
        # grid import/export: 
        # Energy Balance 
        # energy cost
        # Total energy cost
        
        
        """

        # limits on batery


        # Observation space
        self.observation_space = gym.spaces.Box(low=np.float32(lowlim), high=np.float32(highlim), shape=(self.var_dim,), dtype='float32')

        
        #Actions
        # The battery actions are the selection of charging power but these are discretized between zero and charge_lim
        
        # The charge steps are defined: 0 is the min, self.charge_lim is the max and there's 
        # (self.charge_lim/minimum_charge_step)+1) numbers in the array. Basically it increases the minimum charge step each number     
        
        self.charge_lim=charge_lim # The battery charges charge_lim when it charges
        self.discharge_lim=-charge_lim # The battery discharges charge_lim when it discharges
        self.minimum_charge_step =min_charge_step

        #Number of actions (you can tweek this to get different number of actions)
        self.num_actions=4

        self.chargedischarge_steps=np.linspace(-self.charge_lim,self.charge_lim,int((self.num_actions*self.charge_lim/self.minimum_charge_step)+1)) #definition of charge and discharge actions
        
        # Inserting a zero in the middle of the actions vector
        self.chargedischarge_steps = np.insert(self.chargedischarge_steps, int(len(self.chargedischarge_steps)/2), 0) 
        
        self.action_space = gym.spaces.Discrete((len(self.chargedischarge_steps)))
        

        #TODO: Battery can only charge but we want it to be abble to also discharge. that way we can increase the number of days considered. Define actions so that discharging to feed the load is possible

    
    def get_charge_discharge(self,action):
        #translates the action number into a charging power in discharge_steps
        return self.chargedischarge_steps[action]
    


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
        
        action_=self.get_charge_discharge(action)
    
    
        if self.t==len(self.data)-1:
            
            self.R_Total.append(self.R)
            self.c_Total.append(self.Totc)
            print(self.R)
            # print('abort')

            self.n_episodes+=1
            done=True
            # print(self.n_episodes)
            return np.array((self.g,self.l,self.soc,self.soc1,self.r,self.delta,self.grid),dtype='float32'),0,done, {}
        
        
        self.t+=1
        # print(self.t)
        # print(action_charge)
        
        # O self.data é o data das duas colunas verticais do auxfunctions
        self.g=self.data[self.t][0] # valor da generation para cada instante
        self.l=self.data[self.t][1] # valor da load para cada instante

        self.soc1 = self.soc #State of Charge at t-1
        
        self.soc+=self.eta*(action_)*self.dh # New State of Charge
        
        self.tar=0.17 # grid tariff in €/kWh #Electricity tariff
        
        # self consumption (it should be 0<SC<1 but it is possible to have SC>1)
        #TODO: Think about how to maintain 0<SC<1
        
        if self.g!=0:
            self.sc=(self.l+action_)/self.g
        else:
            self.sc=0

        reward=self.get_reward(action)

        self.R+=reward #Total Reward is the summation of previous rewards
        self.r=reward #Present reward
        
        
        # energy defict/super-avit
        self.delta=self.g-self.l
        
        # grid import/export:
        self.grid=action_-self.delta
        
        # Energy Balance 
        self.bal=self.grid_2-action_+self.delta
        
        #energy cost
        self.c=self.tar*self.grid*self.dh
        self.Totc+=self.c
        
        

        info={}

        observation=np.array((self.g,self.l,self.soc,self.soc1,self.r,self.delta,self.grid),dtype='float32')
        # print(observation)
        # print(observation,reward,done)

        return observation, reward, done, info
    
    
    def get_reward(self,action):
        
        # Modification Antonio
        # action_charge=self.get_charge(action)       
        # action_discharge=self.get_discharge(action)
        
        action_=self.get_charge_discharge(action)
        """
        The objective of the battery is to charge when there is PV gen available and discharge when there is energy deficit
        """
        
        #Defined in this way rewards are greater if actions are equal to delta 
        
        #Hipotese 1
        
        # if self.soc <= self.soc_max and self.soc >= 0:
        #     if (action_-self.delta) >= 0:
        #         r=-0.2*np.exp((action_-self.delta)/0.1)+3
            
        #     elif (action_-self.delta) <= 0:
        #         r=-0.2*np.exp(-(action_-self.delta)/0.4)+3
                
        #     else:
        #         r=-10*abs(action_)
        
        # else:
        #     r=-10*abs(action_)
        
        
        #Hipotese 2        
        
        if self.soc <= self.soc_max and self.soc >= 0:
            r=np.exp(-(action_-self.delta)**2/0.1) # Gaussian function that approximates a tep function
        
        else:
            r=0
        
        return r
    
    
    def reset(self):
        """
        Reset the environment state and returns an initial observation

        Returns:
        -------
        observation (object): The initial observation for the new episode after reset
        :return:
        
        we are assuming that the agnet takes no action at t=0    
        
        """

        done=False
        
        # We can choose to reset to a random state or to t=0
        self.t=0 # start at t=0
        # self.t = rnd.randrange(0, 46, 2) # a random initial state

        self.g=self.data[0,0]
        self.l=self.data[0,1]
        
        #Initial SOC (can be tweeked)
        # self.soc=0
        self.soc=0.2*self.soc_max
        
        
        self.soc1=0
        self.tar=0.17
        self.R=0
        
        # self consumption
        #we are starting at t=0 with sc=0 because there is no sun. the initial state depends on the action and we cannot have actions on reset()
        self.sc=0
        
        # self.PV=0
        
        self.delta=self.g-self.l
        self.r=1/((0-abs(self.delta)+0.001))
        
        # self.grid=-self.delta
        self.grid_2=-self.delta
        
        self.bal=0
        self.c=self.tar*self.grid_2*self.dh
        self.Totc=self.c

        observation=np.array((self.g,self.l,self.soc,self.soc1,self.r,self.delta,self.grid),dtype='float32')

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

# In[]:
# charge_lim =2
# minimum_charge_step=0.5

# charge_steps=np.linspace(0,charge_lim,int((charge_lim/minimum_charge_step)+1)) #definition of actions
# charge_steps

# discharge_steps=np.linspace(-charge_lim,0,int((charge_lim/minimum_charge_step)+1))
# discharge_steps

# action=23
# int((action-1)/len(charge_steps))
# int(action-1-int((action-1)/len(charge_steps))*len(charge_steps))