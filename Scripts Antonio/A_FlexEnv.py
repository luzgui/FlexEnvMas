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
        highlim=np.array([self.T, 10,10,self.soc_max,self.soc_max,1,100,1,100,1000,1000,1000,1000,1000,1000,1000])
        lowlim=np.array([0,0,0,0,0,0,-100,0,-100,0,0,0,0,0,0,0])
        
#         high = np.array([self.x_threshold * 2,
#                          np.finfo(np.float32).max, # Infinito
#                          self.theta_threshold_radians * 2,
#                          np.finfo(np.float32).max,],dtype=np.float32) # Infinito
        
#          self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # limits on batery
        self.charge_lim=charge_lim # The battery charges charge_lim when it charges
        self.discharge_lim=-charge_lim # The battery discharges charge_lim when it discharges

        # Observation space
        self.observation_space = gym.spaces.Box(low=np.float32(lowlim), high=np.float32(highlim), shape=(16,), dtype='float32')

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
        # 9 --> grid: grid energy
        # 10 --> PV: Energy used from the battery

        # 11 --> delta: energy defict/super-avit
        # 12 --> grid import/export: 
        # 13 --> Energy Balance 
        # 14 --> energy cost
        # 15 --> Total energy cost


        """
        self.varnames=('time','gen','load','soc','soc_1','tar','R','sc','r','grid','PV_used','delta','I/E','bal','c','Tc')

        #Actions
        # The battery actions are the selection of charging power but these are discretized between zero and charge_lim
        
        # The charge steps are defined: 0 is the min, self.charge_lim is the max and there's 
        # (self.charge_lim/minimum_charge_step)+1) numbers in the array. Basically it increases the minimum charge step each number
        self.minimum_charge_step =min_charge_step
        # self.charge_steps=np.linspace(0,self.charge_lim,int((self.charge_lim/self.minimum_charge_step)+1)) #definition of charge actions
        # self.discharge_steps=np.linspace(-self.charge_lim,0,int((self.charge_lim/self.minimum_charge_step)+1)) #definition of discharge actions
        self.chargedischarge_steps=np.linspace(-self.charge_lim,self.charge_lim,int((self.charge_lim/self.minimum_charge_step)+1)) #definition of charge and discharge actions
        
        # Inserting a zero in the middle of the actions vector
        self.chargedischarge_steps = np.insert(self.chargedischarge_steps, int(len(self.chargedischarge_steps)/2), 0) 
        
        self.action_space = gym.spaces.Discrete((len(self.chargedischarge_steps)))
        

        
        # Modification António
        # self.discharge_steps=np.linspace(-charge_lim,0,int((self.charge_lim/self.minimum_charge_step)+1)) #definition of actions
        # self.action_space2 = gym.spaces.Discrete(len(self.discharge_steps)-1)

        #TODO: Battery can only charge but we want it to be abble to also discharge. that way we can increase the number of days considered. Define actions so that discharging to feed the load is possible


    # Modification António
    # def get_charge(self,action):
    #     #translates the action number into a charging power in charge_steps
    #     return self.charge_steps[int((action-1)/len(self.charge_steps))] 
    
    # def get_discharge(self,action):
    #     #translates the action number into a charging power in discharge_steps
    #     return self.discharge_steps[int(action-1-int((action-1)/len(self.charge_steps))*len(self.charge_steps))]
    
    def get_charge_discharge(self,action):
        #translates the action number into a charging power in discharge_steps
        return self.chargedischarge_steps[action]
    
    # def get_discharge(self,action):
    #     #translates the action number into a charging power in charge_steps
    #     return self.discharge_steps[action]


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
        
        action_=self.get_charge_discharge(action)
        
        
        ## Modification António
        # if self.g == 0:
        #     action_charge = 0

        # self.t=self.t

        # if self.t>=len(self.data)-1 or self.soc > self.soc_max or self.soc <= 0:
        if self.t==len(self.data)-1:
            done=True
            self.R_Total.append(self.R)
            self.c_Total.append(self.Totc)
            print(self.R)
            # print('abort')

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
        # print(action_charge)
        
        # O self.data é o data das duas colunas verticais do auxfunctions
        self.g=self.data[self.t][0] # valor da generation para cada instante
        self.l=self.data[self.t][1] # valor da load para cada instante
        
        self.soc1 = self.soc
        # print(action)
        
        self.soc+=self.eta*(action_)*self.dh
        
        ## Modification Antonio
        # if self.soc > self.l:
        #     self.soc-=self.l
        
        ## Modification Antonio
        # (It goes bad as the actions can go in the wrong way but as self.g is 0, it doesn't affect the SOC)
        # self.soc += self.eta*action_charge*self.dh*self.g #TODO: Em vez de concentrar nos SOC's concentro-me nas actions
        
        ## Modification (Antonio) - Trying to maintain 0<SC<SOC_max
        # if self.soc < self.soc_max:
        #     self.soc = self.soc
        # else:
        #     self.soc = self.soc_max
        
        self.tar=0.17 # grid tariff in €/kWh
        
        # If the action is to discharge, then the self.grid is the difference between the discharge amount and the load needed
        # If the action is to charge the battery or do nothing to the battery, then the self.grid is the load needed for that instant 
        if action_<0:
            if abs(action_)<=self.l:
                self.grid =self.l+action_
            else:
                self.grid=0
        else:
            self.grid =self.l
            
        # If the action is to discharge, the amount the battery discharges is the amount of energy used because of the PV    
        if action_<0:
            self.PV+=abs(action_)

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
        
        # load = self.l
        
        # energy defict/super-avit
        self.delta=self.g-self.l
        
        # grid import/export:
        self.grid_2=action_-self.delta
        
        # Energy Balance 
        self.bal=self.grid_2-action_+self.delta
        
        #energy cost
        
        # if self.grid>=0: #import is a cost
        #     self.c=self.tar*self.grid*self.dh
        # elif self.grid<0: #export is a revenue
        #     self.c=-(0.1*self.tar)*abs(self.grid)*self.dh # export its payed at 10% of the import cost
        
        self.c=self.tar*self.grid_2*self.dh
        self.Totc+=self.c
        
        

        info={}

        observation=np.array((self.t,self.g,self.l,self.soc,self.soc1,self.tar,self.R,self.sc,self.r,self.grid,self.PV,self.delta,self.grid_2,self.bal,self.c, self.Totc),dtype='float32')
        # print(observation)
        # print(observation,reward,done)

        return observation, reward, done, info
    
    
    def get_reward(self,action):

        # Modification Antonio
        # action_charge=self.get_charge(action)       
        # action_discharge=self.get_discharge(action)
        
        action_=self.get_charge_discharge(action)
        """
        The objective of the battery is to maximize self-consumption, 
        i.e charge the battery when there is PV production available.
        """
        
        #Defined in this way rewards are greater if actions are equal to delta 
        if self.soc <= self.soc_max and self.soc >= 0:
            r=1/((action_-self.delta))
        else:
            r=-abs(action_)
     
        
        # TODO: define a new reward based on total energy cost

        # reward=0
        # if self.g>0:
        # if self.delta > 0:
        #     eps_sc=0.3 #allowed variation percentage
        #     eps_soc=1
        #     sc_opt=1

        #     #If charging translates into a greater SC then r=1
        #     # if (self.sc <= (1+eps_sc)*sc_opt and self.sc >= (1-eps_sc)*sc_opt) and (0 < self.soc <=self.soc_max) and (-0.05 < self.grid_2 < self.l ):
        #     #     reward1=np.float(1)
        #     # else:
        #     #     reward1=-abs(action_)
                
        #     if self.soc <= self.soc_max and self.soc >= 0:
        #         r=1/((action_-self.delta)+0.001)
        #         # r=1/((abs(action_)-abs(self.delta)+0.001))
        #         # r=(min((self.l+action_),self.g)/self.g)+(self.soc/self.soc_max)
        #     else:
        #         r=-abs(action_)
        # # elif self.g==0:
        # elif self.delta < 0:
        #     if self.soc <= self.soc_max and self.soc >= 0:
        #     # r=min(min(action_,0),self.l)/self.l
        #         r=1/((action_-self.delta)+0.001)
        #         # r=1/((abs(action_)-abs(self.delta)+0.001))
            
        #     else:
        #             r=-abs(action_)    
        # else:
        #     r=-abs(action_)
                
        # elif self.g==0 and action_>0:
            # r=-abs(action_)
            
        # else:
        #     r=-abs(action_)
            
        reward=r
        
        
        # Modification (António)
        # if self.soc > self.soc_max: # If the SOC becomes bigger then the SOC_max the reward is really negative (Probably more useful when the battery can discharge)
        #     # reward = -300    
        #     reward2 = -3000*(self.soc-self.soc_max) # The bigger the difference from the SOC_max, the worse
        # else:
        #     reward2 =0
        # # if self.soc == self.soc_max:
        # #     reward = 60
        
        # # if self.soc >= self.soc_max and action>=0: # If the battery charges but the SOC is already at its max (or bigger, which is not feasible in real life), the reward is really negative
        # #     # reward = -300*action                  # Doesn't make much sense to punish accordingly to the action, it should be awful to charge when the SOC is at SOC_max
        # #     reward = -600
        # # Tested and it went well with 5e5 steps
        
        # if self.soc >= self.soc_max and self.soc > self.soc1: # If the battery charges but the SOC is already at its max (or bigger, which is not feasible in real life), the reward is really negative
        #     # reward = -300*action                  # Doesn't make much sense to punish accordingly to the action, it should be awful to charge when the SOC is at SOC_max
        #     reward3 = -800
        # else:
        #     reward3 = 0
        # # Tested but probably not going well
        
        # # If the battery charges whwn there is no generation, the reward is negative.
        # # The battery charges when the current SOC is bigger then the previous one 
        # if self.g <=0 and self.soc > self.soc1: 
        #     reward4 = -4000
        # else:
        #     reward4= 0 
        # # Tested and it went well
            
        # # If the battery charges when there is generation, the reward is positive.
        # # The battery charges when the current SOC is bigger then the previous one 
        # # It can't be too positive otherwise it will charge every instant there is generation, which is not good
        # # (This doesn't make much sense)
        # # if self.g > 0 and self.soc > self.soc1: 
        # #     reward = 200*self.g # Trying to put the emphasis where there is more generation
        
        # # The objective is to go as fast as possible to the self.soc_max when there is generation.
        # # As such, for each instant that generation is available, the bigger the difference between SOC and SOC_max, the worse it is
        # # (This influences the code)
        # # if self.g > 0 and self.soc < self.soc_max: 
        # #     reward5 = -500*(self.soc_max-self.soc)
        # # else:
        # #     reward5 =0
            
        # # Every instant the SOC is between 0 and the SOC, it gets a positive reward
        # if 0 <= self.soc <= self.soc_max:
        #     reward6 = 300
        # else:
        #     reward6 =0
        # # Tested and it went well
            
        # # If the SOC is already at SOC_max and there is generation in that instant but the battery doesn't charge, it gets a positive reward     
        # # if self.soc == self.soc_max and self.soc == self.soc1 and self.g>0:
        # #     reward7 = 1000
        # # else:
        # #     reward7 = 0
            
        # # If the SOC is already close to the SOC_max and there is generation in that instant but the battery doesn't charge, it gets a positive reward
        # # 0.2 is the minimum step it can increase charging, so if it is less than the minimal charge away from the optimal value, it gets a positive reward
        # if self.soc_max-self.minimum_charge_step <=self.soc < self.soc_max and self.soc == self.soc1 and self.g>0:
        #     reward7 = 1000
        # else:
        #     reward7 = 0
        
        # # Modification (António)
        
        # # A way to prevent the discharge to be bigger than the load at each instant
        # if action_<0 and abs(action_)>self.l:
        #     reward8 = -50*((abs(action_)-self.l))
        # else:
        #     reward8=0
        
        # # A way to prevent the discharge to be smaller than the load at each instant
        # # (This nfluences the actions, not desirable)
        # # if action_<0 and self.l>0 and abs(action_)<self.l:
        # #     reward9 = -50*((self.l-abs(action_)))
        # # else:
        # #     reward9 = 0
            
        # # A way to prevent the SOC beocming less than 0
        # if self.soc<0:
        #     reward10 = -1000
        # else:
        #     reward10 = 0
        
        # # A way to punish if the SOC is smaller or equal to zero and the action is to take some load            
        # if self.soc<=0 and action_<0:
        #     reward11= -2000
        # else:
        #     reward11=0
       
        # # A way to reward if the SOC is bigger than zero and the action is to take some load         
        # # if self.soc>0 and action_discharge<0:
        # #     reward12 = 50
        # # else:
        # #     reward12 = 0
        
        # # If the action is to take an amount close to the load it is good.
        # # The self.l-minimum step is the nearest the action can be to the load
        # # if self.l-self.minimum_charge_step<action_<self.l+self.minimum_charge_step:
        # #     reward13=200
        # # else:
        # #     reward13=0
            
        # # If the load is bigger than 0 but there is no SOC than the action should be 0    
        # # if self.l>0 and self.soc<=0 and action_==0:
        # #     reward14=700
        # # else:
        # #     reward14 = 0
        
        # reward15 = -self.grid*50
        
        # reward = reward1+reward2+reward3+reward4+reward6+reward7 + reward8+reward10+reward11 +reward15
            
        return reward


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
        self.soc=0
        self.soc1=0
        self.tar=0.17
        self.R=0
        
        # self consumption
        #we are starting at t=0 with sc=0 because there is no sun. the initial state depends on the action and we cannot have actions on reset()
        self.sc=0
        
        self.PV=0
        
        self.delta=self.g-self.l
        self.r=1/((0-abs(self.delta)+0.001))
        
        self.grid=-self.delta
        self.grid_2=-self.delta
        
        self.bal=0
        self.c=self.tar*self.grid_2*self.dh
        self.Totc=self.c

        observation=np.array((self.t,self.g,self.l,self.soc,self.soc1,self.tar,self.R,self.sc,self.r,self.grid,self.PV,self.delta,self.grid_2,self.bal,self.c, self.Totc),dtype='float32')

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