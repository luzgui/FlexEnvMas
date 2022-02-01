import gym
import numpy as np

import matplotlib.pyplot as plt
import time

class FlexEnv(gym.Env):
    """
    A custom OpenAI Gym environment for managing a electrical battery
    """

    def __init__(self, data,soc_max,eta,charge_lim,min_charge_step, reward_type):
        """
        data: a pandas dataframe with number of columns N=# of houses or loads and time-horizon T=Total number of timesteps
        soc_max: maximum battery state-of-charge
        eta: Charging efficiêncy
        charge_lim: maximum charging power
        """

        self.reward_type=reward_type

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
        self.grid=0
        self.I_E = 0
        self.bat_used=0
        self.delta=0

        self.__version__ = "0.0.1"
        
        # Modify the observation space, low, high and shape values according to your custom environment's needs
        highlim=np.array([10,10,self.soc_max,self.soc_max,10,10,10])
        lowlim=np.array([0,0,0,0,-10,0,-10])
        
        #Names of variables
        self.varnames=('gen','load','soc','soc_1','delta','grid','I/E')
        
        #Number of variables
        self.var_dim=len(self.varnames)
        

    
        # Observation space
        self.observation_space = gym.spaces.Box(low=np.float32(lowlim), high=np.float32(highlim), shape=(self.var_dim,), dtype='float32')

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
        self.charge_lim=charge_lim # The battery has a limit of charging of charge_lim
        self.discharge_lim=-charge_lim # The battery has a limit of discharging of charge_lim
        self.minimum_charge_step =min_charge_step
        # self.charge_steps=np.linspace(0,self.charge_lim,int((self.charge_lim/self.minimum_charge_step)+1)) #definition of charge actions
        # self.discharge_steps=np.linspace(-self.charge_lim,0,int((self.charge_lim/self.minimum_charge_step)+1)) #definition of discharge actions
        self.chargedischarge_steps=np.linspace(-self.charge_lim,self.charge_lim,int((self.charge_lim/self.minimum_charge_step)+1)) #definition of charge and discharge actions
        
        # Inserting a zero in the middle of the actions vector
        self.chargedischarge_steps = np.insert(self.chargedischarge_steps, int(len(self.chargedischarge_steps)/2), 0)
        self.action_space = gym.spaces.Discrete((len(self.chargedischarge_steps)))
        

    
    def get_charge_discharge(self,action):
        #translates the action number into a charging power in discharge_steps
        return self.chargedischarge_steps[action-1]
    


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
        
        
    
        reward=self.get_reward(action, reward_type=self.reward_type)
        
        # if self.t>=len(self.data)-1 or self.soc >= self.soc_max:
        if self.t==len(self.data)-1:
            # done=True
            self.R_Total.append(self.R)
            print(self.R)

            self.n_episodes+=1
            done = True

            return np.array((self.g,self.l,self.soc,self.soc1,self.delta,self.grid,self.I_E),dtype='float32'),0,done, {}
        else:
            done = False

        
        self.t+=1
        
        # O self.data é o data das duas colunas verticais do auxfunctions
        self.g=self.data[self.t][0] # valor da generation para cada instante
        self.l=self.data[self.t][1] # valor da load para cada instante
        
        
        self.delta = self.g-self.l
        self.soc1 = self.soc
        
        
        # Charging discharging dynamics
        if action_ > 0: #charging
            self.soc+=self.eta*(action_)*self.dh # New State of Charge
        elif action_ < 0: #discharging
            self.soc+=((action_)/self.eta)*self.dh
        elif action_ == 0: #do nothing
            self.soc = self.soc
        
        
        self.tar=0.17 # grid tariff in €/kWh
        
        # If the action is to discharge, then the self.grid is the difference between the discharge amount and the load needed
        # If the action is to charge the battery or do nothing to the battery, then the self.grid is the load needed for that instant 
        if action_<0: # Discharge the battery
            if abs(action_)<=abs(self.delta):
                self.grid =abs(self.delta)-abs(action_)
            else:
                self.grid=0
        else: # Charge the battery
            if self.delta < 0:
                self.grid = abs(self.delta)
            else:
                self.grid = 0
        
                
        # IMport/Export
        self.I_E = action_-self.delta
            
        # If the action is to discharge, the amount the battery discharges is the amount of energy used   
        if action_<0:
            self.bat_used+=abs(action_)

        # self consumption (it should be 0<SC<1 but it is possible to have SC>1)
        #TODO: Think about how to maintain 0<SC<1
        
        if self.g!=0:
            self.sc=(self.l+action_)/self.g
        else:
            self.sc=0

        
        # print(reward)
        # reward=self.get_reward(action, reward_type=self.reward_type)
        self.R+=reward
        self.r=reward
        
        
        #energy cost
        self.c=self.tar*self.grid*self.dh

        info={}

        observation=np.array((self.g,self.l,self.soc,self.soc1,self.delta,self.grid,self.I_E),dtype='float32')
        # print(observation)
        # print(observation,reward,done)

        return observation, reward, done, info
    


    def get_reward(self,action, reward_type):

        # Modification Antonio
        # action_charge=self.get_charge(action)       
        # action_discharge=self.get_discharge(action)
        
        
        
        action_=self.get_charge_discharge(action)
        
        
        #hipothesis 1
        if reward_type==1:
        
        
        
        # Rewards mais geral possível (soc, g, soc_max, l, minimum_charge_step,action_,grid,delta)
        # Charge
        
            if action_ > 0:
                if self.soc_max>= self.soc >= 0 and self.delta>0 and action_<=self.delta:
                    reward_charge = abs(action_)*2
                else:
                    reward_charge =-abs(action_)*2
            else:
                reward_charge =0
                
            # Discharge
            if action_< 0:
                if self.soc_max>= self.soc >=0 and self.delta<0:
                    
                    if abs(self.delta)<abs(action_): 
                        # Este se calahr era melhor multiplicar para os rewards positivos serem mais positivos
                        reward_discharge =abs(self.delta)-(abs(action_)-abs(self.delta)) # Tentar perceber até onde é que faz sentido se ele descarregar bue se isso é realmente mau, pode ser só indiferente
                        # if reward_discharge < 0:
                        #     reward_discharge = 0
                        
                    else:
                        reward_discharge = abs(action_)
    
                else:
                    reward_discharge = -abs(action_)*2
            else:
                reward_discharge= 0
                
            
            # Do nothing
            if action_==0 :
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
                reward=np.exp(-(action_-self.delta)**2/0.01) # Gaussian function that approximates a tep function
            
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
        

    

        observation=np.array((self.g,self.l,self.soc,self.soc1,self.delta,self.grid,self.I_E),dtype='float32')

        return observation


    def render(self, mode='human', close=False):
        """
        :param mode:
        :return:
        """
        return


def makeplot(T,soc,sol,gen,load,delta,env):

    fig, ax = plt.subplots(figsize=(10,7))
    # t=np.arange(0,T,1)
    ax.plot(load,label='load')
    # ax.plot(sol+load,label='load+bat_charge')
    ax.plot(sol,label='bat_charge')
    ax.plot(soc,label='soc')
    ax.plot(gen,label='gen')
    ax.plot(delta,label='delta')
    
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