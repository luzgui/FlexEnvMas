import gymnasium  as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import random as rnd
import re
import math


from ray.rllib.env.multi_agent_env import MultiAgentEnv, make_multi_agent

# from shiftenvRLlib import ShiftEnv

from state_vars import *
from termcolor import colored

class ShiftEnvMas(MultiAgentEnv):
    def __init__(self,config):
        super(MultiAgentEnv).__init__()
        

        self.env_config=config
        
        #COMMUNITY /COMMON parameters
        self.num_agents=config['num_agents']
        self.agents_id=config['agents_id']
        self._agent_ids=self.agents_id

        self.info = config['env_info']

        self.reward_type=config["reward_type"]
        self.mas_setup=config['mas_setup']
        self.tar_type=config['tar_type']
        
        
        self.data=config["data"]
          
        self.tstep_size=config["step_size"]
        
        
        self.T=len(self.data.loc[self.agents_id[0]]) # Time horizon 
        self.Tw=config["window_size"] #window horizon
        self.tstep_per_day=config["tstep_per_day"]
        self.dh=self.tstep_size*(1/60.0) # Conversion factor energy-power
        self.tstep_init=0 #initial timestep in each episode
        
        #
        self.t_ahead_hours=2 #number of hours to look ahead
        self.t_ahead=self.t_ahead_hours*(60/self.tstep_size) #number of timeslots that each actual timeslot is loloking ahead (used in update_forecast) 
        #
        
        self.min_max=self.data['minutes'].max() #maximum timeslot??
        
        #get the possoble initial timeslots from data
        self.select=self.data.loc['ag1'][self.data.loc['ag1']['minutes']==0] #get all o mionutes index
        self.allowed_inits=list(self.select.index)
        
        
        
        #AGENTS INDIVIDUAL parameters
        #Appliance profile
        self.profile=config["profile"]
        
        
        #dataframes from dictionaries
        # self.T_prof={ag:[len(self.profile[ag])] for ag in self.agents_id}
        self.T_prof={ag:len(self.profile[ag]) for ag in self.agents_id}
        self.T_prof=pd.DataFrame.from_dict(self.T_prof, orient='index', columns=['T_prof'])
        
        
        # total energy needed for the appliance
        self.E_prof={ag:sum(self.profile[ag]) for ag in self.agents_id}
        
        
        self.E_prof=pd.DataFrame.from_dict(self.E_prof, orient='index', columns=['E_prof'])
        
        #this must be set as a moving variable that updates the deliver time according to the day
        self.t_deliver=config["time_deliver"]
        self.t_deliver=pd.DataFrame.from_dict(self.t_deliver, orient='index', columns=['t_deliver'])
        
        self.agents_params=pd.concat([self.T_prof,self.E_prof,self.t_deliver],axis=1)
        
        
        

        self.R_Total=[] # A way to see the evolution of the rewards as the model is being trained
        self.n_episodes=0
        
        
        #new vars
        self.L_s=np.zeros(self.T) # we make a vector of zeros to store the shiftable load profile
        # self.l_s=self.L_s[0] 
        
        #initialize with the first element in 
        self.t_shift=0
        
        
        # defining the state variables
        
        self.max_gen=100.0
        self.max_load=100.0
        
        
        #get the state variables[]
        self.state_vars, self.vars_list=get_state_vars(self)    
        self.var_class={'gen','load','delta','excess'} #this list of variables will be tested in update_forecasts()
            
    
        #extract the names of variables in env.data and take out the minutes that we dont need    
        self.ag_var_class=[k for k in self.data.keys() if 'ag' in k]

        
        
        
        self.obs=None   
            
        #Number of variables to be used
        self.var_dim=len(self.state_vars.keys())
                
        
                
        self.highlim=np.array([value['max'] for key, value in self.state_vars.items()])
        self.lowlim=np.array([value['min'] for key, value in self.state_vars.items()])
            
        
        #Action space
        # self.action_space = spaces.Discrete(2) # ON/OFF
        
        self.action_space = spaces.Discrete(2) # ON/OFF
        
        
        # Observation space      
        # self.observation_space =spaces.Dict({
        #     aid: spaces.Dict({
        #         "action_mask": spaces.Box(0.0, 1.0, shape=(self.action_space.n,)),
        #         "observations": spaces.Box(low=np.float32(self.lowlim), high=np.float32(self.highlim), shape=(self.var_dim,))}) for aid in self.agents_id})
        
        # self.observation_space = spaces.Dict({
        #     "action_mask": spaces.Box(-10.0, 10.0, shape=(self.action_space.n,)),
        #     "observations": spaces.Box(low=self.lowlim, high=self.highlim, shape=(self.var_dim,))})
        
        
        self.observation_space = spaces.Dict({
            "action_mask": spaces.Box(-10.0, 10.0, shape=(self.action_space.n,),dtype=np.float32),
            "observations": spaces.Box(low=np.float32(self.lowlim), high=np.float32(self.highlim), shape=(self.var_dim,), dtype=np.float32)})
        
        
        #debbuging
        
        # self.observation_space = spaces.Dict({
        #     "observations": spaces.Box(low=np.float32(self.lowlim), high=np.float32(self.highlim), shape=(self.var_dim,), dtype=np.float32)})
                
        #Training/testing termination condition
        
        self.done_cond=config['done_condition'] # window or horizon mode
        self.init_cond=config['init_condition'] 
        
        
        # app conection counter
        self.count=0
        
        #seed 
        self.seed=config['seed']
        
        
        
        
        #VARIABLE INITIATIATIONS
        #initiate N X var_dim dataframe for storing state
        self.state=pd.DataFrame(index = self.agents_id+['global'],columns=self.state_vars.keys())
        
        #Auxiliary variables used to compute intermediary quantities
        self.cost_aux=pd.DataFrame(index = self.agents_id+['global'],columns=['cost','cost_s','cost_s_x'])
        
        self.delta_aux=pd.DataFrame(index = self.agents_id+['global'],columns=['cost','cost_s','cost_s_x'])
        
        #Mask
        #we have a dataframe cell for each action mask
        self.mask=pd.DataFrame(index = self.agents_id, columns=np.arange(self.action_space.n))
        
        
        # self.state_aux=pd.DataFrame(index = self.agents_id)
        
        
        
        print(f'Created an multiagent environment with {self.num_agents} agents')
        print(colored('Envrironment configurations:','green'))
        print('Type of reward:', colored(self.env_config['reward_type'],'red'))
        print('Multiagent setup for reward:', colored(self.env_config['mas_setup'],'red'))
        print('Environment purpose:', colored(self.env_config['env_info'],'red'))
        
    
        self._spaces_in_preferred_format = True
    
    def reset(self):
        """
        Reset the environment state and returns an initial observation
        
        Returns:
        -------
        observation (object): The initial observation for the new episode after reset
        :return:
        """
        
        #COMMUNITY /COMMON variables (intialization equal for all agents)
        self.done=pd.DataFrame([False]*self.num_agents, index=self.agents_id,columns=['done'])
        # self.done.loc['__all__']=all(self.done.values) #does RLLIB need this in the environment or only at the output dictionary??
        
        self.R={aid:0 for aid in self.agents_id}
        
        #initial timestep
        self.tstep=self.get_init_tstep()
        self.tstep_init=self.tstep # initial timestep
        # print(self.tstep)
        self.state['tstep']=self.tstep
        #minutes
        self.minutes=self.data.loc['ag1',self.tstep]['minutes']
        self.state['minutes']=self.minutes
        #sine/cosine
        self.sin=np.sin(2*np.pi*(self.minutes/self.min_max))
        self.cos=np.cos(2*np.pi*(self.minutes/self.min_max))
        self.state['sin']=self.sin
        self.state['cos']=self.cos
        #tariffs (for now all agents get the same tariff)
        self.tar_buy,self.tar_sell=self.get_tariffs(0) #tariff for the present timestep
        self.state['tar_buy']=self.tar_buy
        self.state['tar_buy0'], _ =self.get_tariffs(1)
    
        #inititialize binary variables
        self.state[['y','y_1','y_s']]=0
        
        
        # Initialize history of actions
        # self.hist=pd.DataFrame(index = range(self.Tw),columns=self.agents_id)
        # self.hist=pd.DataFrame(index=tuple([(a,t) for a in self.agents_id for t in range(self.tstep,self.tstep+self.Tw)]), columns=['hist'])
        
        
        
        # self.hist.append(self.y) # CHECK IF NEEDED
        
        #Initial energy to consume
        self.state['E_prof_rem'].update(self.agents_params['E_prof'])

        
        #update forecasts
        self.update_forecast()
        
        
        #Initial mask: # we concede full freedom for the appliance 
        self.mask.loc[:,:]=np.ones([self.num_agents,self.action_space.n])
        
        
        
        # self.obs={"action_mask": np.array([1,1]), # we concede full freedom for the appliance 
        #       "observations": self.get_obs()
        #       }



###########   Things I dont know exactly what they make or if are important ####


        # self.L_s=np.zeros(self.T)
        # self.load_s=0   
        # self.R=0
        # self.r=0
        # self.c_T=0
        # self.t_shift=0
        # self.load_s=0 #machine starts diconected
        
        
        
        # #Costs and deltas that depend on the action are not used in observation
        
        # self.delta_c=(self.load0+self.load_s)-self.gen0
        # self.delta_s=self.load_s-self.gen0
        
        # # self.load_s=self.L_s[0] #initialize with the first element in L_s
        
        

        # (self.load0+self.load_s)-self.gen0
        # self.cost=max(0,self.delta_c)*self.tar_buy + min(0,self.delta_c)*self.tar_sell
        # self.cost_s=max(0,self.delta_s)*self.tar_buy + min(0,self.delta_s)*self.tar_sell
        
        # self.excess=max(0,-self.delta0)   
        # self.cost_s_x=max(0,self.load_s-self.excess0)*self.tar_buy + min(0,self.load_s-self.excess0)*self.tar_sell
        
        
        
        ### Conditions for random timsetep @ start
        
        # if the random initial time step is after the delivery time it must not turn on
        # if self.minutes >= self.t_deliver-self.T_prof*self.tstep_size:
        #     self.E_prof=0 #means that there is no more energy to consume 
        #     self.y_s=self.T_prof #means taht it connected allready the machine T_prof times
        # else:
        #     self.E_prof=self.profile.sum()*self.dh #energy needed for the appliance
        #     self.y_s=0 # means that it has never connected the machine
        
        
        
        #Create histories
        # self.state_hist=pd.DataFrame(index=tuple([(a,t) for a in self.agents_id+['global'] for t in range(self.tstep,self.tstep+self.Tw)]), columns=list(self.state_vars.keys())+['action','reward'])
        
        #initialy history is just the all history
        self.state_hist=self.state.copy()


        
        return self.get_env_obs()
        
        

        
    def step(self, action):
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
        
        

        
        #action will come as a dcitionary {aid:action,.....}
        self.action=pd.DataFrame.from_dict(action,orient='index',columns=['action'])
        
        #reward as a dictionary for alll agents
        self.reward=self.get_env_reward()
        
        
       
        
        #Register action and reward histories
        if self.tstep==self.tstep_init:
            self.action_hist=self.action.copy()
            self.reward_hist=pd.DataFrame.from_dict(self.reward, orient='index',columns=['reward'])
            
        else:
            self.action_hist=pd.concat([self.action_hist,self.action])
            self.reward_hist=pd.concat([self.reward_hist, pd.DataFrame.from_dict(self.reward, orient='index',columns=['reward'])])
            
        
        #check weather to end or not the episode
        self.check_term() 
        
        #saving history state in state_hist
        # self.state_hist.update(self.state.set_index([self.state.index,'tstep']))
        # self.state_hist=pd.concat([self.state_hist,self.state])
       
        
        self.tstep+=1 # update timestep
        # print(self.tstep)
        #update state variables  
        self.state_update() 
        #update all masks
        self.update_all_masks() 
        
        # print(self.state['E_prof_rem'])
        # self.R+=self.reward
        self.R={aid:self.R[aid]+self.reward[aid] for aid in self.agents_id}
        
        # self.r=self.reward
        
        
        
        
        self.state_hist=pd.concat([self.state_hist,self.state])
        
        
        
        ###########   Things I dont know exactly what they make or if are important ####
        #accumulated total cost
        # self.c_T+=self.cost_s
        
        
        # info={aid:'learning ongoing' for aid in self.agents_id}
        # info={info[k] for k in info}
        
        # obs=self.get_env_obs()
        
        # self.assert_type(obs)
        
        
        
        
        return self.get_env_obs(), self.reward, self.get_env_done(), {}
        
        
    
        
    def get_agent_reward(self, agent):
        "Computes the reward for each agent as a float"
        
        if self.reward_type == 'excess_cost_max':
            # The reward should be function of the action
            if self.minutes == self.min_max-self.agents_params.loc[agent]['T_prof']*self.tstep_size and self.state.loc[agent]['y_s']  !=self.agents_params.loc[agent]['T_prof']:
                agent_reward=-5.0
#This agent specific reward must variable according to the situation (machines invlved and time horizon)
    
            else:
                
                if self.mas_setup == 'cooperative_colective':
                    agent_reward=0
                    
                else:
                
                    agent_reward=-max(0,((self.action.loc[agent]['action']*self.profile[agent][0])-self.state.loc[agent]['excess0']))*self.state.loc[agent]['tar_buy']
                                
            return agent_reward
        
    
    # def reward_shapping(self,agent):
    #     return self.action.loc[agent]['action']*10*self.state.loc[agent]['excess0']
    
    
    
    def get_env_reward(self):
        '''
        Returns the reward for each agent in the environment as a dictionary for algorithm processing. 
        
        
        inspect self.mas_setup for actual setup
        
        - Cooperative: All agents get the same reward given by the sum of all agents rewards "
        
        - Competitive: each agent has an individual reward
        
        - Cooperative_colective: all agents get the same reward given by the collective purchase of energy from the grid (all loads are summed and the excess is collective)
        '''
        
        if self.mas_setup == 'cooperative':
        #cooperative // common reward
            R=sum([self.get_agent_reward(aid) for aid in self.agents_id])
            return {aid: R for aid in self.agents_id}
        
        #Competitive / individual rewards
        elif self.mas_setup == 'competitive':
            return {aid:self.get_agent_reward(aid) for aid in self.agents_id}
    
        elif self.mas_setup == 'cooperative_colective':
            AgentLoads=[self.action.loc[agent]['action']*self.profile[agent][0] for agent in self.agents_id]
            # R=sum(agents loads)- Excess
            R=-max(0,(sum(AgentLoads)-self.state.loc[self.agents_id[0]]['excess0']))*self.state.loc[self.agents_id[0]]['tar_buy']
            return {aid: R+self.get_agent_reward(aid) for aid in self.agents_id}
            
    
    def get_agent_obs(self, agent):
        assert agent in self.agents_id, 'Agent does not exist in this community'
        # return np.array(self.state.loc[agent], dtype=np.float32)
        return np.array(self.state.loc[agent], dtype=np.float32)
    
    
    def get_env_obs(self):
        "Returns the current state of the environment (observations, mask) in a    dictionary"
        # obs={aid:{'action_mask':np.array(self.mask.loc[aid], dtype=np.float32), 
        #           'observation': self.get_agent_obs(aid)} 
        obs={aid:{'action_mask':np.array(self.mask.loc[aid]), 
                  'observations': self.get_agent_obs(aid)} 
              for aid in self.agents_id}
        
        
        #debugging
        # obs={aid:{'observations': self.get_agent_obs(aid)} for aid in self.agents_id}
        
        
        
        return obs
    

    def get_env_done(self):
        "Returns the current done for all agents in a dictionary (observation, mask)"
        done_dict={aid:self.done.loc[aid]['done'] for aid in self.agents_id}
        done_dict['__all__']=all(self.done.values) #RLlib needs this
        
        return done_dict
                

    def get_init_tstep(self):
        '''
        A function that returns the initial tstep of the episode"
        
        - mode_window: we allways start at the beggining of the day and advance Tw timesteps but choose randomly what day we start
        
        - mode_window_seq; we allways start at the beggining of the day and advance Tw timesteps and days are chosen sequentially

        - mode_random: radom initiatiation any timestep
        
        - mode_horizon: always start at t=0 and advance Tw

        
        '''
        

        if self.init_cond == 'mode_window':
            
            # t=rnd.randrange(0, self.T-self.Tw-1) # a random initial state in the whole year
            # t=rnd.choice([k*self.Tw for k in range(int((self.T/self.Tw)-1))])
            
            t=rnd.choice(self.allowed_inits)
            
            # we allways start at the beggining of the day and advance Tw timesteps but choose randomly what day we start
            assert self.data.loc['ag1',t]['minutes']==0, 'initial timeslot not 0'
            return t
        
        elif self.init_cond == 'mode_window_seq': #sequential
            t=self.allowed_inits[0] #get first day
            self.allowed_inits.remove(t)
            return t
        
        elif self.init_cond=='mode_window_no-repeat':
            t=rnd.choice(self.allowed_inits) # day is is chosen randomly
            self.allowed_inits.remove(t) #but it doenst repeat we remove the vallue from the list
            return t
        
        
        elif self.init_cond == 'mode_horizon': 
            #episode starts at t=0
            return 0

        
        
    def update_forecast(self):
        "What is happening: the 'number' in the variable name (example: gen2, gen3 ) is used as the time (t) that is incremented in the current timestep (self.tstep). We can multiply t*4 to make it span 24 hours with 2h intervals"
        
        for agent in self.agents_id:
        
            for var in self.var_class:
                var_keys=[key for key in self.state_vars.keys() if var in key]
                
                for k in var_keys:
                    t=re.findall(r"\d+", k)
                    # print(var)
                    # print(t)
                    t=int(t[0])
                    
                    if self.tstep+self.t_ahead*t >= self.T: #if forecast values fall outside global horizon T
                        # setattr(self,k, 0)
                        self.state.loc[agent,k]=0
                        # self.state.loc[agent,k]=0 # maybe this to solve the 'value is trying to be set on a copy of a slice from a DataFrame' warning
                    else:
                        # setattr(self,k, self.data.iloc[self.tstep+self.t_ahead*t][var] )
                        
                        tstep_to_use=self.tstep+self.t_ahead*t
                        self.state.loc[agent,k]=self.data.loc[agent,tstep_to_use][var]
                



    def state_update(self):
        
        #Variables update
        self.state['tstep']=float(self.tstep)

        #Tariffs
        self.tar_buy,self.tar_sell=self.get_tariffs(0) #tariff for the present timestep
        self.state['tar_buy']=self.tar_buy
        self.state['tar_buy0'], _ =self.get_tariffs(1)
    
        #update forecasts
        self.update_forecast()
        
        #Minutes
        self.minutes=self.data.iloc[self.tstep]['minutes']
        self.state['minutes']=self.minutes
        #sine/cosine
        self.sin=np.sin(2*np.pi*(self.minutes/self.min_max))
        self.cos=np.cos(2*np.pi*(self.minutes/self.min_max))
        self.state['sin']=self.sin
        self.state['cos']=self.cos
        
        # Binary variables
        
        #y update
        self.state['y'].update(self.action['action']) #matches by index
        #y_1 and y_s update
        for aid in self.agents_id:
            self.state.loc[aid,'y_s']+=self.action.loc[aid]['action'] #update y_s
            
            if self.tstep > self.tstep_init+1:
                # self.y_1=self.hist[-2] #penultiumate element to get the previous action
                # self.state.loc[aid,'y_1']=self.state_hist.loc[aid,self.tstep-1]['y']
                # self.state.copy().loc[aid]['y_1']=self.state_hist.loc[aid][self.state_hist.tstep[aid]==self.tstep-1]['y']
                
                self.state.at[aid,'y_1']=self.state_hist.loc[aid][self.state_hist.tstep[aid]==self.tstep-1]['y']    
            else:
                self.state.loc[aid,'y_1']=0


        # E_pro_rem and y_s
        if self.minutes==0: #It restarts in the beggining of the day 
            self.state['E_prof_rem'].update(self.agents_params['E_prof'])
            self.state['y_s'].update(0)
            

        #update remaining energy that needs to be consumed
        for aid in self.agents_id:
            self.state.loc[aid,'E_prof_rem']-=self.action.loc[aid]['action']*self.profile[aid][0]

        # self.E_prof-=self.action*self.profile[0]


        
        # #convert action to power (constant profile)
        # self.load_s=self.action*self.profile[0]
        
        
        
        # if self.minutes ==0: #the number of connected timeslots resets when a new day starts
        #     self.t_shift=0
        #     self.y_s=0
            
            
        
        
        
        # #deficit vs excess
        # self.delta_c = (self.load0+self.load_s)-self.gen0 # if positive there is imports from the grid. If negative there are exports to the grid 
        
        # self.delta_s=self.load_s-self.gen0
        

        # #energy cost
        
        # # cost considering the full load
        # self.cost=max(0,self.delta_c)*self.tar_buy + min(0,self.delta_c)*self.tar_sell
        # #cost considering only the appliance
        # self.cost_s=max(0,self.delta_s)*self.tar_buy + min(0,self.delta_s)*self.tar_sell
        
        
        # self.excess=max(0,-self.delta0)   
        # self.cost_s_x=max(0,self.load_s-self.excess0)*self.tar_buy + min(0,self.load_s-self.excess0)*self.tar_sell


    def update_all_masks(self):
        "loops over agents and updates individual masks. Returns all masks"
        
        for aid in self.agents_id:

            if self.action.loc[aid]['action']==1 and self.state.loc[aid]['y_s'] < self.agents_params.loc[aid]['T_prof']:
                self.mask.loc[aid]=[0.0,1.0]
    
            elif self.state.loc[aid]['y_s'] >= self.agents_params.loc[aid]['T_prof']:
                self.mask.loc[aid]=[1.0,0.0]
                
            else:
                self.mask.loc[aid] = np.ones(self.action_space.n)
            
        
        return self.mask
        
        

    def get_tariffs(self, tsteps_ahead):
        "get tarrifs in €/kWh for argument tstep_ahead (integer number of timesteps) ahead of self.minutes" 
        
        
        if  self.tar_type=='bi':
            
            hour_start=8
            hour_end=22
            
            if self.minutes + tsteps_ahead*self.tstep_size >= self.tstep_size*(60/self.tstep_size)*hour_start and self.minutes <=self.tstep_size*(60/self.tstep_size)*hour_end:
                tar_buy=0.1393
            else:
                tar_buy=0.0615
            # self.tar_buy=0.17*(1-(self.gen/1.764)) #PV indexed tariff 
                
            tar_sell=0.0 # remuneration for excess production
        
        elif self.tar_type=='flat':
            tar_buy=0.10
            tar_sell=0.0
        
        return tar_buy, tar_sell
        
        
    def check_term(self):
        if self.tstep==self.get_term_cond():

            self.R_Total.append(self.R)
            # print('sparse', self.R)
            print(self.R)
            # print(self.mask)

            self.n_episodes+=1
            
            self.done.loc[self.agents_id] = True #update done for all agents

            
            # return self.get_env_obs(), self.reward, self.get_env_done(), {'episode has ended'}
        
        else:
            self.done.loc[self.agents_id] = False
    
    def get_term_cond(self):
        "A function to get the correct episode termination condition according to weather it is in window or horizon mode defined in the self.done_cond variable"
        
        if self.done_cond == 'mode_window': # episode ends when when Tw timeslots passed
            return self.tstep_init+self.Tw-1
        elif self.done_cond == 'mode_horizon': #episode ends in the end of the data length
            return self.T
        
    def get_state(self):
        return self.state
         
    
    def assert_type(self,obs):
        for key in obs.keys():
            for key2 in obs[key]:
                for k in range(len(obs[key][key2])):
                    val=obs[key][key2][k]
                    # print('tipos')
                    # print('ta-se bem')
                    assert  math.isnan(val) != True, 'there is a nan'
                    assert type(val)==np.float32, 'other type'
                    # print(val)
                    # print(type(val))
     
        
    #functions needed for eliminating warnings 
    def action_space_sample(self,keys):
        return {aid:rnd.randint(0,1) for aid,a in zip(self.agents_id,[1,0,0,1])}
    
    def observation_space_sample(self):
        return self.reset()
    
    def action_space_contains(self, x):
        return True
    
    def observation_space_contains(self, x):
        return True        