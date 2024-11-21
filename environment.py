#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 11:45:44 2024

@author: omega
"""

import gymnasium  as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import random as rnd
import re
import math
from termcolor import colored
from icecream import ic


from ray.rllib.env.multi_agent_env import MultiAgentEnv, make_multi_agent

# from state_vars import *
from termcolor import colored

from dataprocessor import DataProcessor, EnvDataProcessor
from utilities import *

from reward import Reward
from state_update import StateUpdate 



class FlexEnv(MultiAgentEnv):
    """
    Multiagent Environment Collective self-consumption maximization with shiftable loads
    - Each agent only has one shiftable appliance
    - Shiftable appliances may be differnet in parameters
    - There can be any number of agents
    
    - The time reference of the environment (self.tstep) is the timeslot position IN THE environment dataset self.T self.tstep in ['t_init', 'tend_']
    
    """
    def __init__(self,env_config):
        
        super().__init__()
        
        self.env_config=env_config
        
        self.processor=DataProcessor()
        
        
        self.com=self.env_config['community']
        self.com_vars=self.env_config['com_vars']
              
        self.reward_obj=Reward(self)
        #COMMUNITY /COMMON parameters
        self.agents_id=list(self.com.agents.keys())
        # self._agent_ids=self.get_agent_ids()
        self._agent_ids=self.agents_id
        
        self.info = self.com.problem_conf['env_info']
        self.Tw=self.com.problem_conf["window_size"] #window horizon
        
        
        self.data=self.com.com_data
        self.env_processor=EnvDataProcessor(self.data,self.Tw) #environment data processor
        
        self.stats=self.env_processor.get_data_stats() #define this in an environment processor
        self.daily_stats=self.env_processor.get_daily_stats()
          
        self.tstep_size=self.com.problem_conf['step_size']
        
        # Time horizon of the dataset
        self.T=self.com.problem_conf['t_end']-self.com.problem_conf['t_init']
        
        self.tstep_per_day=self.com.problem_conf['tstep_per_day']
        # self.dh=self.tstep_size*(1/60.0) # Conversion factor power-energy
        self.dh=self.com.problem_conf['step_size']*(1/60.0)
        self.tstep_init=self.com.problem_conf['t_init'] #initial timestep of the episode
        self.tstep_end=self.com.problem_conf['t_end'] #final timestep of the dataset
        
        # FORECAST
        self.t_ahead_hours = self.com.problem_conf["t_ahead_hours"] #number of hours to look ahead
        self.t_ahead=self.t_ahead_hours*(60/self.tstep_size) #number of timeslots that each actual timeslot is loloking ahead (used in update_forecast) 
        #
        
        self.min_max=self.data['minutes'].max() #maximum timeslot??
        
        #get the possoble initial timeslots from data        
        #get all o mionutes index
        # self.allowed_inits=self.data[self.data['minutes']==0].index.get_level_values(1).unique().tolist()
        # self.allowed_inits=self.env_processor.get_allowed_inits(self.com.problem_conf['allowed_init_config'],
        #                                                         self.agents_params)
        
        
        #AGENTS INDIVIDUAL parameters
        #Appliance profile    
        self.agents_params=self.com.com_prefs
        
        self.allowed_inits=self.env_processor.get_selected_allowed_inits(self.com.problem_conf['allowed_init_config'],
                                                                self.agents_params)
        
        self.allowed_inits.pop() #remove last day due to the existence of an extra timestep at the end of the episode that produces and error for T+1
        
        
        
        
        

        self.R_Total=[] # A way to see the evolution of the rewards as the model is being trained
        self.n_episodes=0
        
        
        #new vars
        # self.L_s=np.zeros(self.T) # we make a vector of zeros to store the shiftable load profile
        # self.l_s=self.L_s[0] 
        
        #initialize with the first element in 
        # self.t_shift=0
        
        
        ##### defining the state variables #####
        #get limits on state variables
        # self.max_gen=100.0
        # self.max_load=100.0
        self.max_load=round(self.processor.get_limits(self.data, 'max', 'load'),2)
        self.max_gen=round(self.processor.get_limits(self.data, 'max', 'gen'),2)
        
        # #update missing variables HERE
        # self.com_vars.update_var_list(['gen','load', 'excess'], 'max', self.max_gen)
        # self.com_vars.update_var_list(['gen','load', 'excess'], 'min', 0)
        # self.com_vars.update_var('E_prof_rem', 'max',self.agents_params['E_prof'].max())
        # self.com_vars.update_var('y_s', 'max',self.agents_params['T_prof'].max() )
        
        #unormalized state variables
        self.state_vars_unormal, _ = self.com_vars.get_state_vars(normalize=False)  
        
        #get the state variables based on normalization config
        self.state_vars, self.vars_list=self.com_vars.get_state_vars(self.com.problem_conf['normalization'])  
        
        self.var_class={'gen','load','delta','excess'} #this list of variables will be tested in update_forecasts()
        
        #Number of variables to be used
        self.var_dim=len(self.state_vars.keys())
        
        #creategh statte update object
        self.state_upd=StateUpdate(self)
        
        
        #extract the names of variables in env.data and take out the minutes that we dont need    
        # self.ag_var_class=[k for k in self.data.keys() if 'ag' in k]

        self.obs=None   
            
        
                
        #Action space        
        self.action_space = spaces.Discrete(2) # ON/OFF
    
        # Observation space   
        highlim=np.array([value['max'] for key, value in self.state_vars.items()])
        lowlim=np.array([value['min'] for key, value in self.state_vars.items()])
        
        self.observation_space = spaces.Dict({
            "action_mask": spaces.Box(-10.0, 10.0, shape=(self.action_space.n,),dtype=np.float32),
            "observations": spaces.Box(low=np.float32(lowlim), high=np.float32(highlim), shape=(self.var_dim,), dtype=np.float32)})
        
                
        #Training/testing termination condition
        self.done_cond=self.com.problem_conf['done_condition']
        self.init_cond=self.com.problem_conf['init_condition'] 
        
        
        #seed 
        # self.seed=config['seed']
        # self.seed=np.random.seed(config['seed'])
        
        
        
        
        #VARIABLE INITIATIATIONS
        #initiate N X var_dim dataframe for storing state
        self.state=pd.DataFrame(index = self.agents_id+['global'],columns=self.state_vars.keys())
        #tracks some variables that are not used in observation as an input to policy but are needed for post processing
        self.state_shadow=pd.DataFrame(index = self.agents_id+['global'],columns=['tstep'])
                
        #Mask
        #we have a dataframe cell for each action mask
        self.mask=pd.DataFrame(index = self.agents_id, columns=np.arange(self.action_space.n))
        
        
        # self.state_aux=pd.DataFrame(index = self.agents_id)
        
        
        
        print(f'Created an multiagent environment with {self.com.num_agents} agents')
        print(colored('Envrironment configurations:','green'))
        print('Reward Function:', colored(self.com.scenarios_conf['reward_func'],'red'))
        # print('Multiagent setup for reward:', colored(self.com.scenarios_conf['game_setup'],'red'))
        print('Environment purpose:', colored(self.com.problem_conf['env_info'],'red'))
        
    
        self._spaces_in_preferred_format = True
        

    def reset(self, *, seed=None, options=None):
        """
        Reset the environment state and returns an initial observation
        
        Returns:
        -------
        observation (object): The initial observation for the new episode after reset
        :return:
        """
        
        

        
        #COMMUNITY /COMMON variables (intialization equal for all agents)
        self.done=pd.DataFrame([False]*self.com.num_agents, index=self.agents_id,columns=['done'])
        # self.done.loc['__all__']=all(self.done.values) #does RLLIB need this in the environment or only at the output dictionary??
        
        self.R={aid:0 for aid in self.agents_id}
        
        #initial timestep
        self.tstep=self.get_init_tstep()
        self.minutes=self.data.loc['ag1',self.tstep]['minutes']
        
        # self.tstep=35040
        self.tstep_init=self.tstep # initial timestep
        # print(colored('Initial timestep','red'),self.tstep)
        self.state_shadow['tstep']=self.tstep
        #minutes
        
        #update the state
        self.state_upd.update_features()
        
        
        # self.state['minutes']=self.minutes
        #sine/cosine
        # self.state['sin']=np.sin(2*np.pi*(self.minutes/self.min_max))
        # self.state['cos']=np.cos(2*np.pi*(self.minutes/self.min_max))
        
        #tariffs (for now all agents get the same tariff)
        # self.tar_buy,self.tar_sell=self.get_tariffs(0) #tariff for the present timestep
        # utilities.print_info('All agents are getting the same tariff for now')
        # self.state['tar_buy']=self.com.get_tariffs_by_mins(self.tstep)
        # self.state['tar_buy0']=self.com.get_tariffs_by_mins(self.tstep+1)
        
        #inititialize binary variables
        # self.state[['y','y_1','y_s']]=0
        # self.state[['y_s']]=0
        
        
        # Initialize history of actions
        # self.hist=pd.DataFrame(index = range(self.Tw),columns=self.agents_id)
        # self.hist=pd.DataFrame(index=tuple([(a,t) for a in self.agents_id for t in range(self.tstep,self.tstep+self.Tw)]), columns=['hist'])
        
        
        
        # self.hist.append(self.y) # CHECK IF NEEDED
        
        #Initial energy to consume
        # self.state['E_prof_rem'].update(self.agents_params['E_prof'])
        # self.state.update({'E_prof_rem': self.agents_params['E_prof']})

        
        #update forecasts
        # self.update_forecast()
        
        # for aid in self.agents_id:
        #     self.state.loc[aid,'pv_sum']=self.data.loc['ag1'][self.tstep:self.tstep_init+self.Tw]['gen'].sum()
            # self.state.loc[aid, self.state.columns.str.contains(r'^tar\d+$')]=self.get_future_values(8).loc[aid]['tar_buy'].values
            
            # s=self.get_episode_data().loc[aid]['tar_buy']
            # self.state.loc[aid,'tar_d']=self.state.loc[aid,'tar_buy']-min(s.loc[self.tstep:self.tstep_init+self.Tw-1])
        
        # self.update_tariffs()
        #Initial mask: # we concede full freedom for the appliance 
        self.mask.loc[:,:]=np.ones([self.com.num_agents,self.action_space.n])
        
        
        
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
        self.state_hist=pd.concat([self.state_hist,self.state_shadow], axis=1)


        self.make_state_norm() #make the self.state_norm attribute for the normalized state
        
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
         
        
        self.tstep+=1 # update timestep
        #check weather to end or not the episode
        Done=self.check_term() 
        #saving history state in state_hist
        # self.state_hist.update(self.state.set_index([self.state.index,'tstep']))
        # self.state_hist=pd.concat([self.state_hist,self.state])
       
        # print(self.tstep)
        #update state variables  
        self.state_update()
        
        #uncomment to check variables limits in each timestep
        self.check_obs_within_lims()
        
        #update all masks
        self.update_all_masks() 
        self.R={aid:self.R[aid]+self.reward[aid] for aid in self.agents_id}

        
        self.state_hist=pd.concat([self.state_hist,pd.concat([self.state,self.state_shadow],axis=1)])

        
        
        self.make_state_norm() #make the normalized state
        
        # self.check_term() #do we need to terminate here?
        # return self.get_env_obs(), self.reward, self.get_env_done(), {}
        return self.get_env_obs(), self.reward, self.env_done, {}
        
        
    
    
    
    def get_env_reward(self):
        return self.reward_obj.reward_func()
            
            
            
    
    def get_agent_obs(self, agent):
        assert agent in self.agents_id, 'Agent does not exist in this community'
        
        if self.com.problem_conf['normalization'] == True: #return the nomrlized state
            return np.array(self.state_norm.loc[agent], dtype=np.float32)
        else:
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
        # ic(done_dict)
        
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
            print('chosen timestep:',t)
            print('days left:',len(self.allowed_inits))
            return t
        
        elif self.init_cond=='mode_window_no-repeat':
            t=rnd.choice(self.allowed_inits) # day is is chosen randomly
            self.allowed_inits.remove(t) #but it doenst repeat we remove the vallue from the list
            print('chosen timestep:',t)
            print('days left:',len(self.allowed_inits))
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
                    t=int(t[0])
                    
                    # if self.tstep+self.t_ahead*t >= self.tstep_init+self.T: #if forecast values fall outside global horizon T
                    # if self.tstep+self.t_ahead*t >= self.T-1: #if forecast values fall outside global horizon T
                    if self.tstep+self.t_ahead*t >= self.tstep_end-1: #if forecast values fall outside global horizon T
                        self.state.loc[agent,k]=0
                        # self.state.loc[agent,k]=0 # maybe this to solve the 'value is trying to be set on a copy of a slice from a DataFrame' warning
                    else:
                        tstep_to_use=self.tstep+self.t_ahead*t
                        self.state.loc[agent,k]=self.data.loc[agent,tstep_to_use][var]
                

    def update_tariffs(self):
        t_ahead=8
        for aid in self.agents_id:
            # import pdb
            # pdb.pdb.set_trace()
            
            df_out=self.get_future_values(t_ahead)
            
            if df_out.empty:
                vals=self.com.agents[aid].tar_max*np.ones(t_ahead)
            else: 
                vals=df_out.loc[aid]['tar_buy'].values

            vals=list(vals)
            while len(vals) < t_ahead:
                vals.append(self.com.agents[aid].tar_max)  # Append 1 to the end of the list

            self.state.loc[aid, self.state.columns.str.contains(r'^tar\d+$')]=vals
            
            s=self.get_episode_data().loc[aid]['tar_buy']
            ss=s.loc[self.tstep:self.tstep_init+self.Tw-1]
            if ss.empty:
                self.state.loc[aid,'tar_d']=self.state.loc[aid,'tar_buy']
            else:
                self.state.loc[aid,'tar_d']=self.state.loc[aid,'tar_buy']-min(s.loc[self.tstep:self.tstep_init+self.Tw-1])    
            

    def state_update(self):
        
        #Variables update
        self.state_shadow=pd.DataFrame({'tstep': [self.tstep]*(len(self.agents_id)+1)},index=self.state.index)
        # self.state_shadow=pd.concat([self.state_shadow,new_df])
        
        self.minutes=self.data.loc['ag1', self.tstep]['minutes']

        #update the state
        self.state_upd.update_features()
        


        


    def update_all_masks(self):
        "loops over agents and updates individual masks. Returns all masks"
        
        for aid in self.agents_id:

            if self.action.loc[aid]['action']==1 and self.state.loc[aid]['y_s'] < self.agents_params.loc[aid]['T_prof']:
            # if self.action.loc[aid]['action']==1 and self.state.loc[aid]['E_prof_rem'] > 0:
                self.mask.loc[aid]=[0.0,1.0]
    
            elif self.state.loc[aid]['y_s'] >= self.agents_params.loc[aid]['T_prof']:
            # if self.state.loc[aid]['E_prof_rem'] <= 0:
                self.mask.loc[aid]=[1.0,0.0]
                
            else:
                self.mask.loc[aid] = np.ones(self.action_space.n)
            
        
        return self.mask
        
        

    def check_term(self):
        if self.tstep>=self.get_term_cond():
            
            self.R_Total.append(self.R)
            # print('ts_init:',self.tstep_init)
            print(f"ts_init: {self.tstep_init} | pv_sum: {np.round(self.state_hist.loc['ag1','pv_sum'].iloc[0],2)}")
            print('rewards:',self.R)
            print('w1',np.round(self.reward_obj.w1,2),'|','w2',np.round(self.reward_obj.w2,2))
            ts_start = {aid: self.get_start_ts(self.action_hist.loc[aid].values) for aid in self.agents_id}
            print('actions:',ts_start,'|','action len', len(self.action_hist.loc['ag1']))
            self.n_episodes+=1
            self.done.loc[self.agents_id] = True #update done for all agents
            self.env_done=self.get_env_done()
            self.check_hist_within_lims()

            # import pdb
            # pdb.pdb.set_trace()
            return True
        
        else:
            
            self.done.loc[self.agents_id] = False
            self.env_done=self.get_env_done()
            
            return False
    
    def get_term_cond(self):
        "A function to get the correct episode termination condition according to weather it is in window or horizon mode defined in the self.done_cond variable"
        
        if self.done_cond == 'mode_window': # episode ends when when Tw timeslots passed
            return self.tstep_init+self.Tw
        elif self.done_cond == 'mode_horizon': #episode ends in the end of the data length
            return self.T
        
         
    
    def make_state_norm(self):
        '''
        Creates the attribute self.state_norm
        
        This normalization uses statistics from the dataset 
        to normalize the state before entering the NN'''
        self.state_norm=self.state.copy()
        # self.state=self.state.astype('object')
        self.state_norm=self.state_norm.astype('object') #casting as differente datatype solves the new pandas warning 
        for aid in self.agents_id:
            for key in self.state.columns:
                # print(key)
                if key=='tstep': #convert timestep to sin
                    self.state_norm.loc[aid,key]=np.sin(2*np.pi*(self.state.loc[aid,key])/self.T)
                
                if key=='minutes': #convert minutes to cos with a phase
                    self.state_norm.loc[aid,key]=np.cos(2*np.pi*(self.state.loc[aid,key]/self.T)+np.pi)
                
                if key=='y_s': #normalize by the app profile timeslots
                    self.state_norm.loc[aid,key]=self.state.loc[aid,key]/self.agents_params.loc[aid]['T_prof']
                
                if key=='E_prof_rem':
                    self.state_norm.loc[aid,key]=self.state.loc[aid,key]/self.agents_params.loc[aid]['E_prof']
                    
                
                if key == 'pv_sum':
                    pv_sum_day=self.state_hist[key][aid].max()
                    self.state_norm.loc[aid,key]=self.state.loc[aid,key]/pv_sum_day

                # Normalize all tariffs by the maximum tariff
                tar_stats=self.get_episode_data().loc[aid]['tar_buy'].describe()
                
                # self.state_norm.loc[aid, self.state_norm.columns.str.contains('tar')]=(self.state.loc[aid, self.state.columns.str.contains('tar')]-tar_stats['mean'])/(tar_stats['max']-tar_stats['min'])
                
                self.state_norm.loc[aid, self.state_norm.columns.str.contains('tar')]=self.state.loc[aid, self.state.columns.str.contains('tar')]/self.com.agents[aid].tar_max
                
                if key == 'tar_mean':
                    self.state_norm.loc[aid,key]=self.state.loc[aid,key]
                    
                if key == 'tar_stdev':
                    self.state_norm.loc[aid,key]=self.state.loc[aid,key]
                        
                if key == 'tar_var':
                    self.state_norm.loc[aid,key]=self.state.loc[aid,key]
            


                for var in self.var_class:
                    if var in key:
                        
                        #mean normalization using statistics from the training dataset
                        self.state_norm.loc[aid,key]=(self.state.loc[aid,key]-self.stats[aid].loc['mean'][var])/(self.stats[aid].loc['max'][var]-self.stats[aid].loc['min'][var])
                               
                        #standartization
                        # self.state_norm.loc[aid,key]=(self.state.loc[aid,key]-self.stats[aid].loc['mean'][var])/self.stats[aid].loc['mean'][var]
                        
                        # maxnormalization
                        # self.state_norm.loc[aid,key]=self.state.loc[aid,key]/self.stats[aid].loc['max'][var]
        # print('time_step',self.tstep)        
        # print(self.state_norm)
                
                    
                
    
    
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
    
    def seed(self,seed):
        # return np.random.seed(self.env_config['seed'])
        return np.random.seed(seed)
    
    def get_agent_ids(self):
        return set(self.agents_id)
    
    def get_agent(self, agent_id):
        return self.com.agents[agent_id]
    
    def check_obs_within_lims(self):
        result_df=pd.DataFrame(columns=self.state.columns, index=self.agents_id)
        for ag in self.agents_id:
            for key in self.state_vars_unormal.keys():
                val=self.state.loc[ag][key]
                max_val=self.state_vars_unormal[key]['max']
                min_val=self.state_vars_unormal[key]['min']
                result_df.loc[ag,key]=bool((val >= min_val) & (val <= max_val))
                
                assert result_df.all().all(), f" '{key}' outside limits in agent '{ag}', min_val:'{min_val}' val:'{val}', max_val:'{max_val}'"
                
    def check_hist_within_lims(self):
        # result_df=pd.DataFrame(columns=self.state_hist.columns, index=self.agents_id)
        # print('Aqui estou Manuel Acacio')
        result_df=self.state_hist.copy()
        result_df = result_df.astype(float)
        result_df=result_df.loc[result_df.index.str.contains('ag')]
        result_df.loc[:, :] = np.nan
        
        for key in self.state_vars_unormal.keys():
            vals=self.state_hist.loc[self.state_hist.index.str.contains('ag'), [key]]
            max_val=self.state_vars_unormal[key]['max']
            min_val=self.state_vars_unormal[key]['min']
            result_df[key]=(vals >= min_val) & (vals <= max_val)    
        # assert result_df.all().all(), 'something is outside of limits'
        # import pdb
        # pdb.pdb.set_trace()
        try:
            assert result_df.all().all(), 'something is outside of limits'
        except AssertionError as e:
            print(e)
            print(result_df)
            false_indices = result_df.index[result_df.isin([False]).any(axis=1)]
            false_columns = result_df.columns[result_df.isin([False]).any(axis=0)]
            print("Rows with False values:")
            print(false_indices)
            print("Columns with False values:")
            print(false_columns)
        
        
        return result_df
    
    def get_episode_tstep_indexes(self):
        "returns a list of the tsteps in the present episode"
        return list(range(self.tstep_init, self.tstep_init+self.Tw))
    
    def get_episode_data(self):
        n=1
        "this function returns the data for this episode and the next n-1 episode"
        
        df=self.data.loc[(slice(None), slice(self.tstep_init, self.tstep_init+n*self.Tw-1)), :]
        df=df.copy()
        for aid in self.agents_id:
            df.loc[aid,'tar_buy']=list(self.com.agents[aid].tariff)
            
        return df
    
    
    def get_future_values(self, tstep_ahead):
        """Return the values in df for future tsteps starting in tstep and spaced tstep_ahead."""
        # Create a list of future indices
        df=self.get_episode_data()
        future_indices = [self.tstep + 4 * i for i in range(1, tstep_ahead + 1)]  # Start from t+4
        vals=list(df.index.get_level_values(1).unique())
        indices=list(set(future_indices) & set(vals))
        indices=sorted(indices) 

        idx = pd.IndexSlice
        df_out=df.loc[idx[:,indices,:]]
        
        return df_out
    
    
    def get_start_ts(self, action_list):
        
        if sum(action_list)==0:
            return 'no action'
        else:
            return np.argmax(action_list!=0)
        
                
    # def check_obs_within_lims(self):
    #     result_df = pd.DataFrame(index=self.agents_id)
        
    #     for key, limits in self.state_vars_unormal.items():
    #         min_val = [limits['min']]
    #         max_val = [limits['max']]
    #         vals= self.state.loc[self.state.index.str.contains('ag'), [key]]
            
    #         within_limits = (vals >= min_val) & (vals <= max_val)
    #         result_df[key] = within_limits.all(axis=1)
            
    #         # import pdb
    #         # pdb.pdb.set_trace()
            
            # if not within_limits.all(axis=None):
            #     invalid_agents = result_df.index[~within_limits.all(axis=1)]
            #     for ag in invalid_agents:
            #         val = vals.loc[ag]
            #         min_val = limits['min']
            #         max_val = limits['max']
            #         msg = f"'{key}' outside limits in agent '{ag}', min_val: {min_val}, val: {val}, max_val: {max_val}"
            #         assert False, msg
        
    #     return result_df
                
                    

    
    
    
    
    