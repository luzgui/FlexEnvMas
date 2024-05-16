#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 10:42:29 2024

@author: omega
"""
import os
from os import path
from pathlib import Path
import sys
import time
import datetime
from datetime import datetime
from termcolor import colored

from dataprocessor import DataPostProcessor, YAMLParser

import pandas as pd
import numpy as np

from plots import Plots

class TestEnv():
    def __init__(self,env, tester, file_experiment,test_config_file):
        self.env=env
        self.tester=tester
        self.processor=DataPostProcessor(env)
        self.plot=Plots()
        self.experiment_config=YAMLParser().load_yaml(file_experiment)
        self.test_config=YAMLParser().load_yaml(test_config_file)
        self.exp_name=self.experiment_config['exp_name']
        self.test_name= self.test_config['test_name']
        self.folder_name='Train_'+self.exp_name + '_'+'Test_'+self.test_name 
        
        
    def test(self,n_episodes,results_path, plot=True):
        """save_metrics=True will make Results from testing (plots and metrics) to be saved in the trainable folder (results_path) alongside the checkpoints"""

        episode_metrics=pd.DataFrame()
        env_state_conc=pd.DataFrame()
        
        
        #choose the policy mapping function according to policies in algorithm config
        
        pols=list(self.tester.config.policies.keys())

        if 'pol_ag' in pols[0]: # each agent has its policy
            policy_mapping_func=self.policy_mapping_fn
        elif pols[0]=='shared_pol': # theres a shared policy for agents
            policy_mapping_func=self.policy_mapping_fn_shared
        else:
            policy_mapping_func=self.policy_mapping_fn
            
        k=0
        
        while k < n_episodes:
        
            mask_track=[]
        
            T=self.env.Tw*1

            print('testing....')
            for i in range(T):
                if i==0:
                    obs = self.env.reset()
                
                actions=self.get_actions(obs,policy_mapping_func)
                obs, reward, done, info = self.env.step(actions)

                
            # we are summing the total cost and making a mean for delta    


            full_state, env_state=self.processor.get_post_data()
            
            episode_metrics=pd.concat([self.processor.get_episode_metrics(full_state,
                                                           env_state,k),
                                                           episode_metrics])
            
                    
            env_state_conc=pd.concat([env_state_conc, env_state.reset_index()],
                                     axis=0).reset_index(drop=True)
            
                    
            if plot: 
                self.plot.makeplot_bar(env_state, None)
                
                
                
            k+=1
            
        #save results in a fodler inside the trainable folder
        if results_path:
            test_results_path=os.path.join(results_path,self.folder_name)
            if not os.path.exists(test_results_path) and not os.path.isdir(test_results_path):
                os.makedirs(test_results_path)
                print(colored('folder created' ,'red'),test_results_path)
            
            
            filename_metrics='metrics_'+str(k)+'_eps'+'.csv'
            filename_metrics=os.path.join(test_results_path,filename_metrics)
            episode_metrics.to_csv(filename_metrics)
            print(colored('Metrics saved to' ,'red'),filename_metrics)
            
            filename_env_state='env_state_'+str(k)+'_eps'+'.csv'
            filename_env_state=os.path.join(test_results_path,filename_env_state)
            env_state_conc.to_csv(filename_env_state)
            print(colored('env_state saved to' ,'red'),filename_env_state)
            
            filename=[filename_metrics,filename_env_state]
            
        else:
            filename_metrics=''
            filename_env_state=''
            filename=[filename_metrics,filename_env_state]
            
            
        return full_state, env_state_conc, episode_metrics, filename
    


    
    def policy_mapping_fn(self, agent_id):
        'Policy mapping function'
        return 'pol_' + agent_id
    
    def policy_mapping_fn_shared(self, agent_id):
        'Policy mapping function with shared policy'
        return 'shared_pol' # p
    
    def get_actions(self, obs, map_func):
        'resturns the actions of the agents'
        if type(obs)==dict:
            actions = {aid:self.tester.compute_single_action(obs[aid],
                                                         policy_id=map_func(aid)) for aid in self.env.agents_id}
        elif type(obs)==tuple:
            actions = {aid:self.tester.compute_single_action(obs[0][aid],
                                                         policy_id=map_func(aid)) for aid in self.env.agents_id}
        return actions
    

        
         
         
class SimpleTestEnv(TestEnv):
    def __init__(self,env, tester):
        self.env=env
        self.tester=tester
        self.processor=DataPostProcessor(env)
        self.plot=Plots()
        self.counter=0
        # super().__init__(env, tester)
     
        

        
    def get_actions(self, obs, map_func):
        "Overrides the superclass methods"
        action_plan=self.get_action_plan()        
        actions = {aid: action_plan[aid][self.counter] for aid in self.env.agents_id}
        self.counter+=1
        return actions
    
        
    def get_action_plan(self):
        actions={}
        starts=dict(zip(self.env.agents_id, [47,48,49,32,36,45,50][0:len(self.env.agents_id)]))
        
        for ag in self.env.agents_id:
            agent=self.env.com.get_agent_obj(ag)
            D=agent.apps[0].duration/self.env.tstep_size
            actions[ag]=self.create_binary_vector(self.env.Tw,D,starts[ag])
        
        return actions
          
    def create_binary_vector(self, T, D, t):
        """
        Create a binary vector with zeros everywhere except for a specific duration D starting at time t.

        """
        binary_vector = np.zeros(T, dtype=int)
        # import pdb
        # pdb.pdb.set_trace()
        binary_vector[t:int(t+D)] = 1
        return binary_vector
            
        
    def transition_test(self, var_out):
        obs0=self.env.reset()
        action=self.env.action_space_sample(keys=None)
        obs1=self.env.step(action)
        
        if var_out=='state_hist':
            return self.env.state_hist
        elif var_out=='obs':
            print('action', action)
            return obs1
        
    def episode_test(self):
        
        obs=self.env.reset()
        action_plan=self.get_action_plan()
        actions={}
        
        for i in range(self.env.Tw):
            actions = {aid: action_plan[aid][i] for aid in self.env.agents_id}  
            print('iteration', i)
            obs, reward, done, info = self.env.step(actions)
        
        return self.env
    
    def test_full_state(self,df):
        for aid in self.env.agents_id:
            # c=df[]
            print('rewardsxx:' f'"cens {aid}"')


class DummyTester:
    def __init__(self,env):
        self.config=DummyConfig(env)
        # self.config.policies={aid: None for aid in env.agents_id}
        
class DummyConfig:
    def __init__(self,env):
        self.policies={aid: None for aid in env.agents_id} 
                
    
        
        