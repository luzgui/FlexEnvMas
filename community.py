#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 14:57:51 2024

@author: omega
"""
import pandas as pd
from agents import Agent
from resources import *
from dataprocessor import YAMLParser, GecadDataProcessor, DataProcessor

class Community:
    def __init__(self, 
                 agents_config,
                 apps_config,
                 scenarios_config,
                 problem_config,
                 data_file):
        
        self.parser=YAMLParser()
        self.agents_conf=self.parser.load_yaml(agents_config)
        self.apps_conf=self.parser.load_yaml(apps_config)
        self.scenarios_conf=self.parser.load_yaml(scenarios_config)
        self.problem_conf=self.parser.load_yaml(problem_config)
        
        #Make agent DataProcessor
        self.agent_processor=GecadDataProcessor(problem_config,
                                                agents_config,
                                                data_file)
        #Make community DataProcessor
        self.com_processor=DataProcessor()
        
        #Assemble agents
        self.assemble_agents()
        self.num_agents=len(self.agents)
        
        self.print_info()
        
        #Make community data
        self.com_data=self.com_processor.make_com_data(
                                                        self.agents, 
                                                        self.problem_conf['t_init'], 
                                                        self.problem_conf['t_end'], 
                                                        self.problem_conf['pv_factor'])
        #get profiles from agents                                            
        self.get_profiles()
        self.get_agents_prefs()
        
        
        
        
    def get_agent_obj(self,agent_id):
        return self.agents[agent_id]
    
    def print_info(self):
        print('Created an Energy Community with the following characteristics')
        print('number of agents: ', self.num_agents)
    
            
    def assemble_agents(self):
        """Assembles the agents and returns a dictionary of agent objects built from config files."""
        agents = {}
        for agent_id, agent_conf in self.agents_conf.items():
            app_list = self._assemble_appliances(agent_conf['appliances'])
            agent_cls = self._get_class(agent_conf['agent_cls'])
            # data_proc_class=self._get_class(agent_conf['data_info']['data_proc_cls'])
            agents[agent_id] = agent_cls(agent_id, 
                                         self.agent_processor, 
                                         app_list,
                                         agent_conf)
            
        self.agents=agents
    
    def _assemble_appliances(self, appliance_list):
        """Assembles appliances for a given agent."""
        app_list = []
        for app_name in appliance_list:
            app_info = self.apps_conf[app_name]
            app_cls = self._get_class(app_info['appliance_cls'])
            app_params = app_info.get('params', {})
            app_list.append(app_cls(app_name,
                                    **app_params))
        return app_list
    
    def _get_class(self, class_name):
        """Retrieve a class object by its name."""
        return globals().get(class_name)
    
    def get_agents_prefs(self):
        # Initialize an empty list to store individual agent preferences
        agent_preferences = {}

        # Iterate over each agent and retrieve their preferences
        for agent in self.agents.values():
            agent_preferences[agent.id]=agent.get_params()
        # print(agent_preferences)
        # Concatenate all agent preferences into a single DataFrame
        prefs_df=pd.DataFrame.from_dict(agent_preferences,orient='index')

        self.com_prefs=prefs_df

    
    def get_profiles(self):
        """
        Assembles the shiftable profiles for all the agents in the community
        """
        profiles={}
        for agents in self.agents.values():
            
            profiles[agents.id]=list(agents.get_shift_profiles(self.problem_conf['dataset_unit'], 
                                                          self.problem_conf['step_size']).values())[0]
            
        self.profiles=profiles

    def get_tariffs(self,timeslot):
        df=pd.DataFrame()
        for agent in self.agents.values():
            df.loc[agent.id,'tar_buy']= agent.get_tariff(timeslot)
            
        return df
    
    def get_tariffs_by_mins(self,timeslot):
        """
        returns a pd.dataframe with the tariffs for all agents in the community
        """
        df=pd.DataFrame()
        for agent in self.agents.values():
            df.loc[agent.id,'tar_buy']= agent.get_tar_by_mins(timeslot)
            
        return df
            
            
        
