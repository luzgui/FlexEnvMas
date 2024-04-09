#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 21:40:53 2024

@author: omega
"""

from dataprocessor import YAMLParser
from utilities import *

class StateVars():
    """
    class to hold variables to be used in state and observation (with its own parameters) and methods to manipulate and extract information regarding vars  
    """
    
    def __init__(self,vars_file):
        self.parser=YAMLParser()
        self.vars=self.parser.load_yaml(vars_file)
        # self.vars_to_use, self.vars_list=self.get_state_vars()

    def get_state_vars(self, normalize=False):
        """
        returns a dictionary with the list of vars to be used and a complete list of vars with info on if it should be used or not
        """
        #normalize all vars
        if normalize:
            self.update_normalize()

            
        vars_list = {k: self.vars[k]['use'] for k in self.vars.keys()}
        # Extract variables to use
        vars_to_use = {k: self.vars[k] for k in self.vars.keys() if self.vars[k]['use']}
        # Remove 'use' key from extracted variables
        vars_to_use = {k: v.copy() for k, v in vars_to_use.items()}
        for v in vars_to_use.values():
            del v['use']
            
        return vars_to_use, vars_list
    
    def update_normalize(self):
            for var in self.vars.keys():
                self.update_var(var, 'max', 1)
                self.update_var(var, 'min', -1)

    
    
    def update_var(self, variable_name, value_key, new_value):
        """Updates a single variable"""
        
        if variable_name in self.vars and value_key in self.vars[variable_name]:
            self.vars[variable_name][value_key] = new_value
        else:
            print(f"Variable '{variable_name}' or value key '{value_key}' not found.")
    
    
    def update_var_bulk(self, substring, value_key,new_value):
        """Updates all variables containing substring"""
        for variable_name in self.vars:
            if substring in variable_name:
                self.update_var(variable_name, value_key,new_value)
                
    def update_var_list(self, substrings, value_key, new_value):
        """Updates all variables containing any of the substring list elements"""
        for variable_name in self.vars:
            if any(substring in variable_name for substring in substrings):
                self.update_var(variable_name, value_key,new_value)


