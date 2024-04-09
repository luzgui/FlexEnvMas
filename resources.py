#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 14:29:21 2024

@author: omega
"""


class BaseResource():
    def __init__(self, name):
        self.name=name

    def get_action_space(self):
        "returns the action space to be used in RL environment"
        pass
    
    def get_mask(self):
        raise NotImplementedError("Appliances must have a mask method")
    
        
    



class ShiftApp(BaseResource):
    def __init__(self, 
                 name, 
                 base_load, 
                 duration):
        
        self.base_load=base_load
        self.duration=duration
        self.name=name
    
    def get_power(self):
        return self.base_load*self.duration
    
    def get_profile(self, unit: str, step_size):
        """
        returns the profile in kw or kwh
        """
        mins_per_hour=60.0
        
        if unit=='kwh':
            load_kwh=self.base_load*(step_size/mins_per_hour)
            return [load_kwh]*int((self.duration/step_size))
        elif unit=='kw':
            return [self.base_load]*int((self.duration/step_size))
        
    def get_total_energy(self):
        "returns the app total energy consumption in kwh"
        return self.get_power()*(1/60)
    
    def get_action_space(self):
        "returns the action space to be used in RL environment"
        pass

    
class StorageApp(BaseResource):
    def __init__(self, name, max_ch, max_disch,capacity):
        self.max_ch=max_ch
        self.max_disch=max_disch
        self.cap=capacity
        self.name=name
        
    def get_profile(self):
        pass


    
    
    
    
    
    
    
    