#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 14:53:54 2024

@author: omega
"""
import pandas as pd
from resources import ShiftApp
import numpy as np

class Agent():
    def __init__(self,
                 agent_id,
                 processor,
                 app_list, 
                 agent_conf):
        
        self.conf=agent_conf
        self.id=agent_id
        self.load_id=agent_conf['params']['load_id']
        self.pv_id=agent_conf['params']['pv_id']
        self.apps=app_list
        self.del_time=agent_conf['params']['delivery_time']
        self.processor=processor
        
        self.data=self.processor.get_agent_data(['minutes',self.load_id,self.pv_id])
        
        self.agent_params=self.get_params()
        self.tariff=self.make_tariff()
        self.tar_max=max(self.tariff)
        
    
        
    def get_observation(self):
        "must return the agent observation"
        pass
    
    def make_tariff(self):
        """Generate tariff time-series"""
        num_timesteps = (24*60)//self.processor.step_size
        tar_conf=self.conf['tariffs']
        tar_type=tar_conf['tar_type']
        
        
        if tar_type == 'double_rate':
            dr_conf=tar_conf['double__rate_params']
            
            hour_start = dr_conf['empty_start']
            hour_end = dr_conf['empty_end']
            tariff_series = np.zeros(num_timesteps)
            for i in range(num_timesteps):
                if (i * self.processor.step_size >= hour_start * 60) and (i * self.processor.step_size <= hour_end * 60):
                    tariff_series[i] = dr_conf['no_empty_val']
                else:
                    tariff_series[i] = dr_conf['empty_val']
                    
        elif tar_type == 'flat':
            tariff_series = np.full(num_timesteps, tar_conf['flat_val'])
        else:
            tariff_series = np.zeros(num_timesteps)
            print('No tariffs defined')
            
        return tariff_series
    

        
    
    def get_tariff_24(self):
        "returns the tariff for the day hourly"
        return self.tariff[0::int(60/self.processor.step_size)]
    
    def get_tar_by_timeslot(self,timeslot):
        return self.tariff[timeslot]
    
    
    def get_tar_by_mins(self,timeslot):
        
        minutes=self.data.loc[timeslot]['minutes']
        "get tarrifs in €/kWh for argument tstep_ahead (integer number of timesteps) ahead of self.minutes" 
        tar_conf=self.conf['tariffs']
        tar_type=tar_conf['tar_type']
        
        
        if tar_type == 'double_rate':
            dr_conf=tar_conf['double__rate_params']
            
            hour_start = dr_conf['empty_start']
            hour_end = dr_conf['empty_end']
            
            #condition is enforced based on minutes
            if minutes >= self.processor.step_size*(60/self.processor.step_size)*hour_start and minutes <=self.processor.step_size*(60/self.processor.step_size)*hour_end:
                tar_buy=dr_conf['no_empty_val']
            else:
                tar_buy=dr_conf['empty_val']
            # self.tar_buy=0.17*(1-(self.gen/1.764)) #PV indexed tariff 
            # tar_sell=0.0 # remuneration for excess production
        elif tar_type == 'flat':
            tar_buy=tar_conf['flat_val']
            # tar_sell=0.0
        return tar_buy
    
    
    def get_shift_profiles(self,unit, step_size):
        """
        Composes the appliance profile for shiftable apps using the profiles of individual apps.
        """
        appliance_profile = {}
        for app in self.apps:
            if isinstance(app, ShiftApp): #must be a ShiftApp object
                app_profile = app.get_profile(unit, step_size)
                appliance_profile[app.name]=app_profile
        return appliance_profile
    
    def get_params(self):
        df=pd.DataFrame()
        app=self.apps[0] # this is hardcoded tio get 1 single app. Adapt if more than 1 app per agent is needed
        # import pdb
        # pdb.pdb.set_trace()
        df={'T_prof': len(app.get_profile('kwh',15)),
            'E_prof': app.get_total_energy(),
            't_deliver': self.del_time,
            'tar_type': self.conf['tariffs']['tar_type']}
        
        return df
    
    
    # def rename_data_cols(self):
    #     """
    #     When changing the load_id from configs, agent data columns names must remain refering to the agent id
    #     """
        
    #     if self.id != self.load_id:
    #         print('mudei cenas')
    #         # Define the new column names
    #         new_column_names = ['minutes', self.id, 'PV'+self.id]
    
    #         # Rename columns using the new column names
    #         self.data.rename(columns=dict(zip(self.data.columns, new_column_names)), inplace=True)
    #     else:
    #         pass
        
    
    
    
    # def get_limits(self, mode,var):
    #     df=self.data.describe()
    #     df.rename(index={'ag2': 'load', 'PV2': 'gen'}, inplace=True)
        
        
    #     return df[mode, var]
        
        
        
        
        
        
    

        