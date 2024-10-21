#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 17:57:46 2024

@author: omega
"""

import numpy as np

class StateUpdate():
    "This class contains the method to update the state dataframe in the environment object of FlexEnv class"
    
    def __init__(self, env):
        self.env=env
        self.vars=env.state_vars
        
        
        self.update_funcs = {
              'minutes': self.update_minutes,
              'sin': self.update_sin,
              'cos': self.update_cos,
              'load0': self.update_load0,
              'gen0': self.update_gen0,
              'excess0':self.update_excess0,
              'tar_buy': self.update_tar_buy,
              'E_prof_rem': self.update_E_prof,
              'y_s': self.update_ys,
              'pv_sum': self.update_pv_sum,
              'tar1': self.update_tar1,
              'tar2': self.update_tar2,
              'tar3': self.update_tar3,
              'tar4': self.update_tar4,
              'tar5': self.update_tar5,
              'tar6': self.update_tar6,
              'tar7': self.update_tar7,
              'tar8': self.update_tar8,
              'tar_d':self.update_tard
        }
        
    def update_features(self):
        for feature, func in self.update_funcs.items():
            if feature in self.vars:
                func()  # Call the corresponding update function

      
    def update_minutes(self):
        self.env.state['minutes']=self.env.minutes

        
        
    def update_sin(self):
        self.env.state['sin']=np.sin(2*np.pi*(self.env.minutes/self.env.min_max))
        
    def update_cos(self):
        self.env.state['cos']=np.cos(2*np.pi*(self.env.minutes/self.env.min_max))
    
    def update_load0(self):
        for ag in self.env.agents_id:
            self.env.state.loc[ag,'load0']=self.env.data.loc[ag,self.env.tstep]['load']
    
    def update_gen0(self):
        for ag in self.env.agents_id:
            self.env.state.loc[ag,'gen0']=self.env.data.loc[ag,self.env.tstep]['gen']
            
    def update_excess0(self):
        for ag in self.env.agents_id:
            self.env.state.loc[ag,'excess0']=self.env.data.loc[ag,self.env.tstep]['excess']
            
    def update_tar_buy(self):
        tars=self.env.com.get_tariffs_by_mins(self.env.tstep)
        for ag in self.env.agents_id:
            self.env.state.loc[ag,'tar_buy']=tars.loc[ag,'tar_buy']
    
    def update_E_prof(self):
        for ag in self.env.agents_id:
            if self.env.tstep==self.env.tstep_init or self.env.minutes == 0:
                self.env.state.loc[ag,'E_prof_rem']=self.env.agents_params.loc[ag,'E_prof']
            else:
                new_e_val=round(self.env.action.loc[ag]['action']*self.env.com.agents[ag].apps[0].base_load*self.env.tstep_size/60, 2)
                self.env.state.loc[ag,'E_prof_rem']-=new_e_val
                self.env.state.loc[ag,'E_prof_rem']=round(self.env.state.loc[ag,'E_prof_rem'],2)
    
    def update_ys(self):
        for ag in self.env.agents_id:
            if self.env.tstep==self.env.tstep_init or self.env.minutes == 0:
                self.env.state.loc[ag,'y_s']=0.0
            else:
                self.env.state.loc[ag,'y_s']+=self.env.action.loc[ag]['action']
                
    def update_pv_sum(self):
        for ag in self.env.agents_id:
            self.env.state.loc[ag,'pv_sum']=self.env.data.loc[ag][self.env.tstep:self.env.tstep_init+self.env.Tw]['excess'].sum()
            
                
    def update_tard(self):
        for ag in self.env.agents_id:
            tars=self.env.get_episode_data().loc[ag]['tar_buy']
            future_vals=tars.loc[self.env.tstep:self.env.tstep_init+self.env.Tw-1]
            
            if future_vals.empty:
                self.env.state.loc[ag,'tar_d']=self.env.state.loc[ag,'tar_buy']
            else:
                self.env.state.loc[ag,'tar_d']=self.env.state.loc[ag,'tar_buy']-min(future_vals)
       
        
    def update_tar(self,t_ahead,var):
        for ag in self.env.agents_id:
            tars=self.env.get_episode_data().loc[ag]['tar_buy']
            tstep_to_use=self.env.tstep+4*t_ahead

            if tstep_to_use >= self.env.tstep_init+self.env.Tw-1:
                # self.env.state.loc[ag,var]=1.0
                self.env.state.loc[ag,var]=self.env.com.agents[ag].tar_max
            else:
                self.env.state.loc[ag,var]=tars.loc[tstep_to_use]

  
    def update_tar1(self):
        return self.update_tar(1,'tar1')
    def update_tar2(self):
        return self.update_tar(2,'tar2')
    def update_tar3(self):
        return self.update_tar(3,'tar3')
    def update_tar4(self):
        return self.update_tar(4,'tar4')
    def update_tar5(self):
        return self.update_tar(5,'tar5')        
    def update_tar6(self):
        return self.update_tar(6,'tar6')
    def update_tar7(self):
        return self.update_tar(7,'tar7')
    def update_tar8(self):
        return self.update_tar(8,'tar8')    
        
        
                
        
                
            
            
        
        
        
        
        
        
        
        
        
        
        

               
        
    
        
        
