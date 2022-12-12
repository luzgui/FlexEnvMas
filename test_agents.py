#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 16:42:06 2022

@author: omega
"""
import pandas as pd

from auxfunctions_shiftenv import *
from shiftenvRLlib_mas import ShiftEnvMas
from plotutils import makeplot

def test(tenv, tester, n_episodes):

    # n_episodes=1
    
    metrics_experiment=pd.DataFrame(columns=['cost','delta_c','gamma'], 
                                    index=range(n_episodes))
    k=0
    
    while k < n_episodes:
    
        mask_track=[]
    
        obs = tenv.reset()
        
        
        T=tenv.Tw*1
        # num_days_test=T/tenv.tstep_per_day
    
        # metrics_episode=pd.DataFrame(columns=['cost','delta_c','gamma'], index=range(T))
    
        for i in range(T):
            actions=get_actions(obs, tester, tenv.agents_id,policy_mapping_fn)
            obs, reward, done, info = tenv.step(actions)
            
            
            #compute metrics per episode
            # cost=max(0,action*tenv.profile[0]-tenv.excess0)*tenv.tar_buy
            # delta_c=(tenv.load0+action*tenv.profile[0])-tenv.gen0
            # gamma=self_suf(tenv,action)
            
            
            # metrics_episode.iloc[i]=[cost,delta_c,gamma]
            
            
        # we are summing the total cost and making a mean for delta    
        from auxfunctions_shiftenv import get_post_data
        full_state, env_state=get_post_data(tenv)
        
        
        
        from plotutils import makeplot
        makeplot(T,
                 [],
                 env_state['shift_T'],
                 env_state['gen0'],
                 env_state['load_T'],
                 env_state['tar_buy'],
                 tenv, 
                 0,
                 0) #
            
        k+=1
        
        
        return full_state, env_state
        