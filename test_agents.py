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
from auxfunctions_shiftenv import get_post_data




def test(tenv, tester, n_episodes):

    # n_episodes=1
    episode_metrics=pd.DataFrame()
    
    
    
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


        full_state, env_state=get_post_data(tenv)
        
        episode_metrics=pd.concat([get_episode_metrics(full_state, tenv,k),episode_metrics])
        
        # makeplot(T,
        #          [],
        #          env_state['shift_T'],
        #          env_state['gen0'],
        #          env_state['load_T'],
        #          env_state['tar_buy'],
        #          tenv, 
        #          0,
        #          0) #
            
        k+=1
        
    
        
    return full_state, env_state, episode_metrics



def get_episode_metrics(full_state,environment,k):
    agents_id=full_state.index.unique()
    metrics=pd.DataFrame(index=full_state.index.unique())
    Total_load=[]
    
    
    #Per agent metrics
    for ag in agents_id:
        # cost
        full_state.loc[ag,'cost']=(full_state.loc[ag]['action']*environment.profile[ag][0]-full_state.loc[ag]['excess0'])*full_state.loc[ag]['tar_buy']
        

        # cost=pd.DataFrame([max(0,k) for k in full_state.loc[ag]['cost']])
        pos_cost=pd.DataFrame([max(0,k) for k in full_state.loc[ag]['cost']])
        full_state.loc[ag,'cost_pos']=pos_cost.values        
        
        # Self-Sufficiency
        full_state.loc[ag,'load']=full_state.loc[ag]['action']*environment.profile[ag][0]
        
        full_state.loc[ag,'selfsuf']=full_state.loc[ag][['load','excess0']].min(axis=1)/environment.E_prof.loc[ag]['E_prof'] #dh its just here as a patch
        
        #group everything
        metrics.loc[ag,'cost']=full_state.loc[ag,'cost_pos'].sum()
        metrics.loc[ag,'selfsuf']=full_state.loc[ag,'selfsuf'].sum()
        

        
        
        
    #community metrics
    SS_temp=pd.concat([full_state[['minutes','load']]\
        .set_index('minutes')\
        .groupby(level=0).sum(),
        full_state.iloc[0:environment.Tw][['minutes','excess0']].set_index('minutes')],axis=1)
    
    metrics.loc['com','selfsuf']=SS_temp.min(axis=1).sum()/(environment.E_prof['E_prof'].sum()*
                                  (environment.Tw/environment.tstep_per_day)) #number of days
    

    

    
    # metrics.loc['com','selfsuf']=full_state['selfsuf'].sum()/environment.num_agents
    metrics.loc['com','cost']=full_state['cost_pos'].sum()
    
    #create index for test episode number
    metrics['test_epi']=k
    metrics_out=metrics.set_index('test_epi',drop=True, append=True)

    return metrics_out
        
    
    
    # full_track=pd.concat([state_track, action_reward_track,metrics_episode],axis=1)
    # full_track_filter=full_track[['tstep','minutes','gen0','load0','delta0','excess0','tar_buy','E_prof', 'action', 'reward','cost', 'delta_c', 'gamma']]

    # #gamma per epsidode is beying divided by the total amount of energy that appliances need to consume.
    # metrics_experiment.iloc[k]=[metrics_episode['cost'].sum(),
    #                             metrics_episode['delta_c'].mean(), 
    #                             metrics_episode['gamma'].sum()/(num_days_test*tenv.E_prof)] 
    
    # # print(metrics_episode['gamma'].sum()/(num_days_test*tenv.E_prof))
    # # print(full_track['load0'].sum())
    
    # print(tenv.E_prof/full_track['excess0'].sum())


      