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
from termcolor import colored
import os




def test(tenv, tester, n_episodes,results_path, plot=True):
    """save_metrics=True will make Results from testing (plots and metrics) to be saved in the trainable folder (results_path) alongside the checkpoints"""

    episode_metrics=pd.DataFrame()
    env_state_conc=pd.DataFrame()
    
    
    #choose the policy mapping function according to policies in algorithm config
    
    pols=list(tester.config.policies.keys())
    if 'pol_ag' in pols[0]: # each agent has its policy
        policy_mapping_func=policy_mapping_fn_test
    elif pols[0]=='shared_pol': # theres a shared policy for agents
        policy_mapping_func=policy_mapping_fn_test_shared
        
        

    k=0
    
    while k < n_episodes:
    
        mask_track=[]
    
        obs = tenv.reset()
            
        
        T=tenv.Tw*1
        # num_days_test=T/tenv.tstep_per_day
    
        # metrics_episode=pd.DataFrame(columns=['cost','delta_c','gamma'], index=range(T))
    
        for i in range(T):
        # while any(tenv.done)==False:
            
            actions=get_actions(obs, tester, tenv.agents_id,policy_mapping_func)
            obs, reward, done, info = tenv.step(actions)
            # ic(len(tenv.state_hist.loc['ag1']))
            # ic(len(tenv.action_hist.loc['ag1']))
            # ic(len(tenv.reward_hist.loc['ag1']))
            
            # if done['__all__']==True:
            #     print('episode terminated')
            #     print('timestep: ', tenv.tstep)
            #     break
                
            #compute metrics per episode
            # cost=max(0,action*tenv.profile[0]-tenv.excess0)*tenv.tar_buy
            # delta_c=(tenv.load0+action*tenv.profile[0])-tenv.gen0
            # gamma=self_suf(tenv,action)
            
            
            # metrics_episode.iloc[i]=[cost,delta_c,gamma]
            
            
        # we are summing the total cost and making a mean for delta    

        from auxfunctions_shiftenv import get_post_data
        full_state, env_state=get_post_data(tenv)
        
        episode_metrics=pd.concat([get_episode_metrics(full_state,
                                                       env_state,
                                                       tenv,k),
                                                       episode_metrics])
        
                
        env_state_conc=pd.concat([env_state_conc, env_state.reset_index()],
                                 axis=0).reset_index(drop=True)
        
                
        if plot: 
            # from plotutils import makeplot
            makeplot(T,
                      [],
                      env_state['shift_T'],
                      env_state[[k for k in env_state.columns if 'shift_ag' in k and 'coef' not in k]],
                      env_state['gen0_ag1'],
                      env_state['baseload_T'],
                      env_state['tar_buy'],
                      tenv, 
                      env_state['Cost_shift_T'].sum(),
                      0) #
            
            
            
            
        k+=1
        
    #save results in a fodler inside the trainable folder
    if results_path:
        test_results_path=os.path.join(results_path,'test_results')
        if not os.path.exists(test_results_path) and not os.path.isdir(test_results_path):
            os.makedirs(test_results_path)
            print(colored('folder created' ,'red'),test_results_path)
        
        
        filename_metrics='metrics_'+tenv.env_config['exp_name']+'_'+str(k)+'_eps'+'.csv'
        filename_metrics=os.path.join(test_results_path,filename_metrics)
        episode_metrics.to_csv(filename_metrics)
        print(colored('Metrics saved to' ,'red'),filename_metrics)
        
        filename_env_state='env_state_'+tenv.env_config['exp_name']+'_'+str(k)+'_eps'+'.csv'
        filename_env_state=os.path.join(test_results_path,filename_env_state)
        env_state_conc.to_csv(filename_env_state)
        print(colored('Metrics saved to' ,'red'),filename_env_state)
        
        filename=[filename_metrics,filename_env_state]
        
    else:
        filename_metrics=''
        filename_env_state=''
        filename=[filename_metrics,filename_env_state]
        
        
    return full_state, env_state_conc, episode_metrics, filename






def get_episode_metrics(full_state,env_state,environment,k):
    agents_id=full_state.index.unique()
    metrics=pd.DataFrame(index=full_state.index.unique())
    Total_load=[]
    
    
    #Per agent metrics
    
    
    for ag in agents_id:
        
        
        
        # per agent cost considering the 
        # full_state.loc[ag,'cost']=(full_state.loc[ag]['action']*environment.profile[ag][0]-full_state.loc[ag]['excess0'])*full_state.loc[ag]['tar_buy']
        
        
        

        # # cost=pd.DataFrame([max(0,k) for k in full_state.loc[ag]['cost']])
        # pos_cost=pd.DataFrame([max(0,k) for k in full_state.loc[ag]['cost']])
        # full_state.loc[ag,'cost_pos']=pos_cost.values        
        
        # Self-Sufficiency
        # full_state.loc[ag,'load']=full_state.loc[ag]['action']*environment.profile[ag][0]
        
        # full_state.loc[ag,'selfsuf']=full_state.loc[ag][['load','excess0']].min(axis=1)/environment.E_prof.loc[ag]['E_prof'] #dh its just here as a patch
        
        
        
        #group everything
        metrics.loc[ag,'cost']=full_state.loc[ag,'r_cost_pos'].sum()
        # metrics.loc[ag,'selfsuf']=full_state.loc[ag,'selfsuf'].sum()
        
        # should per agent x_ratio have the sharing coefficient? 
        metrics.loc[ag,'x_ratio']=full_state.loc['ag1']['excess0'].sum()/environment.E_prof.loc[ag,'E_prof']
        
        
    
    #community metrics
    # SS_temp=pd.concat([full_state[['minutes','load']]\
    #     .set_index('minutes')\
    #     .groupby(level=0).sum(),
    #     full_state.iloc[0:environment.Tw][['minutes','excess0']].set_index('minutes')],axis=1)
    
    # metrics.loc['com','selfsuf']=SS_temp.min(axis=1).sum()/(environment.E_prof['E_prof'].sum()*
    #                               (environment.Tw/environment.tstep_per_day)) #number of days
    
    metrics.loc['com','selfsuf']=env_state[['shift_T','excess']].min(axis=1).sum()/environment.E_prof['E_prof'].sum()
    
    #compute the ratio between energy needed and excess available
    # E_ratio=full_state.loc['ag1']['excess0'].sum()/environment.E_prof['E_prof'].sum()
    
    E_ratio=env_state['excess'].sum()/environment.E_prof['E_prof'].sum()
    metrics.loc['com','x_ratio']=E_ratio
    
    metrics.loc['com','x_sig']=sigmoid(0.5,6.2,2,1,E_ratio)

    
    
    
    # metrics.loc['com','selfsuf']=full_state['selfsuf'].sum()/environment.num_agents
    # metrics.loc['com','cost']=full_state['cost_pos'].sum()
    metrics.loc['com','cost']=env_state['Cost_shift_T'].sum() #the cost of consuming aggregated shiftable load
    
    
    #Binary metrics
    min_cost=full_state['tar_buy'].min()*environment.E_prof['E_prof'].sum()
    #1 if the cost is greater that mininmum tarif cost of community
    metrics.loc['com','y']=int(bool(metrics.loc['com','cost'] > min_cost))
    
    #cost variation relative to the min cost (-1 zero cost)
    # min_cost=environment.tar_buy*environment.E_prof['E_prof'].sum()
    metrics['cost_var']=(metrics.loc['com']['cost']-min_cost)/min_cost
    
    
    #year season
    metrics['day']=environment.tstep_init/environment.tstep_per_day
    metrics['season']=get_season(metrics.loc['com']['day'])
    
    
    
    #create index for test episode number
    metrics['test_epi']=k
    metrics_out=metrics.set_index('test_epi',drop=True, append=True)
    
    
    #year season
    
    
    
    return metrics_out
        


def get_season(day):
        # "day of year" ranges for the northern hemisphere
    spring = range(80, 172)
    summer = range(172, 264)
    fall = range(264, 355)
    # winter = everything else
    
    if day in spring:
        season = 'spring'
    elif day in summer:
        season = 'summer'
    elif day in fall:
        season = 'fall'
    else:
        season = 'winter'
    
    return season
        
    
    # full_track=pd.concat([state_track, action_reward_track,metrics_episode],axis=1)
    # full_track_filter=full_track[['tstep','minutes','gen0','load0','delta0','excess0','tar_buy','E_prof', 'action', 'reward','cost', 'delta_c', 'gamma']]

    # #gamma per epsidode is beying divided by the total amount of energy that appliances need to consume.
    # metrics_experiment.iloc[k]=[metrics_episode['cost'].sum(),
    #                             metrics_episode['delta_c'].mean(), 
    #                             metrics_episode['gamma'].sum()/(num_days_test*tenv.E_prof)] 
    
    # # print(metrics_episode['gamma'].sum()/(num_days_test*tenv.E_prof))
    # # print(full_track['load0'].sum())
    
    # print(tenv.E_prof/full_track['excess0'].sum())


      