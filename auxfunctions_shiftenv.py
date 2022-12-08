#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 14:37:41 2022

@author: omega
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy.random as rnd
import time
import random as rnd

from ray.tune import analysis
from ray.tune import ExperimentAnalysis
from pathlib import Path




def make_cyclical(series, max_val): # transforma valores como dia e hora em valores cíclicos de sin e cos para remover efeitos indesejados
    data_sin = np.sin( 2 * np.pi * series / max_val )
    data_cos = np.cos( 2 * np.pi * series / max_val )
    return list(data_sin), list(data_cos)


def make_minutes(data, timesteps):
    
    time=pd.DatetimeIndex(pd.to_datetime(data[0:timesteps,0]))
    hour=time.hour.to_numpy
    minute=time.minute.to_numpy
    minutes=[hour()[k]*60+minute()[k] for k in range(timesteps)]
    
    return minutes
    

def make_env_data_mas(data,timesteps, load_id, pv_factor,num_agents, agents_id):
    "(data: timeseries, load_num: house number, pv_factor, num_agents, agents_id (list string)"
    

  
    df=pd.DataFrame()
    
    load_names=['load_ag'+str(k) for k in range(num_agents)]
    
    df['minutes']=data.iloc[0:timesteps]['minutes']
    df[load_id]=data.iloc[0:timesteps][load_id]
    df['gen']=pv_factor*abs(data.iloc[0:timesteps]['PV0'])
    
    # delta and excess are COLLECTIVE, i.e computed based on aggregated quantities
    df['delta']=df[load_id].sum(axis=1)-df['gen']
    df['excess']=[max(0,-df['delta'][k]) for k in range(timesteps)] 
    
    frames=[]
    for lid,aid in zip(load_id,agents_id):
        df_temp=df.copy()[['minutes',lid,'gen','delta','excess']]
        label=[aid]*len(df_temp)
        # df_temp['agent_id']=label
        df_temp.rename(columns={lid:'load'}, inplace=True)
        
        frames.append(df_temp)    
        
    df_final=pd.concat(frames, keys=agents_id)
        

    return df_final


def get_raw_data(file, datafolder):
    
    dt=15
    
    cons_data=pd.read_excel(datafolder + '/' + file, 'Total Consumers')
    cons_data.columns=['ag'+str(k) for k in range(len(cons_data.columns))]
    
    prod_data=pd.read_excel(datafolder + '/' + file, 'Total Producers')
    prod_data.columns=['PV'+str(k) for k in range(len(prod_data.columns))]
    
    
    #create a vector of minutes
    mins=pd.DataFrame(np.tile(np.linspace(0,1440-dt,num=int((24*60/dt))),366), columns=['minutes'])
    
    cons_data=pd.concat([mins,cons_data,prod_data],axis=1)

    return cons_data

    

def make_env_data(data,timesteps, load_id, pv_factor):
    "(data: timeseries, laod_num: house number, pv_factor"
    
    df=pd.DataFrame()
    
    df['minutes']=data.iloc[0:timesteps]['minutes']
    df['load']=data.iloc[0:timesteps][load_id]
    df['gen']=pv_factor*abs(data.iloc[0:timesteps]['PV'])
    df['delta']=df['load']-df['gen']
    df['excess']=[max(0,-df['delta'][k]) for k in range(timesteps)] 
    
    # gen=np.zeros(timesteps)
    # minutes=make_minutes(data,timesteps) # make minutes vector

    return df


def self_suf(env,action):
    # if var > 0 and env.gen0 != 0:
        # g=var
    # elif var <= 0:
        # g=env.load0
    if env.gen0 != 0:
        # g=min(env.load0+action*env.profile[0],env.gen0) #for the total load 
        g=min(action*env.profile[0],env.excess0) #self sufficency just for the load
    elif env.gen0 ==0:
        g=0
    
    return g


def get_checkpoint(log_dir,exp_name,metric,mode):
    #Recover the tune object from the dir
    # The trainable must be initialized # reuslts must be stored in the same analysis object
    # metric='training_iteration'
    
    experiment_path=os.path.join(log_dir, exp_name)
    # experiment_path=os.path.join(raylog, 'GoodExperiments')
    analysis_object = ExperimentAnalysis(experiment_path,
                                         default_metric=metric, 
                                         default_mode=mode)
    
    

    # state=os.path.join(experiment_path, 'experiment_state-2022-12-05_11-03-59.json')
    # state_dict={}
    
    # state=local_trial.get_json_state
    
    # trial2=_load_trial_from_checkpoint(state)
        
    
    #get the best trial checkpoint
    local_trial=analysis_object.get_best_trial()
    
    #identify the dir where is the best checkpoint according to metric and mode
    best_local_dir=analysis_object.get_best_logdir(metric,mode)
    
    checkpoint=analysis_object.get_best_checkpoint(best_local_dir,metric,mode)
    
    #dataframes
    df=analysis_object.dataframe(metric,mode) #get de dataframe results
    best_df=analysis_object.best_result
    
    # bestdir=local_best_dir    
    checkpoint=analysis_object.get_best_checkpoint(best_local_dir,metric,mode)
    
    
    print(mode,checkpoint)
    #recover best agent for te
    
    return checkpoint, df
    

def get_post_data(menv):
    df=pd.DataFrame()
    T=menv.Tw
    for aid in menv.agents_id:
        #we need to take out the last observation because its allways one timestep ahead
        state_hist=menv.state_hist.loc[aid][0:T] 
        action_hist=menv.action_hist.loc[aid]    
        reward_hist=menv.reward_hist.loc[aid]
    
        # df=pd.concat([state_hist,action_hist, reward_hist],axis=1)
    
        df=pd.concat([df,
                       pd.concat([state_hist,action_hist, reward_hist],
                    axis=1)])
    
    #names for variables
    columns_names=[]
    shift_columns_names=['shift_'+aid for aid in menv.agents_id]
    reward_columns_names=['reward_'+aid for aid in menv.agents_id]
    load_columns_names=['load_'+aid for aid in menv.agents_id]
    
   
    columns_names.extend(load_columns_names)
    columns_names.extend(shift_columns_names)
    columns_names.extend(reward_columns_names)
    
    columns_names.extend(['shift_T','load_T','gen0','reward_T','tar_buy'])
    
    #make a new dataframe to store the solutions
    df_post=pd.DataFrame(columns=columns_names)
    
    for aid in menv.agents_id:
        var_ag=[v for v in df_post.columns if aid in v]
        
        for var in var_ag:
            if 'load' in var:
                df_post[var]=df.loc[aid,'load0'].values
            
            if 'shift' in var:
                df_post[var]=df.loc[aid,'action'].values*menv.profile[aid][0]
                
            if 'reward' in var:
                df_post[var]=df.loc[aid,'reward'].values
    
    
    df_post['shift_T']=df_post[shift_columns_names].sum(axis=1)
    df_post['load_T']=df_post[load_columns_names].sum(axis=1)
    df_post['reward_T']=df_post[reward_columns_names].sum(axis=1)
    
    
    
    df_post['gen0']=df.loc[menv.agents_id[0],'gen0'].values #pv production is the same for all and it is collective. So we can use any agent on the agents_id list
    df_post['tar_buy']=df.loc[menv.agents_id[0],'tar_buy'].values
    
                
    return df, df_post

