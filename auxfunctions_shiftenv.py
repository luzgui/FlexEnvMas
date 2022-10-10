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
    

def make_env_data_mas(data,timesteps, load_id, pv_factor, num_agents):
    "(data: timeseries, laod_num: house number, pv_factor"
    
    df=pd.DataFrame()
    
    load_names=['load'+str(k) for k in range(num_agents)]
    
    df['minutes']=data.iloc[0:timesteps]['minutes']
    df[load_names]=data.iloc[0:timesteps][load_id]
    df['gen']=pv_factor*abs(data.iloc[0:timesteps]['PV'])
    df['delta']=df[load_names].sum(axis=1)-df['gen']
    df['excess']=[max(0,-df['delta'][k]) for k in range(timesteps)] 
    
    # gen=np.zeros(timesteps)
    # minutes=make_minutes(data,timesteps) # make minutes vector

    return df


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
    analysis_object = ExperimentAnalysis(experiment_path, default_metric=metric, default_mode=mode)
    
    df=analysis_object.dataframe(metric,mode) #get de dataframe results

    #identify the dir where is the best checkpoint according to metric and mode
    bestdir=analysis_object.get_best_logdir(metric,mode)

    #get the best trial checkpoint
    checkpoint=analysis_object.get_best_checkpoint(bestdir,metric,mode)
    print(mode,checkpoint)
    #recover best agent for te
    
    return checkpoint, df
    




