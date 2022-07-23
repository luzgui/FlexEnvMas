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
    

def make_env_data(data,timesteps, load_num, pv_factor):
    "(data: timeseries, laod_num: house number, pv_factor"
    load=data[0:timesteps,load_num] # Escolhe timestep (um número) 
    gen=abs(data[0:timesteps,0]) # Primeira coluna da data
    # gen=np.zeros(timesteps)
    # minutes=make_minutes(data,timesteps) # make minutes vector
    minutes=data[0:timesteps,-1]
    env_data=np.vstack((gen*pv_factor,1*load,minutes)).T # Duas colunas, a primeira retrata
    return env_data

