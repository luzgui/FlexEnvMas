#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 09:13:19 2023

@author: omega
"""

from __future__ import division
from pyomo.environ import *
from pyomo.opt import SolverFactory
import pandas as pd
import numpy as np
import random



def get_solution(model,agents_id):
    #importing values
    df=pd.DataFrame()
    for (aid,n) in zip(agents_id,range(len(model.I))):
        
        P=np.array([value(model.P[n,t]) for t in model.T])
        x=np.array([value(model.x[n,t]) for t in model.T])
        y=np.array([value(model.y[n,t]) for t in model.T])
        
        df['load_'+aid]=P
        df['action_'+aid]=x
        df['y_act_'+aid]=y
    
    #global    
    pv=np.array([value(model.pv[t]) for t in model.T])
    px=np.array([value(model.Px[t]) for t in model.T])
    tar=np.array([value(model.c[t]) for t in model.T])
    base=np.array([value(model.b[t]) for t in model.T])
    
    C_pos = np.array([value(model.C_pos[t]) for t in model.T])
    C_neg = np.array([value(model.C_neg[t]) for t in model.T])
    C = np.array([value(model.C[t]) for t in model.T])
    sign = np.array([value(model.sign[t]) for t in model.T])                   
    
    df['PV']=pv
    df['tar']=tar
    
    df['shift_T']=df[[k for k in df.columns if 'load' in k]].sum(axis=1)
    df['Excess']=px
    df['base']=base
    df['C_pos']=C_pos
    df['C_neg']=C_neg
    df['C']=C
    df['sign']=sign
        
        # c=np.array([value(model.c[t]) for t in model.T])
        # H=len(c)
        # nI=len(model.I)
        # agents=np.array([value(model.I[t]) for t in model.I])
        
    return df
