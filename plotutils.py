#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 11:46:27 2022

@author: omega
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy.random as rnd
import time
import random as rnd

def makeplot(T,soc,sol,gen,load,delta,env):

    fig, ax = plt.subplots(figsize=(10,7))
    t=np.arange(0,len(gen),1)
    ax.plot(t, load,label='Unflexible load')
    # ax.plot(sol+load,label='load+bat_charge')
    # ax.plot(sol,label='On/Off')
    ax.plot(soc,label='Shiftable load')
    ax.plot(gen,label='PV generation')
    ax.plot(delta,label='delta')
    
    ax.grid()
    ax.legend()
    ax.set(xlabel='Timesteps', ylabel='kWh',
           title='Shiftable device solution')
#     plt.show()
#     time.sleep(0.1)
#     return(ax)
