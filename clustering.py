#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 16:53:35 2023

@author: omega
"""

from tslearn.clustering import TimeSeriesKMeans

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tslearn.clustering import TimeSeriesKMeans
from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, \
    TimeSeriesResampler

import os
from os import path
from pathlib import Path

from auxfunctions_shiftenv import *

cwd=Path.cwd()
datafolder=cwd / 'Data'
raylog=cwd / 'raylog'
prof_folder=raylog / 'profiles'
resultsfolder=cwd / 'Results'

data = get_raw_data('Dataset_gecad_changed.xlsx', datafolder.as_posix())



#%% Data Clustering
seed=1000
model = TimeSeriesKMeans(n_clusters=3, metric="dtw",
                         max_iter=100, random_state=seed)


data=data.set_index(pd.date_range('1/1/2000', periods=len(data), freq='15T'))
series=data.squeeze()

series30=series.resample('0.5H').mean()

# ts=data.iloc[1:100][['ag0','ag1','ag3','ag4','ag5','ag6','ag7','ag8']]
ts=series30.iloc[:,1:52]
ts_np=ts.values
ts_np=np.transpose(ts_np)


ts_r=np.reshape(ts_np, (ts_np.shape[0],ts_np.shape[1],1))

clusters=model.fit_predict(ts_r)


#%%
centers=pd.read_csv(resultsfolder / 'Clustering' / 'centers_k=4_fulldata.csv')



plt.figure()
for k in range(model.n_clusters):
    plt.subplot(model.n_clusters,1,k + 1)
    index=np.array(np.where(clusters==k))
    index=np.reshape(index, (index.shape[1],))

    # plot_data=ts_np[index,:]
    for i in index:
        plt.plot(ts_np[i,:], "k-", alpha=.4)

    # plt.plot(model.cluster_centers_[k], "r-")
    plt.plot(model.cluster_centers_[k], "r-")
    
plt.tight_layout()
plt.show()



#%% Cross data

# clusters=np.random.randint(2, size=51) #dummy classification
# clusters[0]=2 #we allready know that the first building is a commercial one
clusters=pd.read_csv(resultsfolder / 'Clustering' / 'clusters_k=4_fulldata.csv')


data_describe=data.describe()
max_vals=data_describe.loc['max']

P_contr=[3.45,6.9,10.35] #Possible values for contracted power


# Digitize the values of x based on the bin edges
indices = np.digitize(max_vals, P_contr)
indices[indices==3]=2


#attribute contracted power to each house according to its maximum value
Pc=[]
for i in indices:
    Pc.append(P_contr[i])
data_describe.loc['Pc']=Pc #add new line to describe array


#get only consumers
data_describe_cons=data_describe.iloc[:,1:52]
data_describe_cons.loc['cluster']=clusters['0'].values

#count the number of values
Pc_counts=data_describe_cons.loc['Pc'].value_counts()


plt.figure()

data_describe_cons.loc['cluster'].plot.hist()
data_describe_cons.loc['Pc'].plot.hist()

# data_describe_cons.to_csv('data_describe.csv')




