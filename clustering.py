#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 16:53:35 2023

@author: omega
"""

from tslearn.clustering import TimeSeriesKMeans

import numpy as np
import matplotlib.pyplot as plt

from tslearn.clustering import TimeSeriesKMeans
from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, \
    TimeSeriesResampler




data = get_raw_data('Dataset_gecad_changed.xlsx', datafolder.as_posix())


seed=1000
model = TimeSeriesKMeans(n_clusters=3, metric="dtw",
                         max_iter=100, random_state=seed)


# ts=data.iloc[1:100][['ag0','ag1','ag3','ag4','ag5','ag6','ag7','ag8']]
ts=data.iloc[1:1000,1:52]

ts_np=ts.values
ts_np=np.transpose(ts_np)


ts_r=np.reshape(ts_np, (ts_np.shape[0],ts_np.shape[1],1))

clusters=model.fit_predict(ts_r)



plt.figure()
for k in range(model.n_clusters):
    plt.subplot(model.n_clusters,1,k + 1)
    index=np.array(np.where(clusters==k))
    index=np.reshape(index, (index.shape[1],))

    # plot_data=ts_np[index,:]
    for i in index:
        plt.plot(ts_np[i,:], "k-", alpha=.4)

    plt.plot(model.cluster_centers_[k], "r-")
    
plt.tight_layout()
plt.show()