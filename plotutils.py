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


# def makeplot(T, delta, sol, gen, load, tar, env, var_1, var_2):

#     load_full = load+sol

#     fig, ax = plt.subplots(figsize=(10, 7))
#     t = np.arange(0, T, 1)
#     # ax.plot(delta,label='delta_c')
#     ax.plot(t, load_full[0:T], label='Unflexible load')

#     # ax.plot(sol,label='appliance')
#     ax.fill_between(t, sol.values.astype('float64'),
#                     facecolor="none", hatch="////", edgecolor='#1f77b4')
#     # ax.plot(load[0:T],label='Shiftable load')
#     ax.plot(gen[0:T], label='PV generation', color='r')
#     # ax.plot(delta[0:T],label='delta')
#     ax2 = ax.twinx()
#     # ax2.plot(tar[0:T],label='tariff')
#     asdas
#     useful_pv = pd.concat([gen, load_full], axis=1).min(axis=1)
#     ax.fill_between(t, useful_pv[0:T],
#                     label='self-consumption', color='#FFB300')

#     ax.grid()
#     ax.legend()
#     ax.set(xlabel='Timesteps', ylabel='kWh',
#            title=' c= {:f} // r={:f}'.format(var_1, var_2))
#     # title= 'Shiftable device solution c='.format

#     # print(var_1)


# %%
def makeplot(T, delta, sol, gen, load, tar, env, var_1, var_2):

    useful_pv = pd.concat([load + sol, gen], axis=1).min(axis=1)
    surplus_pv = ((load+sol)-gen).clip(upper=0)  # negative, saturates at 0

    # Create two subplots, with the first one being 3 times taller than the second
    f, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True,
                                 figsize=(10, 7), gridspec_kw={'height_ratios': [3, 1]})

    # fig, ax1 = plt.subplots(figsize=(10,7))

    t = np.arange(0, T, 1)/4

    # plot the tarrif as backgrounf colors
    tar_vals = list(tar.unique())
    # tar_vals.append(0.0)
    # tar_vals=[0.0,0.2,0.23,0.34,0.56]

    colors = plt.cm.Oranges(np.linspace(0, 1, 100*len(tar_vals)))

    kc = 0
    for k in tar_vals:
        # print(k)
        ax1.fill_between(t, 0, 1, where=(
            tar >= k), color=colors[kc], alpha=0.5, transform=ax1.get_xaxis_transform())
        kc += 10
        # ax1.fill_between(t, 0, 1, where=tar<=k, alpha=0.03, transform=ax1.get_xaxis_transform())

    # os plotN servem depois para a legenda
    plot1, = ax1.plot(t[0:T], sol.values.astype(
        'float64')+load[0:T].values.astype('float64'), label='Total load', color='#1f77b4')

    plot3, = ax1.plot(t, gen[0:T].values.astype(
        'float64'), label='PV generation', color='#FF8b00')

    plot4 = ax1.fill_between(t, useful_pv[0:T].values.astype(
        'float64'), interpolate=False, label='Useful PV', color='#FFB300')

    # plot4=ax1.fill_between(t,gen[0:T].values.astype('float64'),where=(gen[0:T].values.astype('float64') <= load[0:T].values.astype('float64')),interpolate=True,label='gen_x')

    # ax1.fill_between(t,load[0:T].values.astype('float64'),where=(gen[0:T].values.astype('float64') >= load[0:T].values.astype('float64')),interpolate=True,label='gen_x')

    plot5 = ax1.fill_between(t[0:T], surplus_pv[0:T].values.astype(
        'float64'), label='PV surplus', facecolor="none", hatch='..', edgecolor='#FFB300')

    plot2 = ax1.fill_between(t[0:T], sol.values.astype(
        'float64'), facecolor="none", hatch="////", edgecolor='k', alpha=0.4)  # , linewidth=0.0

    ax2.plot(t, tar[0:T], label='tariff', color='k')

    plt.ylim(bottom=0)

    labels = ['Total load', 'Shiftable load', 'PV', 'Useful PV', 'PV surplus']

    ax1.legend([plot1, plot2, plot3, plot4, plot5], labels)

    ax1.set(ylabel='kWh', title=' c= {:f} // r={:f}'.format(var_1, var_2))
    ax2.set(xlabel='Timesteps', ylabel='€/kWh')

    # make grids
    ax1.grid(visible=True, which='major', alpha=0.07)
    ax1.minorticks_on()
    ax1.grid(visible=True, which='minor',
             color='#999999', linestyle='-', alpha=0.07)

    ax2.grid(visible=True, which='major', alpha=0.07)
    ax2.minorticks_on()
    ax2.grid(visible=True, which='minor',
             color='#999999', linestyle='-', alpha=0.07)






# %%
    # #PLots

    # tarifa=full_track['tar_buy']
    # # tarifa=np.random.uniform(low = 0.0, high = 0.5, size = 48)
    # # tarifa=pd.DataFrame(tarifa, columns=['tar_buy'])

    # makeplot(T,metrics_episode['delta_c'],full_track['action']*0.3,full_track['gen0'],full_track['load0'],tarifa,tenv, metrics_episode['cost'].sum(),full_track['reward'].sum()) #


#     fig, ax = plt.subplots()
# x = np.arange(0, 4 * np.pi, 0.01)
# y = np.sin(x)
# ax.plot(x, y, color='black')

# threshold = 0.75
# ax.axhline(threshold, color='green', lw=2, alpha=0.7)
# ax.fill_between(x, 0, 1, where=y > threshold,
#                 color='green', alpha=0.5, transform=ax.get_xaxis_transform())

# #%%

# from matplotlib.patches import Patch

# fix, ax= plt.subplots()
# ax.plot(df['x'], df['y'])

# cmap = matplotlib.cm.get_cmap('Set3')
# for c in df['color'].unique():
#     bounds = df[['x', 'color']].groupby('color').agg(['min', 'max']).loc[c]
#     ax.axvspan(bounds.min(), bounds.max()+1, alpha=0.3, color=cmap.colors[c])

# legend = [Patch(facecolor=cmap.colors[c], label=c) for c in df['color'].unique()]
# ax.legend(handles=legend)
# plt.xticks(df['x'][::2])
# plt.xlim([df['x'].min(), df['x'].max()])
# #DRAWING EXPECTED LINES

# ax.axvline(x=6)
# ax.axvline(x=11)
# ax.axvline(x=16)
# ax.axvline(x=22)
# ax.axvline(x=27)
# plt.show()

# %%
    # PLots
    # makeplot(T,metrics_episode['delta_c'],full_track['action']*0.3,full_track['gen0'],full_track['load0'],full_track['tar_buy'],tenv, metrics_episode['cost'].sum(),full_track['reward'].sum()) #

# T = 2*48
# delta_c = metrics_episode['delta_c']
# shift_load = full_track['action']*0.3
# pv = full_track['gen0']
# base_load = full_track['load0']
# elec_price = full_track['tar_buy']

# # substituí o tenv por [] só para conseguir correr a coisa
# makeplot(T,delta_c,shift_load,pv,base_load,elec_price,[], metrics_episode['cost'].sum(),full_track['reward'].sum())


#     plt.show()
#     time.sleep(0.1)
#     return(ax)


# fig, ax = plt.subplots()
# ax.fill(np.arange(0,T,1), full_track['gen0'], 'b', np.arange(0,T,1), full_track['load0'], 'r', alpha=0.3, zorder=10)
# plt.show()

# g=full_track['gen0'].values.astype('float64')
# l=full_track['load0'].values.astype('float64')


# fig, ax = plt.subplots()
# # ax.fill_between(np.arange(0,T,1), g,'b',l,'r')
# ax.fill_between(t,useful_pv[0:T],label='PV generation',color='#FFB300')
# plt.show()


# import numpy as np
# import matplotlib.pyplot as plt

# # x = np.linspace(0, 8*np.pi, 1000)
# # sinx = np.sin(x)
# # cosx = np.cos(x/2)/2


# t=np.arange(0,T,1)
# g=full_track['gen0'].values.astype('float64')
# l=full_track['load0'].values.astype('float64')

# plt.figure('Fill', facecolor='lightgray')
# plt.title('Fill', fontsize=18)
# plt.grid(linestyle=':')
# plt.plot(t, l, color='dodgerblue',
#          label='load')
# plt.plot(t, g, color='orangered',
#          label='gen')
# plt.fill_between(t, l, g, l<g,
#                  color='dodgerblue', alpha=.3)
# # plt.fill_between(x, sinx, cosx, sinx>cosx,
# #                   color='orangered', alpha=.3)
# plt.legend()
# plt.savefig('fill_between.png')
# plt.show()


# useful_pv = pd.concat([full_track['gen0'],full_track['load0']],axis=1).min(axis=1)


# ... .fill_between(t,useful_pv[0:T],label='PV generation',color='#FFB300')
