#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 16:33:47 2024

@author: omega
"""
import numpy as np
import pandas as pd
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy.random as rnd
import time
import random as rnd
import seaborn as sns
from icecream import ic

class Plots():
    @staticmethod
    def makeplot(T, delta, sol,sol_ag, gen, load, tar, env, var_1, var_2,filename_save):
        
        # ag_colors=pd.DataFrame(['c','r'],index=sol_ag.columns)
        color_list=['c','r',
                    'blue', 'green','magenta', 
                    'yellow', 'purple', 'orange',
                    'pink','brown']

        ag_colors=pd.DataFrame(color_list[0:len(sol_ag.columns)],index=sol_ag.columns)
        ag_colors.columns=['color']

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
        plot6, = ax1.plot(t[0:T], load[0:T].values.astype('float64'), label='Total baseload', color='#db4ddb',alpha=0.6)

        plot3, = ax1.plot(t, gen[0:T].values.astype(
            'float64'), label='PV generation', color='#FF8b00')

        plot4 = ax1.fill_between(t, useful_pv[0:T].values.astype(
            'float64'), interpolate=False, label='Useful PV', color='#FFB300')

        # plot4=ax1.fill_between(t,gen[0:T].values.astype('float64'),where=(gen[0:T].values.astype('float64') <= load[0:T].values.astype('float64')),interpolate=True,label='gen_x')

        # ax1.fill_between(t,load[0:T].values.astype('float64'),where=(gen[0:T].values.astype('float64') >= load[0:T].values.astype('float64')),interpolate=True,label='gen_x')

        plot5 = ax1.fill_between(t[0:T], surplus_pv[0:T].values.astype(
            'float64'), label='PV surplus', facecolor="none", hatch='..', edgecolor='#FFB300')

        

        plot2 = ax1.fill_between(t[0:T], sol.values.astype(
            'float64'), facecolor="none", hatch="////", edgecolor='k', alpha=0.8)
        
        #individual appliance plots
        for ag in sol_ag.columns: 
            ax1.fill_between(t[0:T], sol_ag[ag].values.astype(
                'float64'), facecolor=ag_colors.loc[ag]['color'],edgecolor='k', alpha=0.6)  # ,
        
        # plot2 = ax1.fill_between(t[0:T], sol_ag['shift_ag1'].values.astype(
        #     'float64'), facecolor='c',edgecolor='k', alpha=1)  # , linewidth=0.0
        
        # plot8 = ax1.fill_between(t[0:T], sol_ag['shift_ag2'].values.astype(
        #     'float64'), facecolor='r',edgecolor='k', alpha=1)  # , linewidt8h=0.0
        

         # , linewidth=0.0
        # plot8 = ax1.fill_between(t[0:T],8 sol_ag['shift_ag2'].values.astype(
        #     'float64'), facecolor="none", hatch="////", edgecolor='g', alpha=0.4)  # , linewidth=0.0
        


        ax2.plot(t, tar[0:T], label='tariff', color='k')

        plt.ylim(bottom=0)

        labels = ['Total load', 'Shiftable load', 'PV', 'Useful PV', 'PV surplus','Total baseload']

        ax1.legend([plot1, plot2, plot3, plot4, plot5, plot6], labels)

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
        
        
        if filename_save:
            plt.savefig(filename_save, dpi=300)
            print('saved figure to', filename_save)