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
import seaborn as sns

class Plots():
    @staticmethod
    def makeplot(T, delta, sol,sol_ag, gen, load, tar, env, var_1, var_2,filename_save):
        """
        This plots env_state data
        """
        
        
        
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
            
            
            

    @staticmethod

    
    def plot_energy_usage(data, filename_save=None):
        # Unpack data
        sol = data['shift_T']
        load = data['baseload_T']
        gen = data['gen0_ag1']
        sol_ag = data[[k for k in data.columns if 'shift_ag' in k and 'coef' not in k]]
        tar = data['tar_buy']
        var_1 = data['Cost_shift_T'].sum()
        var_2 = 0
        
        # Define color scheme
        color_list=['c','r','blue','pink', 'green','magenta', 
                    'yellow', 'purple', 'orange','brown']
        ag_colors = pd.DataFrame(color_list[:len(sol_ag.columns)], index=sol_ag.columns, columns=['color'])
        
        # Calculate useful PV and surplus PV
        useful_pv = pd.concat([load + sol, gen], axis=1).min(axis=1)
        surplus_pv = ((load + sol) - gen).clip(upper=0)  # Negative values saturate at 0
        
        # Create subplots
        # f, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(10, 7), gridspec_kw={'height_ratios': [3, 1]})
        # f, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(10, 7), gridspec_kw={'height_ratios': [3, 1]})
        fig, ax1 = plt.subplots(figsize=(10, 7))
        
        # Plot tariff background colors
        tar_vals = tar.unique()
        colors = plt.cm.Oranges(np.linspace(0, 1, 100 * len(tar_vals)))
        kc = 0
        for k in tar_vals:
            ax1.fill_between(np.arange(len(tar))/4, 0, 1, where=(tar >= k), color=colors[kc], alpha=0.5, transform=ax1.get_xaxis_transform())
            kc += 10
        
        # Plot main components
        # plot1, = ax1.plot(np.arange(len(sol))/4, sol.values + load.values, label='Total load', color='#1f77b4', linewidth=2.0)
        # plot6, = ax1.plot(np.arange(len(sol))/4, load.values, label='Total baseload', color='#db4ddb', alpha=0.6, linewidth=2.0)
        # plot3, = ax1.plot(np.arange(len(sol))/4, gen.values, label='PV generation', color='#FF8b00', linewidth=2.0)
        # plot4 = ax1.fill_between(np.arange(len(sol))/4, useful_pv.values, interpolate=False, label='Useful PV', color='#FFB300')
        # plot5 = ax1.fill_between(np.arange(len(sol))/4, surplus_pv.values, label='PV surplus', facecolor="none", hatch='..', edgecolor='#FFB300')
        
        plot1, = ax1.plot(np.arange(len(sol)), sol.values + load.values, label='Total load', color='#1f77b4', linewidth=2.0)
        plot6, = ax1.plot(np.arange(len(sol)), load.values, label='Total baseload', color='#db4ddb', alpha=0.6, linewidth=2.0)
        plot3, = ax1.plot(np.arange(len(sol)), gen.values, label='PV generation', color='#FF8b00', linewidth=2.0)
        plot4 = ax1.fill_between(np.arange(len(sol)), useful_pv.values, interpolate=False, label='Useful PV', color='#FFB300')
        plot5 = ax1.fill_between(np.arange(len(sol)), surplus_pv.values, label='PV surplus', facecolor="none", hatch='..', edgecolor='#FFB300')
        
        

        # plot2 = ax1.fill_between(np.arange(len(sol))/4, sol.values, facecolor="none", hatch="////", edgecolor='k', alpha=0.8)
        
        # Plot individual appliance loads
        # bottom = np.zeros(len(sol))
        # for ag in sol_ag.columns:
        #     if ag == 'shift_ag3':
        #         continue  # Skip the colum
        #     if ag == 'shift_ag6':
        #         continue  # Skip the colum
        #     else:
        #         color = ag_colors.loc[ag]['color']
        #         # ax1.fill_between(np.arange(len(sol))/4, sol_ag[ag], facecolor=color, edgecolor='k', alpha=0.6, label=ag)
        #         ax1.bar(np.arange(len(sol))/4,  sol_ag[ag], bottom=bottom, color=color, label=ag, alpha=0.8)
        #         bottom += sol_ag[ag]
        # import pdb
        # pdb.pdb.set_trace()
        sol_ag.plot(ax=ax1, kind='bar', stacked=True)
        # for i, ag in enumerate(sol_ag.columns):
        #     color = ag_colors.loc[ag]['color']
        #     ax1.fill_between(np.arange(len(sol))/4, sol_ag[ag] + i * 10, facecolor=color, edgecolor='k', alpha=0.6, label=ag)
            
        # stacked_load = np.zeros(len(sol))
        # for ag in sol_ag.columns:
        #     color = ag_colors.loc[ag]['color']
        #     stacked_load += sol_ag[ag]
        #     ax1.fill_between(np.arange(len(sol))/4, stacked_load, facecolor=color, edgecolor='k', alpha=0.6, label=ag)
        
        # stacked_load = np.zeros(len(sol))
        # for ag in sol_ag.columns[::-1]:  # Reverse the order of iteration
        #     color = ag_colors.loc[ag]['color']
        #     stacked_load += sol_ag[ag]
        #     ax1.fill_between(np.arange(len(sol))/4, stacked_load, facecolor=color, edgecolor='k', alpha=0.2, label=ag)


        # Plot tariff
        
        
        # Legend and labels
        # labels = ['Total load', 'Shiftable load', 'PV', 'Useful PV', 'PV surplus', 'Baseload']
        # ax1.legend([plot1, plot2, plot3, plot4, plot5, plot6], labels, loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5, -0.15))
        
        labels = ['Total load', 'PV', 'Useful PV', 'PV surplus', 'Baseload']
        ax1.legend([plot1, plot3, plot4, plot5, plot6], labels, loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5, -0.15))
        
        
        # ax2.plot(np.arange(len(tar))/4, tar, label='Tariff', color='k', linewidth=2.0)
        
        ax1.set_ylabel('kWh')
        ax1.set_title('c={:f} // r={:f}'.format(var_1, var_2))
        # ax2.set_xlabel('Timesteps')
        # ax2.set_ylabel('€/kWh')
        
        # Gridlines
        ax1.grid(visible=True, which='major', alpha=0.3)
        ax1.minorticks_on()
        ax1.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.15)
        # ax2.grid(visible=True, which='major', alpha=0.3)
        # ax2.minorticks_on()
        # ax2.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.15)
        
        # Save plot if filename is provided
        if filename_save:
            plt.savefig(filename_save, dpi=300)
            print('Saved figure to', filename_save)

    @staticmethod
    def makeplot_bar(data,filename_save):
        sns.set_style("whitegrid", {"axes.facecolor": ".9"})
        sns.set_context(rc = {'patch.linewidth': 0.2,'patch.linecolor':'k'})
        

        time=range(0,len(data))
        
        
        # sol = data['shift_T']
        load = pd.DataFrame(data['baseload_T'])
        gen = pd.DataFrame(data['gen0_ag1'])
        sol_ag = data[[k for k in data.columns if 'shift_ag' in k and 'coef' not in k]]
        tar = data['tar_buy']
        cost = data['Cost_shift_T'].sum()
        var_2 = 0

        width=1
        edgecolor = "k"
        dodge=False
        
        fig, ax = plt.subplots(figsize=(10, 7))
        # fig, ax = plt.subplots()
        sns.barplot(x=time, y='gen0_ag1', data=gen, label='PV', color='orange', ax=ax,width=width,dodge=dodge, alpha=1,edgecolor=edgecolor)
        sns.lineplot(x=time, y='gen0_ag1', data=gen, color='orange', ax=ax, alpha=0.5)
        sns.barplot(x=time, y='baseload_T', data=load, label='Base Load', color='brown', ax=ax,width=width,dodge=dodge,edgecolor=edgecolor, alpha=0.8)

        # saturation = 0.75

        # Iterate over each column and plot the bars
        bottom=load['baseload_T']
        for i, colname in enumerate(sol_ag.columns):
            
            sns.barplot(x=time, y=colname, data=sol_ag, label=colname, color=sns.color_palette("pastel")[i], ax=ax, bottom=bottom,width=width,dodge=dodge,edgecolor=edgecolor)
            bottom += sol_ag[colname]
            
            ax.set_title(f'Community shiftable cost: {cost} (€)')
            
            ax.set_xlabel('hours of the day')
            ax.set_ylabel('kWh')
            
            ax.set_xticks(range(0, len(data)+1, 4))  #  every hour ticks
            ax.set_xticklabels(range(0, 25))


            
            # Gridlines
            ax.grid(visible=True, which='major', alpha=0.3)
            ax.minorticks_on()
            # ax.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.15)
        
        if filename_save:
            plt.savefig(filename_save, dpi=300)
            print('saved figure to', filename_save)
        
        
    

    
            