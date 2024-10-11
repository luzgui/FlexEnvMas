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
    plt.rcParams['figure.dpi'] = 300
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
        # sol_ag.plot(ax=ax1, kind='bar', stacked=True)
        # for i, ag in enumerate(sol_ag.columns):
        #     color = ag_colors.loc[ag]['color']
        #     ax1.fill_between(np.arange(len(sol))/4, sol_ag[ag] + i * 10, facecolor=color, edgecolor='k', alpha=0.6, label=ag)
        
        for ag in sol_ag.columns: 
            ax1.fill_between(np.arange(len(sol)), sol_ag[ag].values.astype(
                'float64'), facecolor=ag_colors.loc[ag]['color'],edgecolor='k', alpha=0.6)  # ,
            
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
    def makeplot_bar(data, filename_save):
        # import pdb
        # pdb.pdb.set_trace()
        sns.set_style("whitegrid", {"axes.facecolor": ".9"})
        sns.set_context(rc={'patch.linewidth': 0.2, 'patch.linecolor': 'k'})
    
        time = range(0, len(data))
    
        # Main data for bar plot
        load = pd.DataFrame(data['baseload_T'])
        gen = pd.DataFrame(data['gen0_ag1'])
        sol_ag = data[[k for k in data.columns if 'shift_ag' in k and 'coef' not in k]]
    
        cost = data['Cost_shift_T'].sum()
        rewards = data.filter(like='reward').sum().values
        rewards = [round(r, 2) for r in rewards]
        
        
        tariffs=data.filter(like='tar_buy_ag')
    
        width = 1
        edgecolor = "k"
        dodge = False
    
        # Create a figure with two subplots (1 for the bar plot, 1 for the line plot underneath)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [4, 1]})
    
        ### Bar Plot on Top ###
        sns.barplot(x=time, y='gen0_ag1', data=gen, label='PV', color='orange', ax=ax1, width=width, dodge=dodge, alpha=1, edgecolor=edgecolor)
        sns.lineplot(x=time, y='gen0_ag1', data=gen, color='orange', ax=ax1, alpha=0.5)
        sns.barplot(x=time, y='baseload_T', data=load, label='Base Load', color='brown', ax=ax1, width=width, dodge=dodge, edgecolor=edgecolor, alpha=0.8)
    
        # Stack additional bar plots on top of base load
        bottom = load['baseload_T']
        for i, colname in enumerate(sol_ag.columns):
            sns.barplot(x=time, y=colname, data=sol_ag, label=colname, color=sns.color_palette("pastel")[i], ax=ax1, bottom=bottom, width=width, dodge=dodge, edgecolor=edgecolor)
            bottom += sol_ag[colname]
    
        # Title, labels, and grid for the bar plot
        ax1.set_title(f'Community shiftable cost: {round(cost, 2)} (€) / r: {rewards}')
        ax1.set_xlabel('')  # No xlabel for the top plot
        ax1.set_ylabel('kWh')
        ax1.set_xticks(range(0, len(data)+1, 4))  # Tick every 4 time steps
        ax1.set_xticklabels(range(0, 25))
        ax1.grid(visible=True, which='major', alpha=0.3)
        ax1.minorticks_on()
    
        ### Line Plot Underneath ###
        # x_labels for the 100 points in the line plot

    
        # Create a custom colormap for the line plot (green -> red)
        cmap = plt.cm.RdYlGn
    
        # Plot multiple lines on the second axis (ax2)
        # for i in range(tariffs.shape[0]):   
        for i, colname in enumerate(tariffs.columns):

            sns.lineplot(x=time, y=colname, data=tariffs, color='black', ax=ax2, alpha=0.5,marker='o', markersize=2)
            

        ax2.set_xticks(range(0, len(data)+1, 4))  # Tick every 4 time steps
        # ax2.set_xticklabels(range(0, 25))
        ax2.grid(visible=True, which='major', alpha=0.3)
        ax2.minorticks_on()    
        # Set labels and grid for the second plot
        ax2.set_ylabel('€/kWh')
        ax2.set_xlabel('Hour of the day')
        # ax2.grid(True)
        ax1.set_xlim(left=0, right=len(data) - 1)  # Align x limits for ax1
        ax2.set_xlim(left=0, right=len(data) - 1)  # Align x limits for ax2
        # Adjust layout for better readability
        plt.tight_layout()
    
        # Save the figure if filename is provided
        if filename_save:
            plt.savefig(filename_save, dpi=300)
            print('Saved figure to', filename_save)
    
        # Show the plots
        plt.show()



    @staticmethod
    def makeplot_bar_old(data,filename_save):
        sns.set_style("whitegrid", {"axes.facecolor": ".9"})
        sns.set_context(rc = {'patch.linewidth': 0.2,'patch.linecolor':'k'})
        

        time=range(0,len(data))
        
        # sol = data['shift_T']
        load = pd.DataFrame(data['baseload_T'])
        gen = pd.DataFrame(data['gen0_ag1'])
        sol_ag = data[[k for k in data.columns if 'shift_ag' in k and 'coef' not in k]]
        # tar = data['tar_buy']
        cost = data['Cost_shift_T'].sum()
        rewards=data.filter(like='reward').sum().values
        rewards=[round(r,2) for r in rewards]
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
            
            
            ax.set_title(f'Community shiftable cost: {round(cost,2)} (€) / r: {rewards}')
            
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
        
        
    @staticmethod
    def make_jointplot(data,x_var,y_var,filename_save):
        
        """Import file flasg if if should be imported"""
        
        g = sns.JointGrid(data=data, x=x_var, y=y_var, 
                          height=7,
                          marginal_ticks=True)
        
        g.plot_joint(sns.scatterplot, s=50, alpha=0.9)
        g.plot_marginals(sns.histplot,kde=True)
        
        a=0.2
        # g.refline(x=0, color='b', alpha=a, label = 'min cost')
        # g.refline(x=-1,color='g',alpha=a)
        # g.refline(x=1,color='k',alpha=a)
        g.refline(x=1)
        g.refline(y=0)
        
    
        g.set_axis_labels('Daily PV availability compared to appliance energy need (-)',f'Daily Savings Rate (-)')
    
        # g.fig.suptitle(exp_name + 'Total year cost: %1.1f €' % m['cost'].sum() +'  n= %1.1f' % int(len(m)))
        
        
        # text='Daily difference in cost'
        text=f'Daily Savings Rate (computed from {x_var}) '
        g.fig.suptitle(text + '(N = %d' % int(3) + ' / ' + 'days= %d)' % int(len(data)))
        
        g.fig.subplots_adjust(top = 0.9)
        
        # plt.axvline(x = 0, color = 'k',linestyle='--', label = 'min cost', alpha=0.6)
        # plt.axvline(x = 1, color = 'k',linestyle='--', label = 'max cost',alpha=0.6)
        # plt.axvline(x = -1, color = 'k',ls='--', label = 'zero cost',alpha=0.6)
        # plt.axhline(y = 1, color = 'c',linestyle='--', label = 'min_energy',alpha=0.6)
        
        g.ax_joint.grid(True)
        g.ax_joint.grid(True, color='gray', linestyle='--', linewidth=0.3)

        if filename_save: g.savefig(filename_save, dpi=300)
        
    @staticmethod
    def make_multi_jointplot(data,x_var,y_var,filename_save):
        # List of dataframes
        dataframes = data
        x_var = x_var
        y_var = y_var

        # Plot multiple dataframes
        fig, ax = plt.subplots(figsize=(10, 7))

        # Plot multiple dataframes on the main plot
        for ds in dataframes:
            sns.scatterplot(x=x_var, y=y_var, data=dataframes[ds], s=50, alpha=0.9, ax=ax,label=ds)
        
        # Add reference lines
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
        ax.axvline(x=1, color='gray', linestyle='--', linewidth=0.5)
        
        # Set axis labels
        ax.set_xlabel('Daily PV availability compared to appliance energy need (-)')
        ax.set_ylabel('%')
        
        # Set title
        text = 'Increase in RL daily cost compared to optimal ((C{RL}+1-C*+1)/C*+1)  '
        # fig.suptitle(text + '(N = %d' % len(dataframes) + ' / ' + 'days= %d)' % sum(len(df) for df in dataframes))
        fig.suptitle(text)
        
        # Add grid lines
        ax.grid(True)
        ax.grid(True, color='gray', linestyle='--', linewidth=0.3)
        ax.legend()
        # Show the plot
        
        # Show the plot
        if filename_save:
            plt.savefig(filename_save, dpi=300)
        plt.show()
           
    def make_compare_plot(data,filename_save):
        """it accepts metrics dataframe and plots the cost of each agent against every other agent
        
        """
        
        # List of variables
        variables = data

        # Determine the number of variables
        num_vars = len(variables)

        # Calculate the number of rows and columns for subplots
        num_rows = num_vars
        num_cols = num_vars

        # Create subplots
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(5*num_cols, 5*num_rows))

        # Flatten axes if there's only one row
        if num_rows == 1:
            axes = [axes]

        # Create scatter plots for each pair of variables
        for i in range(num_vars):
            for j in range(num_vars):
                # if i != j:
                    sns.scatterplot(x=variables[j], y=variables[i], ax=axes[i][j])
                    axes[i][j].set_xlabel(f'Variable {j+1}')
                    axes[i][j].set_ylabel(f'Variable {i+1}')
                    axes[i][j].set_title(f'Variable {j+1} vs Variable {i+1}')
                    min_val = min(variables[i] + variables[j])
                    max_val = max(variables[i] + variables[j])
                    axes[i][j].plot([min_val, max_val], [min_val, max_val], color='r', linestyle='--')

                    
        # Adjust layout
        plt.tight_layout()
        
        if filename_save: plt.savefig(filename_save, dpi=300)
        # Show plots
        plt.show()
        

    @staticmethod
    def make_compare_plot2(data, filename_save=None):
        """
        Accepts a DataFrame and plots the scatter plot of the values in each pair of columns, 
        using the column names as labels.
        
        Parameters:
            data (DataFrame): The input DataFrame containing the data.
            filename_save (str): Optional. File name to save the plot.
        """
        # List of variables
        variables = data.columns
    
        # Determine the number of variables
        num_vars = len(variables)
    
        # Calculate the number of rows and columns for subplots
        num_rows = num_vars
        num_cols = num_vars
    
        # Create subplots
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(5*num_cols, 5*num_rows))
    
        # Flatten axes if there's only one row
        if num_rows == 1:
            axes = [axes]
    
        # Create scatter plots for each pair of variables
        for i in range(num_vars):
            for j in range(num_vars):
                # Skip plotting if same variable
                # if i == j:
                #     axes[i][j].axis('off')
                #     continue
                # import pdb
                # pdb.pdb.set_trace()
                sns.scatterplot(data=data, x=variables[j], y=variables[i], ax=axes[i][j])
                # axes[i][j].set_xlabel(variables[j])
                # axes[i][j].set_ylabel(variables[i])
                # axes[i][j].set_title(f'{variables[j]} vs {variables[i]}')
                # # Add diagonal line
                min_val = min(data[variables[i]].min(), data[variables[j]].min())
                max_val = max(data[variables[i]].max(), data[variables[j]].max())
                axes[i][j].plot([min_val, max_val], [min_val, max_val], color='r', linestyle='--')
                axes[i][j].xaxis.set_major_locator(plt.MaxNLocator(nbins=5))
                axes[i][j].yaxis.set_major_locator(plt.MaxNLocator(nbins=5))
    
        # Adjust layout
        plt.tight_layout()
        
        if filename_save:
            plt.savefig(filename_save, dpi=300)
        # Show plots
        plt.show()
        
    @staticmethod
    def make_multi_histogram(data, filename_save):
        fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)

        for ax,obj_id in zip(axes,data):
            
            df=data[obj_id]
            max_val=df.max().max()
            min_val=df.min().min()
            
            val_width = max_val - min_val
            n_bins = 10
            bin_width = val_width/n_bins
            # import pdb
            # pdb.pdb.set_trace()
            
            hist=sns.histplot(
                data=df,
                multiple='dodge',
                ax=ax,
                bins=n_bins,
                binrange=(min_val, max_val),
                edgecolor='.3',
                linewidth=.5,
                stat='count',
                shrink=.7,
            )
            


            
            ticks = np.arange(0, 0.14, bin_width)
            ax.set_xticks(ticks)
            ax.set_xticklabels([f'{tick:.2f}' for tick in ticks])
        
            # Add title and legends
            print('vla')
            ax.set_title(obj_id, fontsize=15)
            ax.set_xlabel('€/kWh (daily cost per appliance load)', fontsize=12)
            # plt.ylabel('Num of Days', fontsize=12)
        fig.suptitle('Per agent daily costs in the testing comunity')
        
        # Show the plot
        if filename_save:
            plt.savefig(filename_save, dpi=300)
        
        plt.show()
                
        
    @staticmethod    
    def plotline_smooth(data,filename_save):
        "data must be a dataframe with original signal and smoothed signal"
        plt.figure(figsize=(10, 6))
        palette=sns.color_palette()
        i=0
        for obj_id in data:
            # sns.lineplot(data=data[obj_id], label=obj_id)
            sns.lineplot(data=data[obj_id], x=data[obj_id].index, y='episode_reward_mean', label=obj_id, color=palette[i])
            i+=1
            # sns.lineplot(data=df, x=df.index, y='moving_average', label='Moving Average', color='red')
        
        # Add title and labels
        plt.title('Community Mean Reward')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.grid(True)
        plt.legend()
        
        if filename_save:
            plt.savefig(filename_save, dpi=300)
        
        # Show the plot
        plt.show()
        
    @staticmethod   
    def plot_barplot(df,filename_save):
        palette = sns.color_palette("RdYlGn_r", len(df))
        sns.barplot(data=df,palette=palette)
        plt.xlabel('Model')
        plt.ylabel('€')
        plt.title('Yearlly mean cost')
        
        # Show the plot
        if filename_save:
            plt.savefig(filename_save, dpi=300)
    @staticmethod        
    def plot_multi_bar(df,filename_save):
        plt.figure(figsize=(8, 6))
        sns.barplot(data=df,x='Category', y='Value', hue='Label', dodge=True)
        plt.xlabel('Model')
        plt.ylabel('mean yearlly value (€/kWh)')
        plt.title('Yearly mean cost per agent')
        # plt.show()
        
        if filename_save:
            plt.savefig(filename_save, dpi=300)
    
    @staticmethod
    def plot_tarifs_lines(data):
        df = data.filter(like='tar_buy_ag')
        
        # Plot settings
        title = "Tariffs"
        xlabel = "Timestep"
        ylabel = "€/kWh"
        legend = True
        
        # Set Seaborn style
        sns.set(style="whitegrid")
        
        # Create a new figure
        plt.figure(figsize=(10, 6))
        
        # Plot each column in the DataFrame as a separate line
        for column in df.columns:
            sns.lineplot(x=df.index, y=df[column], label=column)
        
        # Add labels and title
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        
        # Enable grid
        plt.grid(True)
        
        # Adjust x-axis ticks to be more readable (rotate by 45 degrees)
        plt.xticks(rotation=45)
        
        # Automatically set the number of x-ticks to make them more readable (optional)
        plt.locator_params(axis='x', nbins=12)  # Reduce the number of ticks to 12
        
        # Show the p

        
        
        
