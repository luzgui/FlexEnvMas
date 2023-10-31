#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 11:46:27 2022

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
def makeplot(T, delta, sol,sol_ag, gen, load, tar, env, var_1, var_2):
    
    ag_colors=pd.DataFrame(['c','r'],index=sol_ag.columns)
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
    
    # plt.savefig('fig1.png', dpi=300)
    

# %% Community Aggregated plots
def make_boxplot(metrics,env):
    
    m=metrics.loc['com']#only using community metrics in plots
    
    
    min_cost=env.tar_buy*env.E_prof['E_prof'].sum()

    #boxplot
     
    # fig, ax = plt.subplots(figsize =(10, 7))
     
    # Creating plot
    # ax.boxplot(m[['cost', 'selfsuf','x_sig']])
    # ax.set(title='Number of days=%i' % len(m))
    # ax.grid(True, which='major')
    
    # ax.set_xticklabels(['cost', 'selfsuf','x_sig'])
    
    # # show plot
    # plt.show()
    
    # scatter plot
    
    # fig1, ax1 = plt.subplots(figsize =(10, 7))
    
    fig, ax = plt.subplots(2,1,figsize =(10, 7))
    
    
    
    ax[0].scatter(m['cost_var'],m['x_ratio'])
    ax[0].set(title='Cost vs Available PV excess for the community | Number of days=%i' % len(m))
    
    ax[0].set_xlabel('%')
    ax[0].set_ylabel('PV Excess to app energy needed ratio')
    
    #plot the min cost for comparison
    ax[0].axvline(x = 0, color = 'r', label = 'min cost')
    ax[0].axvline(x = 1, color = 'k', label = 'max cost')
    ax[0].axvline(x = -1, color = 'g', label = 'zero cost')
    ax[0].axhline(y = 1, color = 'c',linestyle='--', label = 'min_energy')
    ax[0].grid('minor')
    
    ax[0].legend()
    
    
    
    ## Histogram

    ax[1].hist(m['cost_var'], bins=40)
    ax[1].set_ylabel('Number of days in year')
    
    
    ax[1].axvline(x = 1, color = 'k')
    ax[1].axvline(x = 0, color = 'r')
    ax[1].axvline(x = -1, color = 'g')
    ax[1].grid('minor')
    
    fig.tight_layout()
    plt.show()



def make_costplot(df,filename_save,filename_import, save_fig):
    """Import file flasg if if should be imported"""
    
    if filename_import:
        m=pd.read_csv(filename_import,index_col=0)
        m=m.loc['com']
        filename_import=Path(filename_import)
        filename_save=Path(os.path.join(filename_import.parent, 'cost_plot_'+ filename_import.stem))
        filename_save=filename_save.with_suffix("." + 'png')
        exp_name=filename_import.stem
        print('imported data from ', filename_import)
        pass

    else:
        print('imported data from metrics dataframe')
        m=df.loc['com']#only using community metrics in plots
        exp_name='cenas'
        
    
    # if type(filename)==str:
    #     exp_name=filename
    # else:    
    #     exp_name=filename.name
    #     exp_name=exp_name.replace('plot-metrics-',' ')
    #     exp_name=exp_name.replace('.png',' ')
    #     print(exp_name)


    # sns.set_theme()
    
    # g=sns.jointplot(data=m, x="cost_var", y="x_ratio", hue='season',
    #                 ylim=[-1,m['x_ratio'].max()+1],
    #                 height=7)
    
    # g=sns.jointplot(data=m, x="cost_var", y="x_ratio",
    #                 ylim=[-1,m['x_ratio'].max()+1],
    #                 height=7)
    
    
    # fig, axs = plt.subplots(2, 1, figsize=(6, 8))
    
    g = sns.JointGrid(data=m, x="cost_var", y="x_ratio", 
                      height=7,
                      marginal_ticks=True)
    
    # g.plot(sns.scatterplot, sns.histplot)

    
    g.plot_joint(sns.scatterplot, s=50, alpha=0.9)
    g.plot_marginals(sns.histplot)
    
    
    a=0.5
    g.refline(x=0, color='b', alpha=a, label = 'min cost')
    g.refline(x=-1,color='g',alpha=a)
    g.refline(x=1,color='r',alpha=a)
    g.refline(y=1)

    
    g.set_axis_labels('% relative to minimum cost', 
                      'PV Excess to app energy needed ratio')

    g.fig.suptitle(exp_name + 'Total year cost: %1.1f €' % m['cost'].sum() +'  n= %1.1f' % int(len(m)) )
    g.fig.subplots_adjust(top = 0.9)

    # plt.axvline(x = 0, color = 'k',linestyle='--', label = 'min cost', alpha=0.6)
    # plt.axvline(x = 1, color = 'k',linestyle='--', label = 'max cost',alpha=0.6)
    # plt.axvline(x = -1, color = 'k',ls='--', label = 'zero cost',alpha=0.6)
    # plt.axhline(y = 1, color = 'c',linestyle='--', label = 'min_energy',alpha=0.6)
    
    # g.ax_marg_y.remove()

    if save_fig: g.savefig(filename_save, dpi=300)
    
# make_boxplot(metrics,tenv)
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
