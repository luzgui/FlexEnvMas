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

from ray.tune import analysis, Tuner, result_grid
from ray.tune import ExperimentAnalysis, ResultGrid
from pathlib import Path, PurePath

from ray import train
from ray.train import RunConfig, CheckpointConfig, Checkpoint
from trainable import trainable_mas
from icecream import ic


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
    

def make_env_data_mas(data,t_init,t_end, load_id, pv_factor, num_agents, agents_id, cluster):
    "(data: timeseries, load_id: house number, pv_factor, num_agents, agents_id (list string)"
    

    #to extend to n>2 agenst just have to edit this function to create n groups of m elements
    def divide_into_groups(lst, n):
        """Divides lst elements into groups of n elements"""
        return [lst[i:i+n] for i in range(0, len(lst), n)]
    
    # we are gonna transform directly the data dataframe in order to  
    if len(cluster)!=0:
        n=2 #groups of 2
        groups=divide_into_groups(cluster, n)
        
        df_stack=pd.DataFrame()
        df_stack['minutes']=pd.concat([data['minutes']]*n, axis=0)
        
        df_stack=df_stack.reset_index(drop=True)
        
        i=1
        for g in groups:    
            pv_cluster=['PV'+str(k) for k in g]
            cluster_agents=['ag'+str(k) for k in g]
            
            data_group=data[['minutes']+cluster_agents+pv_cluster]
            
            
            for e in [pv_cluster,cluster_agents]:    
                stack=pd.concat([data_group[col] for col in e], axis=0)
                name=''.join(e)[0:2]+str(i)
                df_stack[name]=stack.values
            
            i+=1
        
        data=df_stack
    
    
    #if no cluster it starts here
    pv_id=[k.replace('ag','PV') for k in load_id]
    
    df=pd.DataFrame()
    
    # load_names=['load_ag'+str(k) for k in range(num_agents)]
    
    df['minutes']=data.iloc[t_init:t_end]['minutes']
    df[load_id]=data.iloc[t_init:t_end][load_id]
    
    df[pv_id]=data.iloc[t_init:t_end][pv_id]
    
    # df['gen']=pv_factor*abs(data.iloc[t_init:t_end][pv_id])
    df['gen']=df[pv_id].sum(axis=1)*pv_factor
    
    
    # delta and excess are COLLECTIVE, i.e computed based on aggregated quantities
    df['delta']=df[load_id].sum(axis=1)-df['gen']
    df['excess']=[max(0,-df['delta'][k]) for k in range(t_init,t_end)] 
    
    frames=[]
    for lid,aid in zip(load_id,agents_id):
        df_temp=df.copy()[['minutes',lid,'gen','delta','excess']]
        label=[aid]*len(df_temp)
        # df_temp['agent_id']=label
        df_temp.rename(columns={lid:'load'}, inplace=True)
        
        frames.append(df_temp)    
        
    df_final=pd.concat(frames, keys=agents_id)
    
    
    return df_final




def get_raw_data(file, datafolder,unit):
    
    dt=15
    
    cons_data=pd.read_excel(datafolder + '/' + file, 'Total Consumers')
    cons_data.columns=['ag'+str(k) for k in range(len(cons_data.columns))]
    
    prod_data=pd.read_excel(datafolder + '/' + file, 'Total Producers')
    prod_data.columns=['PV'+str(k) for k in range(len(prod_data.columns))]
    
    
    #create a vector of minutes
    mins=pd.DataFrame(np.tile(np.linspace(0,1440-dt,num=int((24*60/dt))),366), columns=['minutes'])
    
    
    if unit=='kwh':
        dh=dt*(1/60.0)#kw to kwh convertion factor
        cons_data=pd.concat([mins,dh*cons_data,dh*prod_data],axis=1) #kwh
    elif unit=='kw':
        cons_data=pd.concat([mins,cons_data,prod_data],axis=1) #original dataset unit kw

    return cons_data

    

def make_env_data(data,timesteps, load_id, pv_factor):
    "(data: timeseries, laod_num: house number, pv_factor"
    
    df=pd.DataFrame()
    
    df['minutes']=data.iloc[0:timesteps]['minutes']
    df['load']=data.iloc[0:timesteps][load_id]
    df['gen']=pv_factor*abs(data.iloc[0:timesteps]['PV'])
    df['delta']=df['load']-df['gen']
    df['excess']=[max(0,-df['delta'][k]) for k in range(timesteps)] 
    
    # gen=np.zeros(timesteps)
    # minutes=make_minutes(data,timesteps) # make minutes vector

    return df


def self_suf(env,action):
    # if var > 0 and env.gen0 != 0:
        # g=var
    # elif var <= 0:
        # g=env.load0
    if env.gen0 != 0:
        # g=min(env.load0+action*env.profile[0],env.gen0) #for the total load 
        g=min(action*env.profile[0],env.excess0) #self sufficency just for the load
    elif env.gen0 ==0:
        g=0
    
    return g


def get_checkpoint(log_dir,exp_name,metric,mode):
    #Recover the tune object from the dir
    # The trainable must be initialized # reuslts must be stored in the same analysis object
    experiment_path=os.path.join(log_dir, exp_name)
    analysis_object = ExperimentAnalysis(experiment_path,
                                         default_metric=metric, 
                                         default_mode=mode)
    
    
    results=ResultGrid(analysis_object)
                       
                 
    # best_result=results.get_best_result(metric=metric,mode=mode)
    # best_checkpoint=best_result.checkpoint
    df=results.get_dataframe()
    
    best_trial=analysis_object.get_best_trial(metric=metric, mode=mode)  
    best_checkpoint=analysis_object.get_best_checkpoint(best_trial)

    
    # with checkpoint_x.as_directory() as checkpoint_dir:
    #     analysis_object = ExperimentAnalysis(checkpoint_dir,
    #                                          default_metric=metric, 
    #                                          default_mode=mode)
    
    # best_path=experiment_path + '/trainable_mas_shiftenv_16d1b_00000_0_2023-10-09_22-30-38/checkpoint_000003'
    

    
    # train.get_checkpoint(args, kwargs)
    #get the best trial checkpoint
    # local_trial=analysis_object.get_best_trial()
    #identify the dir where is the best checkpoint according to metric and mode
    
    # checkpoint=analysis_object.get_best_checkpoint(best_trial)
    # checkpoint.path=best_path

    # convert the windows path to local linux path
    # c_path=Path(checkpoint.path)   
    # l_path=Path(c_path.name).as_posix()
    # split=l_path.split('\\')
    
    # checkpoint.path=os.path.join(experiment_path,split[-2],split[-1])

    #dataframes
    # df=analysis_object.dataframe(metric,mode) #get de dataframe results
    # best_df=analysis_object.best_result    
    print(mode,best_checkpoint)
    #recover best agent for te
    
    return best_checkpoint, df, best_trial
    

def get_post_data(menv):
    """This function produces dataframes for analysing results from the environment itself, i.e, the environment stores all the state-action history and from there we can recover and post-process data"""
    
    df=pd.DataFrame()
    df_temp=pd.DataFrame()
    T=menv.Tw
    for aid in menv.agents_id:
        #we need to take out the last observation because its allways one timestep ahead
        state_hist=menv.state_hist.loc[aid][0:T] 
        action_hist=menv.action_hist.loc[aid]    
        reward_hist=menv.reward_hist.loc[aid]
   
        df=pd.concat([df,
                       pd.concat([state_hist,action_hist, reward_hist],
                    axis=1)])
    
        
    for aid in menv.agents_id:  
        # df=pd.concat([state_hist,action_hist, reward_hist],axis=1)
        df.loc[aid,'shift']=df.loc[aid]['action']*menv.profile[aid][0]
        df.loc[aid,'shift_base']=df.loc[aid]['action']*menv.profile[aid][0]+df.loc[aid]['load0'] #for each agent the sum of its shiftable load with its base load
        
    df['shift'].name='shiftable load for each agent'
    
    #Sum all variables over the agents for each timestep
    df_group=df.groupby('tstep').sum() # In this dataframe al variables are summed 
    
    for aid in menv.agents_id: 
        df.loc[aid,'shift_T']=df_group['shift'].values #sum of all shiftable loads for all agents
        df.loc[aid,'shift_base_T']=df_group[['shift','load0']].sum(axis=1).values # sum of all shiftable loads and base loads for all agents
        df.loc[aid,'baseload_T']=df_group['load0'].values
        
        
        df.loc[aid,'coef_shift']=df.loc[aid]['shift']/df.loc[aid]['shift_T'] #sharing/load coeficient considering only the shiftable loads
        df.loc[aid,'coef_shift']=df.loc[aid,'coef_shift'].fillna(0)
        
        df.loc[aid,'coef_shift_base']=df.loc[aid]['shift_base']/df.loc[aid]['shift_base_T'] #shating/load coefficient considering shiftable and base load
        df.loc[aid,'coef_shift_base']=df.loc[aid,'coef_shift_base'].fillna(0)
        
        #Real individual cost for each agent AFFECTED by the sharingf coefficient, i.e, we assume that excess is shared according to the load level of each agent
        df.loc[aid,'r_cost']=(df.loc[aid]['shift']-df.loc[aid,'coef_shift']*df.loc[aid,'excess0'])*df.loc[aid,'tar_buy']
        
        df.loc[aid,'r_cost_pos']=df.loc[aid,'r_cost'].clip(lower=0)
        
    
    ### transform the results in order to show them based on the same timesteps 
    df_list=[]
    for aid in menv.agents_id: 
        df_ag=df.loc[aid]
        new_col_names={col_name:col_name+'_'+aid for col_name in df_ag.columns}
        df_ag=df_ag.rename(columns=new_col_names)
        df_ag=df_ag.rename(columns={'tstep_'+aid:'tstep'})
        df_ag=df_ag.set_index('tstep')
        df_list.append(df_ag)
        
    df_select=pd.concat(df_list, axis=1)
    

    #Now we selct only the variables we wanna see
    
    vars_names=['minutes',
                'gen0',
                'load0',
                'excess0',
                'tar_buy',
                'action',
                'reward',
                'shift',
                'cost',
                'baseload_T']
    
    cols_names=[]
    for k in df_select.columns:
        for j in vars_names:
            if j in k:
                cols_names.append(k)
                
    
    #
    df_post=df_select[cols_names] #selct the variables that are relevant
    df_post=df_post.drop(columns=[k for k in df_post.columns if 'tar_buy0' in k]) #we dont need tar_buy0
    
    #reduce common features and filter out the common variables              
    vars_names_filter=['minutes','shift_T','shift_base_T','excess','load0','tar_buy','baseload_T']
    df_post=compare_equals(vars_names_filter, df_post, drop=False) 
    
    
    #compute the Total cost
    df_post_costs=df_post[[k for k in df_post.columns if 'cost_pos' in k]]
    df_post['Cost_shift_T']=df_post_costs.sum(axis=1)

    # compute total reward
    df_post_rewards=df_post[[k for k in df_post.columns if 'reward' in k]] 
    
    if menv.env_config['mas_setup']=='cooperative_colective':#reward is cooperative what means that all agents get the same reward
        print('reward is common for all agents')
        df_post=compare_equals(['reward'], df_post, drop=False)
    
    
        
        

                        
   
        
        
    
    
    # # cols_names=[k for k in vars_names if k in df_select.columns]
    
    # columns_names=[]

    # shift_columns_names=['shift_'+aid for aid in menv.agents_id]
    # reward_columns_names=['reward_'+aid for aid in menv.agents_id]
    # load_columns_names=['load_'+aid for aid in menv.agents_id]
    # coef_columns_names=['coef_'+aid for aid in menv.agents_id]
    # real_cost_columns_names=['real_cost_'+aid for aid in menv.agents_id]
    
   
    # columns_names.extend(load_columns_names)
    # columns_names.extend(shift_columns_names)
    # columns_names.extend(reward_columns_names)
    # columns_names.extend(coef_columns_names)
    # columns_names.extend(real_cost_columns_names)
    
    # columns_names.extend(['shift_T','load_T','gen0','excess0','reward_T','Cost_shift_T','tar_buy'])
    
    # #make a new dataframe to store the solutions
    # df_post=pd.DataFrame(columns=columns_names)
    
    # for aid in menv.agents_id:
    #     var_ag=[v for v in df_post.columns if aid in v]
        
    #     for var in var_ag:
    #         if 'load' in var:
    #             df_post[var]=df.loc[aid,'load0'].values
            
    #         if 'shift' in var:
    #             df_post[var]=df.loc[aid,'action'].values*menv.profile[aid][0]
                
    #         if 'reward' in var:
    #             df_post[var]=df.loc[aid,'reward'].values
                

                
    
    
    # df_post['shift_T']=df_post[shift_columns_names].sum(axis=1)
    # # df_post['']
    
    # #we have to perfomr again this cicle
    # for aid in menv.agents_id:
    #     var_ag=[v for v in df_post.columns if aid in v]
    #     for var in var_ag:
    #             if 'coef' in var:
    #                 df_post[var]=df.loc[aid,'action'].values*menv.profile[aid][0]/df_post['shift_T']
    #                 df_post[var]=df_post[var].fillna(0)  #substitute all nans for zeros 
                
               
    
    # df_post['load_T']=df_post[load_columns_names].sum(axis=1)
    # df_post['reward_T']=df_post[reward_columns_names].sum(axis=1)
    
    
    # df_post['gen0']=df.loc[menv.agents_id[0],'gen0'].values #pv production is the same for all and it is collective. So we can use any agent on the agents_id list
    # df_post['excess0']=df.loc[menv.agents_id[0],'excess0'].values # the excess is the same for all
    
    
    # df_post['tar_buy']=df.loc[menv.agents_id[0],'tar_buy'].values
    
    
    # #computing cost of ONLY the shiftable loads with excess
    # df_temp=df_post['shift_T']-df_post['excess0']
    # df_post['Cost_shift_T']=np.maximum(df_temp,0)*df_post['tar_buy']
    
    
    

        
    return df, df_post



def compare_equals(vars_names,df, drop=True):
    """We are loping over the vars_names_filter and check if the columns for each agent are equal. if they are we are droping them and subsituting by only a column with the common variable its applicable to minutes, total shiftable loads and total bse load"""
    
    df=df.copy()
    for n in vars_names:
        names=[k for k in df.columns if n in k]
        vals_var=[]
        for col0 in names:
            for col1 in [na for na in names if na != col0]:
                # ic(col0)
                # ic(col1)
                # ic(df[col0]==df[col1])
                val=df[col0]==df[col1]
                # ic(all(val))
                vals_var.append(all(val))
                if all(vals_var):
                        df_temp=df[names[0]]
                        df.loc[:,n]=df_temp # bug
                        if drop:
                            df=df.drop(columns=names)
                        # ic(col0)    
                        names.remove(col1)
    return df

#%% Functions for MAS environment

def policy_mapping_fn(agent_id,episode, worker, **kwargs):
    'Policy mapping function'
    return 'pol_' + agent_id

def policy_mapping_fn_shared(agent_id, episode, worker, **kwargs):
    'Policy mapping function with shared policy'
    return 'shared_pol' # parameter sharing must return the same policy for any agent

def policy_mapping_fn_test(agent_id):
    'Policy mapping function'
    return 'pol_' + agent_id

def policy_mapping_fn_test_shared(agent_id):
    'Policy mapping function with shared policy'
    return 'shared_pol' # p


def get_actions(obs,trainer,agents_id, map_func):
    'resturns the actions of the agents'
    if type(obs)==dict:
        actions = {aid:trainer.compute_single_action(obs[aid],
                                                     policy_id=map_func(aid)) for aid in agents_id}
    elif type(obs)==tuple:
        actions = {aid:trainer.compute_single_action(obs[0][aid],
                                                     policy_id=map_func(aid)) for aid in agents_id}
    return actions


def sigmoid(a,b,c,d,x):
    return c/(d+np.exp(-a*x+b))



def get_files(folder,filetype,keyword):
    #we shall store all the file names in this list
    filelist = []

    for root, dirs, files in os.walk(folder):
    	for file in files:
        		filelist.append(os.path.join(root,file))


    filetype_list=[]
    for file in filelist:
        if filetype in file and keyword in file:
            filetype_list.append(file)
            
    return filetype_list
    
    

