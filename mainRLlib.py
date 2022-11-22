import gym

import ray #ray2.0 implementation

from ray import tune, air
# from ray.tune import Analysis
from ray.tune import analysis
from ray.tune import ExperimentAnalysis
from pathlib import Path

#PPO algorithm
from ray.rllib.algorithms.ppo import PPO #trainer
from ray.rllib.algorithms.ppo import PPOConfig #config


from ray.rllib.env.env_context import EnvContext


#models
from ray.rllib.models import ModelCatalog

#math + data
import pandas as pd
import numpy as np

#Plotting
import matplotlib.pyplot as plt

#System
import os
from os import path
import sys
import time

import datetime
from datetime import datetime

#Custom functions
from shiftenvRLlib import ShiftEnv
from auxfunctions_shiftenv import *
from plotutils import makeplot
from models2 import ActionMaskModel

from ray.rllib.utils.pre_checks import env

import random

from trainable import trainable

# from ray.tune.registry import register_env

cwd=os.getcwd()
datafolder=cwd + '/Data'

raylog=cwd + '/raylog'
#Add this folder to path

#%% Make Shiftable loads environment
#import raw data
# data=pd.read_csv(datafolder + '/env_data.csv', header = None).to_numpy()
# data_raw=pd.read_csv(datafolder + '/issda_data_halfyear.csv').to_numpy()
data_raw=pd.read_csv(datafolder + '/issda_data_halfyear.csv')

data=data_raw[['minutes','PV','id2000', 'id2001', 'id2002', 'id2004', 'id2005',
'id2006', 'id2007', 'id2008', 'id2009', 'id2010', 'id2011', 'id2013',
'id2015', 'id2017', 'id2018', 'id2019', 'id2022', 'id2023', 'id2024',
'id2025', 'id2027', 'id2028', 'id2029', 'id2032', 'id2034', 'id2035',
'id2036', 'id2037', 'id2038', 'id2039', 'id2041', 'id2042', 'id2045',
'id2046', 'id2047', 'id2048', 'id2049', 'id2052', 'id2053', 'id2054',
'id2055', 'id2056', 'id2057', 'id2058', 'id2059', 'id2062', 'id2064']]


tstep_size=30 # number of minutes in each timestep
# %% convert to env data
tstep_per_day=48 #number of timesteps per day
num_days=7 #number of days
# timesteps=tstep_per_day*num_days #number of timesteps to feed the agent
timesteps=len(data)-1
load_id='id2000' #ISDDA id of the load to consider

env_data=make_env_data(data, timesteps, load_id, 0.5)

## Shiftable profile example

# shiftprof=0.5*np.ones(6)
# shiftprof=np.array([0.5,0.3,0.2,0.4,0.8,0.3])
shiftprof=np.array([0.3,0.3,0.3,0.3,0.3,0.3])


#%% make train env

# reward_type='next_time_cost'
# reward_type='simple_cost'
# reward_type='shift_cost'
# reward_type='next_shift_cost'
# reward_type='gauss_shift_cost'
# reward_type='excess_cost'
reward_type='excess_cost_max'
# reward_type='excess_cost_3'

env_config={"step_size": tstep_size,
            'window_size':tstep_per_day,
            'tstep_per_day':tstep_per_day,
            "data": env_data,
            "reward_type": reward_type, 
            "profile": shiftprof, 
            "time_deliver": 37*tstep_size, 
            'done_condition': 'mode_window',
            'init_condition': 'mode_window',
            'tar_type':'bi',
            'env_info': 'training environment'}

shiftenv=ShiftEnv(env_config)

env.check_env(shiftenv)

#%% Custom Model
ModelCatalog.register_custom_model('shift_mask', ActionMaskModel)

#%% Make config
# exp_name='Exp-r-max'
# exp_name='Exp-NewState'
exp_name='30set'

config = PPOConfig()\
                .training(lr=1e-5,
                          train_batch_size=8000,
                          model={'custom_model':ActionMaskModel,
                                 'fcnet_hiddens': [128,128],
                                 'fcnet_activation':'relu',
                                 'custom_model_config': 
                                     {'fcnet_hiddens': [128,128]}})\
                .environment(
                    env=ShiftEnv,           
                    observation_space=shiftenv.observation_space,
                    action_space=shiftenv.action_space,
                    env_config=env_config)\
                .resources()\
                .rollouts(num_rollout_workers=7)\
                .debugging(seed=1024,log_level='DEBUG')
                    
                # .evaluation(evaluation_interval=1,
                #             evaluation_num_workers=1,
                #             evaluation_num_episodes=10,) 
                

# config_dict=config.to_dict()
# algo=config.build()
# algo.train()


# pol=algo.get_policy()

# algo.


# config["horizon"]=shiftenv.Tw
#%% Train
tuneobject=tune.run(
    trainable,
    config=config.to_dict(),
    resources_per_trial=tune.PlacementGroupFactory([{'CPU': 1.0},
                                                    {'CPU': 1.0},
                                                    {'CPU': 1.0},
                                                    {'CPU': 1.0},
                                                    {'CPU': 1.0},
                                                    {'CPU': 1.0},
                                                    {'CPU': 1.0},
                                                    {'CPU': 1.0}]),
    local_dir=raylog,
    name=exp_name,
    verbose=3,
)
Results=tuneobject.results_df

#%% instantiate test environment
test_load_id='id2005'
test_env_data=make_env_data(data, timesteps, test_load_id, 0.5)
# test_env_config={"step_size": tstep_size,'window_size':24*2*1, "data": test_env_data ,"reward_type": 2, "profile": shiftprof, "time_deliver": 37*tstep_size, 'done_condition': 'test'}

# tenv=ShiftEnv(test_env_config)


#%% !BUG!
#we can only update the data. not the environment
# bug - ned to come back here and figure out how to make two different environments with different data 
# tenv=shiftenv # this makes the objects connceted


test_env_config=env_config
#I believe that this solves the bug
test_env_config['data']=test_env_data
test_env_config['env_info']='testing environment'
# tenv.data=test_env_data
# tenv=shiftenv
#instantiate a new environment for testing
tenv=ShiftEnv(test_env_config)

#%% Recover checkpoints
#update config for test_env
config.environment(env=ShiftEnv,           
                   observation_space=tenv.observation_space,
                   action_space=tenv.action_space,
                   env_config=test_env_config)

tester=config.build() # create agent for testing
# tester_config=tester.config
# policy=tester.get_policy()
# policy.model.internal_model.base_model.summary()


#define the metric and the mode criteria for identifying the best checkpoint
metric="_metric/episode_reward_mean"
mode="max"
#get the checkpoint
checkpoint, df = get_checkpoint(raylog, exp_name, metric, mode)

# cp_folder='3Oct/trainable_ShiftEnv_448b6_00000_0_2022-10-03_15-54-23'



# checkpoint, df = get_checkpoint(raylog,cp_folder, metric, mode)
#restore checkpoint
tester.restore(checkpoint)
# conf=tester.get_config()

#%% Run Agent (plotting+analytics)
from plotutils import makeplot
# PLot Solutions

## Enjoy trained agent

#An emprirical pseudo-optimal solution
# A=np.zeros(T)
# A[20:26]=1
# A[69:75]=1


costs=[]
rewards=[]
deltas=[]

n_episodes=1

metrics_experiment=pd.DataFrame(columns=['cost','delta_c','gamma'], index=range(n_episodes))
k=0
while k < n_episodes:
    # action_track=[]
    # state_track_temp=[]
    # full_state_track_temp=[]
    mask_track=[]
    
    # full_state_track=[]
    
    obs = tenv.reset()
    
    
    # rewards_track = []
    episode_reward=0
    
    T=tenv.Tw*1
    num_days_test=T/tenv.tstep_per_day
    
    #create a dataframe to store observations
    state_track=pd.DataFrame(columns=tenv.state_vars.keys(), index=range(T))
    action_reward_track=pd.DataFrame(columns=['action','reward'], index=range(T))
    metrics_episode=pd.DataFrame(columns=['cost','delta_c','gamma'], index=range(T))

    state_track.iloc[0]=obs['observations']
    
    
    
    for i in range(T):
        # print(i)
        state_track.iloc[i]=obs['observations']
        
        # print('1_obs_1', obs)
        # state_track_temp.append(obs['observations'])
        # full_state_track_temp.append(tenv.get_full_obs())
        action = tester.compute_single_action(obs)
        
        
        #compute metrics per episode
        cost=max(0,action*tenv.profile[0]-tenv.excess0)*tenv.tar_buy
        delta_c=(tenv.load0+action*tenv.profile[0])-tenv.gen0
        gamma=self_suf(tenv,action)
        
        
        metrics_episode.iloc[i]=[cost,delta_c,gamma]
        
        #append cost
        # cost_temp.append(max(0,action*tenv.profile[0]-tenv.excess)*tenv.tar_buy)
        # action=int(A[i])
        # print('2_action',action)
        obs, reward, done, info = tenv.step(action)
        # full_obs=shiftenv.get_full_obs()
        
        action_reward_track.iloc[i]=[action,reward]
            
        
        # episode_reward += reward
        
        # print(obs)
        # print(action)
        # print(reward)
        # full_state_track.append(full_obs)
        # action_track.append(action)
        mask_track.append(obs['action_mask'])
        
        # rewards_track.append(reward)
     
    # we are summing the total cost and making a mean for delta    
    
    
    
    full_track=pd.concat([state_track, action_reward_track,metrics_episode],axis=1)
    full_track_filter=full_track[['tstep','minutes','gen0','load0','delta0','excess0','tar_buy','E_prof', 'action', 'reward','cost', 'delta_c', 'gamma']]

    #gamma per epsidode is beying divided by the total amount of energy that appliances need to consume.
    metrics_experiment.iloc[k]=[metrics_episode['cost'].sum(),
                                metrics_episode['delta_c'].mean(), 
                                metrics_episode['gamma'].sum()/(num_days_test*tenv.E_prof)] 
    
    # print(metrics_episode['gamma'].sum()/(num_days_test*tenv.E_prof))
    # print(full_track['load0'].sum())
    
    print(tenv.E_prof/full_track['excess0'].sum())
    
    #PLots
    makeplot(T,
             metrics_episode['delta_c'],
             full_track['action']*0.3,
             full_track['gen0'],
             full_track['load0'],
             full_track['tar_buy'],
             tenv, 
             metrics_episode['cost'].sum(),
             full_track['reward'].sum()) #
    
    k+=1
    

       
# boxplot
# fig = plt.figure(figsize =(10, 7))
# plot_vals=plt.boxplot(metrics_experiment, meanline=True, showmeans=True, labels=['App Daily cost (€)','Daily mean delta','App Daily self-sufficiency'])
# medians=[item.get_ydata() for item in plot_vals['medians']] #get the median values
# medians=[round(meds[0],3) for meds in medians]
# means=[item.get_ydata() for item in plot_vals['means']] #get the median values
# means=[round(meds[0],3) for meds in means]   
# plt.grid('minor')
# plt.title(' N={}'.format(round(n_episodes)))
# plt.show()






        
#     # state_track=np.array(state_track_temp)
#     # full_state_track=np.array(full_state_track_temp)
    
#     #Create dataframe state_action
#     state_action_track=(state_track,np.reshape(action_track,(T, 1)), np.reshape(np.array(rewards_track),(T, 1)))
    
    
#     state_action_track=np.concatenate(state_action_track, axis=1)
#     state_action_track=pd.DataFrame(state_action_track, columns=list(tenv.state_vars.keys())+['actions','rewards'])
    
#     state_action_track_filter=state_action_track[['tstep','minutes','gen','load','delta','excess','y','y_s','actions','rewards']]
    
#     # Episode_reward=state_action_track_filter['cost_s'].sum()
    
#     #Plot

        
    
#     #get metrics
    
#     # means
#     delta_c_episode=state_action_track_filter['delta'].mean()
    
#     #sums
#     cost_episode=sum(cost_temp)
#     reward_episode=state_action_track_filter['rewards'].sum()
    
    
#     #lists per episode
#     costs.append(cost_episode) 
#     rewards.append(reward_episode)
#     deltas.append(delta_c_episode)
    
    
 
    
    
    
#     k+=1 
#     # print(cost_episode)
    

# R=pd.DataFrame({'c':costs,'r':rewards})

    

  
# costs

    
# #%%
# k=0
# obs=shiftenv.reset()
# while k<100:
#     # a=shiftenv.action_space.sample()
#     a=0
#     print(a)
#     step_out=shiftenv.step(a)
#     obs=step_out[0]
#     print(obs)
#     assert type(obs)==dict, 'Não é dict'
#     print(k, 'k=')
#     k+=1
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    