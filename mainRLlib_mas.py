import gym

import ray #ray2.0 implementation

from ray import tune, air
# from ray.tune import Analysis
from ray.tune import analysis
from ray.tune import ExperimentAnalysis
from ray.tune import TuneConfig
from ray.tune.execution.trial_runner import _load_trial_from_checkpoint
from ray.tune.experiment import trial



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

from trainable import *

# from ray.tune.registry import register_env
from ray.rllib.examples.env.multi_agent import MultiAgentCartPole, make_multi_agent
from shiftenvRLlib_mas import ShiftEnvMas
# Custom Model
ModelCatalog.register_custom_model('shift_mask', ActionMaskModel)


#
cwd=os.getcwd()
datafolder=cwd + '/Data'
raylog=cwd + '/raylog'
#Add this folder to path

#%% Make Shiftable loads environment
#import raw data
# data_raw=pd.read_csv(datafolder + '/issda_data_halfyear.csv')

# data_raw_cons=pd.read_excel(datafolder + '/Dataset_gecad.xlsx', 'Total Consumers')
data_raw_prod=pd.read_excel(datafolder + '/Dataset_gecad.xlsx', 'Total Producers')


data = get_raw_data('Dataset_gecad.xlsx', datafolder)
# data=data_raw[['minutes','PV0','Ag0','Ag1']]


tstep_size=15 # number of minutes in each timestep
# %% convert to env data
tstep_per_day=96 #number of timesteps per day
num_days=7 #number of days
# timesteps=tstep_per_day*num_days #number of timesteps to feed the agent
timesteps=len(data)-1

# load_id=['id2000', 'id2001','id2002', 'id2004'] #ISDDA id of the load to consider
# load_id=['id2000', 'id2001'] #ISDDA id of the load to consider

# load_id=['ag1','ag2']
load_id=['ag1']


#%% Make env data
num_agents=len(load_id)
agents_id=load_id

# agents_id=['ag'+str(k) for k in range(num_agents)]
#What are agents data?


env_data=make_env_data_mas(data, timesteps, load_id, 4, num_agents,agents_id)

# df=env_data.loc[(slice('ag0', 'ag1'), slice(0, 0)), :]

## Shiftable profile example

# shiftprof=0.5*np.ones(6)
# shiftprof=np.array([0.5,0.3,0.2,0.4,0.8,0.3])

##Agents appliance profiles
AgentsProfiles=np.array([[1.2,1.2,1.2,1.2,1.2,1.2,1.2],
                   [1.5,1.5,1.5,1.5,1.5],
                   [0.6,0.6,0.6,0.6,0.6],
                   [0.9,0.9,0.9,0.9,0.9]], dtype=object)

shiftprof={agent:profile for (agent,profile) in zip(agents_id,AgentsProfiles)}

#Agents delivery times

# time=20 #time of delivery
delivery_times={ag:37*tstep_size for ag in agents_id }


#%% make env config
reward_type='excess_cost_max'


env_config={"step_size": tstep_size,
            'window_size':tstep_per_day,
            'tstep_per_day':tstep_per_day,
            "data": env_data,
            "reward_type": reward_type, 
            "profile": shiftprof, 
            "time_deliver": delivery_times, 
            'done_condition': 'mode_window',
            'init_condition': 'mode_window',
            'tar_type':'bi',
            'env_info': 'training environment',
            'num_agents':num_agents,
            'agents_id':agents_id}


#%% Make config + Environment

exp_name='1_Ag_new'

#Multi-Agent Setup

from shiftenvRLlib_mas import ShiftEnvMas
menv=ShiftEnvMas(env_config) 
env.check_env(menv)

    
config={}    

policies={'pol_'+aid:(None,
                   menv.observation_space,
                   menv.action_space,
                   config,) for aid in menv.agents_id }




#Config
config = PPOConfig()\
                .training(lr=1e-5,
                          num_sgd_iter=1,
                          train_batch_size=1000,
                          model={'custom_model':ActionMaskModel,
                                'fcnet_hiddens': [128,128],
                                'fcnet_activation':'relu',
                                'custom_model_config': 
                                    {'fcnet_hiddens': [128,128]}})\
                .environment(
                    env=ShiftEnvMas,           
                    observation_space=menv.observation_space,
                    action_space=menv.action_space,
                    env_config=env_config)\
                .debugging(seed=1024,log_level='WARN')\
                .rollouts(num_rollout_workers=1)\
                .multi_agent(policies=policies,
                              policy_mapping_fn=policy_mapping_fn)

                # .evaluation(evaluation_interval=1,
                #             evaluation_num_workers=1,
                #             evaluation_num_episodes=10,) 
                # .resources(placement_strategy=tune.PlacementGroupFactory([{'CPU': 1.0}] + [{'CPU': 1.0}] * 1))

#%% Train
# trainer=config.build() 
# trainer.train()
    
from trainable import *


tuneResults=tune.run(trainable_mas,
         config=config.to_dict(),
         resources_per_trial=tune.PlacementGroupFactory([{'CPU': 1.0}] + [{'CPU': 1.0}] * 4),
         local_dir=raylog,
         name=exp_name,
         verbose=3)

# Results=tuneResult



#%% Recover and load checkpoints
#define the metric and the mode criteria for identifying the best checkpoint
metric="_metric/episode_reward_mean"
mode="max"

exp_name='1Ag_0512'
# exp_name='1_Ag_new'
# exp_name='3Oct'
# exp_name='1Ag'
# log_dir=raylog

#get best checkpoint info
best_checkpoint, df, best_trial = get_checkpoint(raylog, exp_name, metric, mode)

#config for best trial
best_config=best_trial.config


#%% Generate data for testing

#indezx for load and for agent is also 'ag'
test_load_id=['ag2'] #selct new loads for testing 
test_agents_id=['ag1'] #choose which agents are in play

test_env_data=make_env_data_mas(data, 
                                len(data)-1, 
                                test_load_id, 
                                4, 
                                num_agents,
                                test_agents_id)

#%% Update config with test data
# !BUG!
#we can only update the data. not the environment
# bug - ned to come back here and figure out how to make two different environments with different data 
best_config['env_config']['data']=test_env_data #update data
best_config['env_config']['env_info']='testing environment' 
#Make the testing environment
tenv=ShiftEnvMas(best_config['env_config'])

#Instantiate and restore agent
tester=PPO(best_config,env=best_config['env'])
tester.restore(best_checkpoint)


# tester_config=tester.config
# policy=tester.get_policy()
# policy.model.internal_model.base_model.summary()

# p0=tester.get_policy('pol_ag0')
# p1=tester.get_policy('pol_ag1')

# # w0=p0.get_weights()
# w1=p1.get_weights()

# weights_file=

# m1=p1.model.internal_model.base_model.summary()




#%% Deploy
#%% Run Agent (plotting+analytics)
from plotutils import makeplot
# Plot Solutions
n_episodes=1

metrics_experiment=pd.DataFrame(columns=['cost','delta_c','gamma'], 
                                index=range(n_episodes))
k=0

while k < n_episodes:

    mask_track=[]

    obs = tenv.reset()
    
    
    T=tenv.Tw*1
    # num_days_test=T/tenv.tstep_per_day

    # metrics_episode=pd.DataFrame(columns=['cost','delta_c','gamma'], index=range(T))

    for i in range(T):
        actions=get_actions(obs, tester, tenv.agents_id,policy_mapping_fn)
        obs, reward, done, info = tenv.step(actions)
        
        
        #compute metrics per episode
        # cost=max(0,action*tenv.profile[0]-tenv.excess0)*tenv.tar_buy
        # delta_c=(tenv.load0+action*tenv.profile[0])-tenv.gen0
        # gamma=self_suf(tenv,action)
        
        
        # metrics_episode.iloc[i]=[cost,delta_c,gamma]
        
        
    # we are summing the total cost and making a mean for delta    
    from auxfunctions_shiftenv import get_post_data
    full_state, env_state=get_post_data(tenv)
    
    
    
    from plotutils import makeplot
    makeplot(T,
             [],
             env_state['shift_T'],
             env_state['gen0'],
             env_state['load_T'],
             env_state['tar_buy'],
             tenv, 
             0,
             0) #
        
    k+=1




    
    # full_track=pd.concat([state_track, action_reward_track,metrics_episode],axis=1)
    # full_track_filter=full_track[['tstep','minutes','gen0','load0','delta0','excess0','tar_buy','E_prof', 'action', 'reward','cost', 'delta_c', 'gamma']]

    # #gamma per epsidode is beying divided by the total amount of energy that appliances need to consume.
    # metrics_experiment.iloc[k]=[metrics_episode['cost'].sum(),
    #                             metrics_episode['delta_c'].mean(), 
    #                             metrics_episode['gamma'].sum()/(num_days_test*tenv.E_prof)] 
    
    # # print(metrics_episode['gamma'].sum()/(num_days_test*tenv.E_prof))
    # # print(full_track['load0'].sum())
    
    # print(tenv.E_prof/full_track['excess0'].sum())
    
    #PLots


