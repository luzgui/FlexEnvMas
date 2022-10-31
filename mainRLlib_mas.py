spaceimport gym

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

# load_id=['id2000', 'id2001','id2002', 'id2004'] #ISDDA id of the load to consider
load_id=['id2000', 'id2001'] #ISDDA id of the load to consider

#%% Number of agents
num_agents=len(load_id)
agents_id=['ag'+str(k) for k in range(num_agents)]

env_data=make_env_data_mas(data, timesteps, load_id, 0.5, num_agents,agents_id)

## Shiftable profile example

# shiftprof=0.5*np.ones(6)
# shiftprof=np.array([0.5,0.3,0.2,0.4,0.8,0.3])

##Agents appliance profiles
AgentsProfiles=np.array([[0.3,0.3,0.3,0.3,0.3,0.3],
                   [0.5,0.5,0.5,0.5,0.5],
                   [0.6,0.6,0.6,0.6,0.6],
                   [0.9,0.9,0.9,0.9,0.9]], dtype=object)

shiftprof={agent:profile for (agent,profile) in zip(agents_id,AgentsProfiles)}

#Agents delivery times
delivery_times={ag:37*tstep_size for ag in agents_id }


#%% make train env
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


#%% Make config

exp_name='mas_0'

#Multi-Agent Setup

from shiftenvRLlib_mas import ShiftEnvMas
shiftenv_mas=ShiftEnvMas(env_config) 
env.check_env(shiftenv_mas)


obs_space=shiftenv_mas.observation_space
action_space=shiftenv_mas.action_space


    
config={}    

policies={'pol_0':(None,
                   obs_space,
                   action_space,
                   config,),
          'pol_1':(None,
                   obs_space,
                   action_space,
                   config)}


def policy_mapping_fn(agent_id):
    if agent_id=='ag0':
        return 'pol_0'
    elif agent_id == 'ag1':
        return 'pol_1'



#Config
config = PPOConfig()\
                .training(lr=1e-5,
                          num_sgd_iter=1,
                          train_batch_size=8000,
                          model={'custom_model':ActionMaskModel,
                                'fcnet_hiddens': [128,128],
                                'fcnet_activation':'relu',
                                'custom_model_config': 
                                    {'fcnet_hiddens': [128,128]}})\
                .environment(
                    env=ShiftEnvMas,           
                    observation_space=shiftenv_mas.observation_space,
                    action_space=shiftenv_mas.action_space,
                    env_config=env_config)\
                .debugging(seed=1024,log_level='WARN')\
                .rollouts(num_rollout_workers=1)\
                .multi_agent(policies=policies,
                              policy_mapping_fn=policy_mapping_fn)


#%% Train
trainer=config.build() 
trainer.train()


#%% Deploy
def get_actions(obs,trainer,agents_id, map_func):
    if type(obs)==dict:
        actions = {aid:trainer.compute_single_action(obs[aid],policy_id=map_func(aid)) for aid in agents_id}
    elif type(obs)==tuple:
        actions = {aid:trainer.compute_single_action(obs[0][aid],policy_id=map_func(aid)) for aid in agents_id}
    return actions



obs=shiftenv_mas.reset()
# print(obs)
for k in range(48):  
    # action={aid:random.randint(0,1) for aid,a in zip(shiftenv_mas.agents_id,[1,0,0,1])}
    actions=get_actions(obs, trainer, shiftenv_mas.agents_id,policy_mapping_fn)
    obs=shiftenv_mas.step(actions)


# history=shiftenv_mas.state_hist

# hist_ag0=history.loc['ag0']
# hist_ag1=history.loc['ag1']




#%% Run Agent (plotting+analytics)
from plotutils import makeplot
# PLot Solutions

tenv=shiftenv_mas

costs=[]
rewards=[]
deltas=[]

n_episodes=1

metrics_experiment=pd.DataFrame(columns=['cost','delta_c','gamma'], 
                                index=range(n_episodes))
k=0

while k < n_episodes:

    mask_track=[]
    
    # full_state_track=[]
    
    obs = tenv.reset()
    
    
    # rewards_track = []
    # episode_reward=0
    
    T=tenv.Tw*1
    # num_days_test=T/tenv.tstep_per_day
    
    # #create a dataframe to store observations
    # state_track=pd.DataFrame(columns=tenv.state_vars.keys(), index=range(T))
    # action_reward_track=pd.DataFrame(columns=['action','reward'], index=range(T))
    # metrics_episode=pd.DataFrame(columns=['cost','delta_c','gamma'], index=range(T))

    # state_track.iloc[0]=obs['observations']
    
    
    
    for i in range(T-1):
        actions=get_actions(obs, trainer, tenv.agents_id,policy_mapping_fn)
        
        
        #compute metrics per episode
        # cost=max(0,action*tenv.profile[0]-tenv.excess0)*tenv.tar_buy
        # delta_c=(tenv.load0+action*tenv.profile[0])-tenv.gen0
        # gamma=self_suf(tenv,action)
        
        
        # metrics_episode.iloc[i]=[cost,delta_c,gamma]
        
        obs, reward, done, info = tenv.step(actions)

        
        # action_reward_track.iloc[i]=[action,reward]
            
        # mask_track.append(obs['action_mask'])
        
        # rewards_track.append(reward)
     
    # we are summing the total cost and making a mean for delta    
    
    
    state_hist=tenv.state_hist
    action_hist=tenv.action_hist    
    reward_hist=tenv.reward    

    shift_loads=pd.DataFrame()
    base_loads=pd.DataFrame()
    
    for aid in tenv.agents_id:
        shift_loads=pd.concat([shift_loads,pd.DataFrame(action_hist.loc[aid].values*tenv.profile[aid][0], columns=[aid])], axis=1)
        base_loads=pd.concat([base_loads,pd.DataFrame(state_hist.loc[aid,'load0'].values, columns=[aid])], axis=1)
        
                
    shift_loads_global=shift_loads.sum(axis=1)    
    base_loads_global=base_loads.sum(axis=1)
    
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
    from plotutils import makeplot
    makeplot(T,
             [],
             shift_loads_global,
             pd.DataFrame(state_hist.loc['ag0','gen0'].values),
             base_loads_global,
             state_hist.loc['ag0','tar_buy'],
             tenv, 
             0,
             0) #
    
    k+=1




                
                # .evaluation(evaluation_interval=1,
                #             evaluation_num_workers=1,
                #             evaluation_num_episodes=10,) 
                # .resources(placement_strategy=tune.PlacementGroupFactory([{'CPU': 1.0}] + [{'CPU': 1.0}] * 1))
                
  

                
# config = PPOConfig()\
#                 .training(lr=1e-5,
#                           num_sgd_iter=1,
#                           train_batch_size=8000,
#                           model={'fcnet_hiddens': [128,128],
#                                  'fcnet_activation':'relu',
#                                  'custom_model_config': 
#                                      {'fcnet_hiddens': [128,128]}})\
#                 .environment(
#                     env=ShiftEnvMas,           
#                     observation_space=shiftenv_mas.observation_space,
#                     action_space=shiftenv_mas.action_space,
#                     env_config=env_config)\
#                 .rollouts(num_rollout_workers=1)




#%% stuff





# action=trainer.compute_single_action(obs['ag1'])



obs_space=shiftenv_mas.observation_space
obs_space
type(obs_space)
# obs_sample=obs_space.sample()

obs_space.contains(obs[0]['ag0'])

obs=shiftenv_mas.reset()
print(obs)
for k in range(20):    
    action={aid:random.randint(0,1) for aid,a in zip(shiftenv_mas.agents_id,[1,0,0,1])}
    obs=shiftenv_mas.step(action)



type(obs)

shiftenv_mas.assert_type(obs2)




def trainable(config):
    # trainer=config.build()
    trainer=PPO(config, env=config["env"])
    trainer.train()
    
    
    

tune.run(trainable, config=config.to_dict(),resources_per_trial=tune.PlacementGroupFactory([{'CPU': 1.0}] + [{'CPU': 1.0}] * 4))



ray.shutdown()





obs_space=shiftenv_mas.observation_space

obs_space.contains(obs)








for k in range(47):

    print(k)

state=env.state

state_hist=env.state_hist
ag0_hist=state_hist.loc['ag0']












# def env_creator(config):
#     return ShiftEnv(config)

# mas_cls=make_multi_agent(env_creator)
# masenv=mas_cls(env_config)


# action={k:random.randint(0,1) for k in range(env_config['num_agents'])}


# obs=masenv.reset()
# ob2=masenv.step(action)



















