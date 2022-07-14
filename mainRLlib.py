import gym

import ray
from ray import tune
from ray.tune import Analysis
from ray.tune import ExperimentAnalysis
from pathlib import Path

from ray.rllib.agents import ppo
from ray.rllib.agents.ppo import PPOTrainer

from ray.rllib.agents import dqn
from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.env.env_context import EnvContext

from ray.rllib.agents import a3c
from ray.rllib.agents.a3c import A3CTrainer


from ray.rllib.agents.a3c import A2CTrainer

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

# from ray.tune.registry import register_env

cwd=os.getcwd()
datafolder=cwd + '/Data'

raylog=cwd + '/raylog'
#Add this folder to path

#%% Make Shiftable loads environment
#import raw data
data=pd.read_csv(datafolder + '/env_data.csv', header = None).to_numpy()
tstep_size=30 # number of minutes in each timestep
# %% convert to env data
tstep_per_day=48 #number of timesteps per day
num_days=7 #number of days
# timesteps=tstep_per_day*num_days #number of timesteps to feed the agent
timesteps=len(data)
load_num=2 #number of the load to consider
env_data=make_env_data(data, timesteps, load_num, 0.2)

## Shiftable profile example

# shiftprof=0.5*np.ones(6)
# shiftprof=np.array([0.5,0.3,0.2,0.4,0.8,0.3])
shiftprof=np.array([0.3,0.3,0.3,0.3,0.3,0.3])


#%% make train env
# env_config={"step_size": tstep_size,'window_size':24*2*1, "data": env_data,"reward_type": 2, "profile": shiftprof, "time_deliver": 37*tstep_size, 'done_condition': 'train'}

# reward_type='next_time_cost'
# reward_type='simple_cost'
reward_type='shift_cost'
# reward_type='next_shift_cost'

env_config={"step_size": tstep_size,'window_size':tstep_per_day, "data": env_data,"reward_type": reward_type, "profile": shiftprof, "time_deliver": 37*tstep_size, 'done_condition': 'mode_window'}

shiftenv=ShiftEnv(env_config)

env.check_env(shiftenv)

#%% Model
ModelCatalog.register_custom_model('shift_mask', ActionMaskModel)

#%% Make config
config=ppo.DEFAULT_CONFIG.copy()

config["env"]=ShiftEnv
config["env_config"]=env_config
config["observation_space"]=shiftenv.observation_space
config["action_space"]=shiftenv.action_space
config['model']['custom_model']=ActionMaskModel # define the custom model
config['model']['custom_model_config']['fcnet_hiddens']=[32,32] # hidden layers in the custom model
# config['model']['fcnet_hiddens']=[256,256,256,256,256]
# config['model']['fcnet_hiddens']=[256]
# config['model']['fcnet_activation']='relu'
config['seed']=1024 #define random seed
config["horizon"]=shiftenv.Tw
#evaluation
config['evaluation_interval']=1
config['evaluation_num_episodes']=10
config['evaluation_num_workers']=1

config['lr']=tune.grid_search([1e-5,1e-4])

#%%Tune esperiments
#experiment name
exp_name='Exp-NewCost'
# exp_name='Exp-FullCost'   
# exp_name='Exp-LowPV'
#allocate resources
resources = PPOTrainer.default_resource_request(config)
#define the metric and the mode criteria for identifying the best checkpoint
metric="_metric/episode_reward_mean"
mode="max"

def experiment(config):
    
    trainer=PPOTrainer(config, env=config["env"])
    weights={}
    
    # setting the seed
    seed=config['seed']
    np.random.seed(seed)
    random.seed(seed)    
    
    for i in range(2):
        print('training...')
        train_results=trainer.train()
# 
        #Metrics we are gonna log from full train_results dict
        metrics={'episode_reward_max', 
              'episode_reward_mean',
              'episode_reward_min',
              'info', 
              'episodes_total',
              'agent_timesteps_total',
              'training_iteration'}
        
        logs={k: train_results[k] for k in metrics}
        
        
        #get model weights
        for k, v in trainer.get_policy().get_weights().items():
                    weights["FCC/{}".format(k)] = v
        
        #save checkpoint
        checkpoint=trainer.save(tune.get_trial_dir())


        #evaluate agent
        print('evaluating...')
        eval_results=trainer.evaluate()
        eval_metrics={'episode_reward_max', 
              'episode_reward_mean',
              'episode_reward_min',}
        eval_logs={'evaluation':{}}
        eval_logs['evaluation']={k: eval_results['evaluation'][k] for k in eval_metrics}

        results={**logs,**weights,**eval_logs}
        # results={**eval_logs}
        tune.report(results)
        
    trainer.stop()


# pol=trainer.get_policy()
# m=pol.model
# m.variables()
# m.forward()
# vf=m.value_function()

# grad=pol.compute_gradients()





tuneobject=tune.run(
    experiment,
    config=config,
    resources_per_trial=resources,
    local_dir=raylog,
    # num_samples=4,
    # stop={'training_iteration': 10},
    checkpoint_at_end=True,
    checkpoint_freq=10,
    # resume=True,
    name=exp_name,
    verbose=0,
    # keep_checkpoints_num=10, 
    # checkpoint_score_attr=metric, 
    # mode='max'
)


Results=tuneobject.results_df

#%% instantiate test environment

test_env_data=make_env_data(data, timesteps, 4, 0.2)
# test_env_config={"step_size": tstep_size,'window_size':24*2*1, "data": test_env_data ,"reward_type": 2, "profile": shiftprof, "time_deliver": 37*tstep_size, 'done_condition': 'test'}

# test_shiftenv=ShiftEnv(test_env_config)

test_shiftenv=shiftenv
test_env_config=env_config
test_shiftenv.data=test_env_data
# test_shiftenv=shiftenv

#%% Recover checkpoints

#instantiate the tester agent
# tester=DQNTrainer(test_config)

# we must eliminate some parameters otw 
# TypeError: Failed to convert elements of {'grid_search': [1e-05, 0.0001]} to Tensor. Consider casting elements to a supported type. See https://www.tensorflow.org/api_docs/python/tf/dtypes for supported TF dtypes.

config=ppo.DEFAULT_CONFIG.copy()
# config=dqn.DEFAULT_CONFIG.copy()
config["env"]=ShiftEnv
config["env_config"]=test_env_config
config["observation_space"]=test_shiftenv.observation_space
config["action_space"]=test_shiftenv.action_space
# config["framework"]="tf2"

# tester=DQNTrainer(config, env=ShiftEnv)
tester=PPOTrainer(config, env=ShiftEnv)



#Recover the tune object from the dir
# The trainable must be initialized # reuslts must be stored in the same analysis object

analysis = ExperimentAnalysis(os.path.join(raylog, exp_name), default_metric=metric, default_mode=mode)
df=analysis.dataframe(metric,mode) #get de dataframe results

#identify the dir where is the best checkpoint according to metric and mode
bestdir=analysis.get_best_logdir(metric,mode)

# bestdir='Exp-PPO-SQR/PPOTrainer_ShiftEnv_6f381_00000_0_2022-05-24_10-25-21/'

# bestdir='/home/omega/Documents/FCUL/Projects/FlexEnv/raylog/Exp-WIN-TAR-AM/PPOTrainer_ShiftEnv_f3e03_00000_0_2022-06-23_17-10-08'

#get the best trial checkpoint
trial=analysis.get_best_checkpoint(bestdir,metric,mode)
#get the string
checkpoint=trial.local_path
print(mode,checkpoint)

#recover best agent for testing
tester.restore(checkpoint)




#%%

from plotutils import makeplot
# PLot Solutions

## Enjoy trained agent



#An emprirical pseudo-optimal solution
# A=np.zeros(T)
# A[20:26]=1
# A[69:75]=1


costs=[]

n_episodes=10
k=0
while k < n_episodes:
    action_track=[]
    state_track_temp=[]
    mask_track=[]

    full_state_track=[]

    obs = test_shiftenv.reset()

    rewards_track = []
    episode_reward=0

    T=test_shiftenv.Tw

    for i in range(T):
        # print(i)
        
        # print('1_obs_1', obs)
        state_track_temp.append(obs['observations'])
        action = tester.compute_single_action(obs)
        # action=int(A[i])
        # print('2_action',action)
        obs, reward, done, info = test_shiftenv.step(action)
        # print('3_obs_2', obs)
        # print('4_rew',reward)
        episode_reward += reward
        
        # print(obs)
        # print(action)
        # print(reward)
        full_state_track.append(obs)
        action_track.append(action)
        mask_track.append(obs['action_mask'])
        
        rewards_track.append(reward)
        
    state_track=np.array(state_track_temp)
    
    #Create dataframe state_action
    state_action_track=(state_track,np.reshape(action_track,(T, 1)), np.reshape(np.array(rewards_track),(T, 1)))
    
    
    state_action_track=np.concatenate(state_action_track, axis=1)
    state_action_track=pd.DataFrame(state_action_track, columns=list(test_shiftenv.state_vars.keys())+['actions','rewards'])
    
    state_action_track_filter=state_action_track[['tstep','minutes','gen','load','delta','excess','cost','cost_s','cost_s_x','y','y_s','actions','rewards']]
    
    Episode_reward=state_action_track_filter['cost_s'].sum()
    
    #Plot
    makeplot(T,state_action_track['load_s'],state_action_track['actions'],state_action_track['gen'],state_action_track['load'],state_action_track['delta_c'],state_action_track['tar_buy'],test_shiftenv) # 
        
    
    #get metrics
    
    costs.append(state_action_track_filter['cost_s_x'].sum()) 
    
    
    
    k+=1 
    
  
    
  

    
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    