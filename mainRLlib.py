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


# from ray.tune.registry import register_env

cwd=os.getcwd()
datafolder=cwd + '/Data'

raylog=cwd + '/raylog'
#Add this folder to path

#%% Make Shiftable loads environment
#import raw data
data=pd.read_csv(datafolder + '/env_data.csv', header = None).to_numpy()
tstep_size=30 # number of minutes in each timestep
#%% convert to env data
# timesteps=48*3
timesteps=len(data)
load_num=2
env_data=make_env_data(data, timesteps, load_num)

## Shiftable profile example

# shiftprof=0.5*np.ones(6)
# shiftprof=np.array([0.5,0.3,0.2,0.4,0.8,0.3])
shiftprof=np.array([0.3,0.3,0.3,0.3,0.3,0.3])


#%% make train env
env_config={"step_size": tstep_size, "data": env_data,"reward_type": 2, "profile": shiftprof, "time_deliver": 37*tstep_size, 'done_condition': 'train'}

shiftenv=ShiftEnv(env_config)

env.check_env(shiftenv)

#%% Model
# from models import ActionMaskModel
# m=ActionMaskModel(shiftenv.observation_space, 
#                   shiftenv.action_space,
#                   shiftenv.action_space.n,
#                   env_config,
#                   'model0')




ModelCatalog.register_custom_model('shift_mask', ActionMaskModel)


#%%Tune esperiments

# experiment(config)
config=ppo.DEFAULT_CONFIG.copy()
# config=dqn.DEFAULT_CONFIG.copy()
# config["env"]=FlexEnv
config["env"]=ShiftEnv


config["env_config"]=env_config
config["observation_space"]=shiftenv.observation_space
config["action_space"]=shiftenv.action_space
config["_disable_preprocessor_api"]=True
# config["double_q"]=True
# config["dueling"]=True
# config["lr"]=tune.grid_search([1e-5, 1e-4, 1e-3])
# config["gamma"]=tune.grid_search([0.8,0.9,0.99])
# config['model']['fcnet_hiddens']=[256,256]
# config['model']['use_lstm']=True

# config['model']['custom_model']=ActionMaskModel
# config['log_level']='INFO'

# config["gamma"]=0.7
# config["kl_coeff"]=tune.grid_search([0.1,0.2,0.3])
# config["train_batch_size"]=tune.grid_search([8000.16000])
# config["sgd_minibatch_size"]=tune.grid_search([128,256])


config["horizon"]=shiftenv.Tw
# config["framework"]="tf2"
# config["eager_tracing"]=True
# config["lr"]=1e-4

# config["lr"]=tune.uniform(0, 2)
# config["framework"]='tf2'
# config["explore"]=False
# config["exploration_config"]={
#     "type": "EpsilonGreedy",
#     "initial_epsilon": 1.0,
#     "final_epsilon": 0.09,
#     "epsilon_timesteps": 10000}

# exp_name='Exp-PPO-Weights'


exp_name='Exp-WIN-TAR-PreProc'

#make a trainable that logs model weights

# def trainable(config):
#     library.init(
#         name=trial_id,
#         id=trial_id,
#         resume=trial_id,
#         reinit=True,
#         allow_val_change=True)
#     library.set_log_path(tune.get_trial_dir())

#     for step in range(100):
#         library.log_model(...)
#         library.log(results, step=step)
#         tune.report(results)


# def trainable(config):
    
#     trainer=PPOTrainer(config)
#     print('hello')
    
#     result={}
#     for k, v in trainer.get_policy().get_weights().items():
#                 result["FCC/{}".format(k)] = v
    
#     tune.report(result)
    
#     return trainer

# from ray.rllib.agents.callbacks import DefaultCallbacks

# class MyCallbacks(DefaultCallbacks):
    
#     def setup():
#         pass
    
#     def on_train_result(self, trainer, result: dict, **kwargs):
#         for k, v in trainer.get_policy().get_weights().items():
#             result["FCC/{}".format(k)] = v
            

# tuneobject2=tune.run(
#     PPOTrainer,
#     config=config,
#     # resources_per_trial=DQNTrainer.default_resource_request(config),
#     local_dir=raylog,
#     # num_samples=4,
#     stop={'training_iteration': 200 },
#     checkpoint_at_end=True,
#     checkpoint_freq=10,
#     name=exp_name,
#     verbose=0,
#     # keep_checkpoints_num=10, 
#     callbacks=[MyCallbacks()],
#     checkpoint_score_attr="episode_reward_mean"
# )


# tuneobject=tune.run(
#     DQNTrainer,
#     config=config,
#     # resources_per_trial=DQNTrainer.default_resource_request(config),
#     local_dir=raylog,
#     # num_samples=4,
#     stop={'training_iteration': 200 },
#     checkpoint_at_end=True,
#     checkpoint_freq=10,
#     name=exp_name,
#     verbose=0,
#     # keep_checkpoints_num=10, 
#     checkpoint_score_attr="episode_reward_mean"
# )


tuneobject=tune.run(
    PPOTrainer,
    config=config,
    # resources_per_trial=DQNTrainer.default_resource_request(config),
    local_dir=raylog,
    # num_samples=4,
    stop={'training_iteration': 300 },
    checkpoint_at_end=True,
    checkpoint_freq=10,
    # resume=True,
    name=exp_name,
    verbose=0,
    # keep_checkpoints_num=10, 
    checkpoint_score_attr="episode_reward_mean"
)



Results=tuneobject.results_df

#%% instantiate test environment

test_env_data=make_env_data(data, 48*1, 4)
test_env_config={"step_size": tstep_size, "data": test_env_data ,"reward_type": 2, "profile": shiftprof, "time_deliver": 37*tstep_size, 'done_condition': 'test'}

test_shiftenv=ShiftEnv(test_env_config)

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

#define the metric and the mode criteria for identifying the best checkpoint
metric='episode_reward_mean'
mode='max'

#Recover the tune object from the dir
# The trainable must be initialized # reuslts must be stored in the same analysis object
analysis = ExperimentAnalysis(os.path.join(raylog, exp_name), default_metric=metric, default_mode=mode)
df=analysis.dataframe(metric,mode) #get de dataframe results

#identify the dir where is the best checkpoint according to metric and mode
bestdir=analysis.get_best_logdir(metric,mode)

# bestdir='Exp-PPO-SQR/PPOTrainer_ShiftEnv_6f381_00000_0_2022-05-24_10-25-21/'

#get the best trial checkpoint
trial=analysis.get_best_checkpoint(bestdir,metric,mode)
#get the string
checkpoint=trial.local_path


#recover best agent for testing
tester.restore(checkpoint)


#%%

from plotutils import makeplot
# PLot Solutions

## Enjoy trained agent
action_track=[]
state_track=[]

obs = test_shiftenv.reset() #use the fuction taht resets to zero

rewards_track = []
episode_reward=0

for i in range(test_shiftenv.T):
    
    state_track.append(obs)
    action = tester.compute_single_action(obs)
    # print(action)
    obs, reward, done, info = test_shiftenv.step(action)
    episode_reward += reward
    
    # print(obs)
    # print(action)
    # print(reward)
    action_track.append(action)
    
    rewards_track.append(reward)
    
state_track=np.array(state_track)

#Create dataframe state_action
state_action_track=(state_track,np.reshape(action_track,(test_shiftenv.T, 1)), np.reshape(np.array(rewards_track),(test_shiftenv.T, 1)))


state_action_track=np.concatenate(state_action_track, axis=1)
state_action_track=pd.DataFrame(state_action_track, columns=list(test_shiftenv.state_vars.keys())+['actions','rewards'])



#Plot
makeplot(48,state_action_track['load_s'],state_action_track['actions'],state_action_track['gen'],state_action_track['load'],state_action_track['delta'],test_shiftenv) # 
    
  