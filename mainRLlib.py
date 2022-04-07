import gym

import ray
from ray import tune
from ray.tune import Analysis
from pathlib import Path

from ray.rllib.agents import ppo
from ray.rllib.agents.ppo import PPOTrainer

from ray.rllib.agents import dqn
from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.env.env_context import EnvContext

from ray.rllib.agents import a3c
from ray.rllib.agents.a3c import A3CTrainer


from ray.rllib.agents.a3c import A2CTrainer

#math + data
import pandas as pd
import numpy as np

#Plotting
import matplotlib.pyplot as plt

#System
import os
import sys
import time

#Custom functions
from flexenvRLlib import FlexEnv
from plotutils import makeplot

cwd=os.getcwd()
datafolder=cwd + '/Data'

raylog=cwd + '/raylog'
#Add this folder to path



#%% Create data for the environment

#impor the data csv
env_data=pd.read_csv(datafolder + '/env_data.csv', header = None)


#make and check the environment
# Select the number of timesteps to consider
# timestesps=141
timesteps=47

#Create environmentn from Gym
# flexenv=fun.make_env(env_data, load_num=2, timestep=timesteps, soc_max=4, eta=0.95, charge_lim=2, min_charge_step=0.02, reward_type=2)


# Create an RLlib Trainer instance to learn how to act in the above
# environment.
load_num=2 




data=env_data
soc_max=4.0
eta=0.95
charge_lim=2.0
min_charge_step=0.02
reward_type=2

load_num=2

env_data=data.to_numpy()
load=env_data[0:timesteps,load_num] # Escolhe timestep (um número) valores da coluna load_num, que representa os gastos uma casa
gen=abs(env_data[0:timesteps,1]) # Primeira coluna da data
# Modification (António) - Multiply the gen by 0.5 to make it smaller generation values
data=np.vstack((gen*1,1*load)).T # Duas colunas, a primeira retrata



env_config={"data": data,"soc_max": 4,"eta": 0.95,"charge_lim": 2,"min_charge_step": 0.02,"reward_type": 2}

flexenv=FlexEnv(env_config)



#Tune esperiments

# experiment(config)
config=ppo.DEFAULT_CONFIG.copy()
config["env"]=FlexEnv
config["env_config"]=env_config
config["observation_space"]=flexenv.observation_space
config["action_space"]=flexenv.action_space
# config["double_q"]=True
# config["dueling"]=True
config["lr"]=tune.grid_search([1e-5, 1e-4, 1e-3])
# config["gamma"]=tune.grid_search([0.8,0.9,0.99])
# config["kl_coeff"]=tune.grid_search([0.1,0.2,0.3])
# config["train_batch_size"]=tune.grid_search([4000,8000])
# config["sgd_minibatch_size"]=tune.grid_search([128,256])


# config["horizon"]=1000
config["framework"]="tf2"
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

tuneobject=tune.run(
    PPOTrainer,
    config=config,
    # resources_per_trial=DQNTrainer.default_resource_request(config),
    local_dir=raylog,
    # num_samples=4,
    stop={'training_iteration': 2 },
    checkpoint_at_end=True,
    checkpoint_freq=10,
    name='Exp-PPO-Tune-gridsearch-cenas',
    verbose=0,
    # keep_checkpoints_num=10, 
    checkpoint_score_attr="episode_reward_mean"
)


Results=tuneobject.results_df


#%% Recover checkpoints

#instantiate the tester agent
# tester=DQNTrainer(test_config)

# we must eliminate some parameters otw 
# TypeError: Failed to convert elements of {'grid_search': [1e-05, 0.0001]} to Tensor. Consider casting elements to a supported type. See https://www.tensorflow.org/api_docs/python/tf/dtypes for supported TF dtypes.

config=ppo.DEFAULT_CONFIG.copy()
config["env"]=FlexEnv
config["env_config"]=env_config
config["observation_space"]=flexenv.observation_space
config["action_space"]=flexenv.action_space
config["framework"]="tf2"
tester=PPOTrainer(config, env=FlexEnv)

#define the metric and the mode criteria for identifying the best checkpoint
metric='episode_reward_mean'
mode='max'

#Recover the tune object from the dir
tuneobject = Analysis(raylog + '/Exp-PPO-Tune-gridsearch', default_metric=metric, default_mode=mode)


#identify the dir where is the best checkpoint according to metric and mode
bestdir=tuneobject.get_best_logdir(metric,mode)

#get the best trial checkpoint
trial=tuneobject.get_best_checkpoint(bestdir,metric,mode)
#get the string
checkpoint=trial.local_path


#recover best agent for testing
tester.restore(checkpoint)



# PLot Solutions

## Enjoy trained agent
action_track=[]
state_track=[]
obs = flexenv.reset()
rewards_track = []
episode_reward=0

for i in range(timesteps):
    
    state_track.append(obs)
    
    action = tester.compute_single_action(obs)
    obs, reward, done, info = flexenv.step(action)
    episode_reward += reward
    

    # print(obs)
    # print(int(action))
    # print(rewards)
    action_track.append(action)
    
    rewards_track.append(reward)
    

state_track=np.array(state_track)

#translate actions into charging power
action_numbers = action_track
# action_track=[flexenv.get_charge_discharge(k) for k in action_track]


#Create dataframe state_action

state_action_track=(state_track,np.reshape(action_track,(timesteps, 1)), np.reshape(np.array(rewards_track),(timesteps, 1)))


state_action_track=np.concatenate(state_action_track, axis=1)
state_action_track=pd.DataFrame(state_action_track, columns=flexenv.varnames+('actions','rewards'))



#Plot

makeplot(48,state_action_track['soc'],state_action_track['actions'],state_action_track['gen'],state_action_track['load'],state_action_track['delta'],flexenv) # O Tempo e o Env
    
  
