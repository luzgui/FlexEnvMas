import gym
import flexenv as flex

import auxfunctions as fun

import stable_baselines3

from stable_baselines3 import DQN, A2C
from stable_baselines3.a2c.policies import MlpPolicy

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env

from stable_baselines3 import SAC
from stable_baselines3 import DDPG

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from stable_baselines3.ddpg.policies import MlpPolicy


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy.random as rnd
import time
import random as rnd


#%% Data and Environment preparation


#Data
cwd=os.getcwd()
datafolder=cwd + '/Data'

#impor the data csv
env_data=pd.read_csv(datafolder + '/env_data.csv', header = None)


#make and check the environment
# Select the number of timesteps to consider
# timestesps=141
timestesps=47

#Create environment
env=fun.make_env(env_data, load_num=4, timestep=timestesps, soc_max=5, eta=0.95, charge_lim=3)

#%% Stable Baselines ALGORITHMS

# Uncomment the algorithm



##DQN
## parameters
gamma=0.99
learning_rate=1e-3
buffer_size=1e6

exploration_fraction=0.01
exploration_final_eps=0.02 
exploration_initial_eps=1.0
train_freq=1
batch_size=32
double_q=True
learning_starts=1000
target_network_update_freq=500

prioritized_replay=False
prioritized_replay_alpha=0.6
prioritized_replay_beta0=0.4
prioritized_replay_beta_iters=None
prioritized_replay_eps=1e-06
param_noise=True
n_cpu_tf_sess=None
verbose=0 
tensorboard_log=None
_init_setup_model=True
policy_kwargs=None
full_tensorboard_log=False
seed=None



## Train model
# model = DQN('MlpPolicy', env, learning_rate=learning_rate, verbose=verbose,batch_size=batch_size,exploration_fraction=exploration_fraction)
# model.learn(total_timesteps=int(3e5))





##Proximal Policy Optimization
model = PPO("MlpPolicy", env, verbose=0)
model.learn(total_timesteps=3e5)


## Other algorithms
#SAC
# model = SAC('MlpPolicy', env, verbose=2).learn(total_timesteps=1e5)

#A2C
# model = A2C(MlpPolicy, env, verbose=1)
# model.learn(total_timesteps=25000)

#%% Load a trained model from the 
## (saved model files must be in .zip )


#Load + Save Model
# Save model
# model.save('ppo_1day')

#Load Model
# filename='dqn_1day'
# model_trained = DQN.load(filename, env=env) #


# filename='ppo_1day'
# model_trained = PPO.load(filename, env=env) #


# model=model_trained

# model_trained = DQN.load('flexenv_dqn', env=env)


#%% Evaluate the agent

mean_reward, std_reward = evaluate_policy(model,env, n_eval_episodes=10)

## Enjoy trained agent
action_track=[]
state_track=[]
obs = env.reset()
for i in range(timestesps):
    action, _states = model.predict(obs, deterministic=True)
    action_track.append(action)
    state_track.append(obs)
    obs, rewards, dones, info = env.step(action)
    
    env.render()

state_track=np.array(state_track)

#translate actions into charging power
action_track=[env.get_load(k) for k in action_track]


#Make plots

# sol=flex.get_actions(action_track,env)
flex.makeplot(48,state_track[:,3],action_track,env.data[:,0],env.data[:,1],env)

flex.reward_plot(env.R_Total)
