import gym

from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.env.env_context import EnvContext

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
from flexenv_rllib import FlexEnv
from plotutils import makeplot

cwd=os.getcwd()
datafolder=cwd + '/Data'

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



config={"data": data,"soc_max": 4,"eta": 0.95,"charge_lim": 2,"min_charge_step": 0.02,"reward_type": 2}

flexenv=FlexEnv(config)




# trainer=DQNTrainer(env=FlexEnv(data,soc_max,eta,charge_lim,min_charge_step, reward_type), config={"framework": "tf2"}) 



trainer = DQNTrainer(
    config={
        # Env class to use (here: our gym.Env sub-class from above).
        "env": FlexEnv,
        # Config dict to be passed to our custom env's constructor.
        "env_config":{"data": data,"soc_max": 4,"eta": 0.95,"charge_lim": 2,"min_charge_step": 0.02,"reward_type": 2},
        # },
        # Parallelize environment rollouts.
        "num_workers": 2,
        "log_level": "WARN",
        "framework": 'tf',
        "eager_tracing": True
    })

# Train for n iterations and report results (mean episode rewards).
for i in range(100):
    results = trainer.train()
    print(f"Iter: {i}; avg. reward={results['episode_reward_mean']}")
    
    

# Get policy

policy=trainer.get_policy()


# PLot Solutions

## Enjoy trained agent
action_track=[]
state_track=[]
obs = flexenv.reset()
rewards_track = []
episode_reward=0

for i in range(timesteps):
    
    state_track.append(obs)
    
    action = trainer.compute_single_action(obs)
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
action_track=[flexenv.get_charge_discharge(k) for k in action_track]


#Create dataframe state_action

state_action_track=(state_track,np.reshape(action_track,(timesteps, 1)), np.reshape(np.array(rewards_track),(timesteps, 1)))


state_action_track=np.concatenate(state_action_track, axis=1)
state_action_track=pd.DataFrame(state_action_track, columns=flexenv.varnames+('actions','rewards'))



#Plot

makeplot(48,state_action_track['soc'],state_action_track['actions'],state_action_track['gen'],state_action_track['load'],state_action_track['delta'],flexenv) # O Tempo e o Env
    
  



    
    