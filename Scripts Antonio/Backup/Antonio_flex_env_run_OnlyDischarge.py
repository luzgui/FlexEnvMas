# In[2]:


import gym
import Antonio_FlexEnv_OnlyDischarge as flex

import Antonio_auxfunctions_OnlyDischarge as fun

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

from time import perf_counter


# # Data and Environment preparation

# In[3]:


#Data
cwd=os.getcwd()
datafolder=cwd + '/Data'

#impor the data csv
env_data=pd.read_csv(datafolder + '/env_data.csv', header = None)

#make and check the environment
# Select the number of timesteps to consider
# timesteps=141
timesteps=47

#Create environment. Based on the aux functions code.
env=fun.make_env(env_data, load_num=4, timestep=timesteps, soc_max=2, eta=0.95, charge_lim=3, min_charge_step=0.2)


# In[4]:


env.observation_space


# In[5]:


env.observation_space.high[3]


# In[6]:


env.action_space


# In[7]:


# help(env.action_space)


# In[8]:


# charge_steps


# In[9]:


data=env_data
load_num=4
timestep=timesteps

env_data
env_data1=data.to_numpy()
load=env_data1[0:timestep,load_num]
gen=abs(env_data1[0:timestep,1])
data1=np.vstack((0.5*gen,6*load)).T


# In[10]:


env_data


# In[11]:


env_data[4]


# In[12]:


env_data1.shape # 1a coluna é a data, 2a é a geração, 3a a 12a são colunas de gastos energéticos


# In[13]:


env_data1


# In[14]:


load.shape


# In[15]:


env_data1[0:timestep,4] # É só um dia de dados 


# In[16]:


data1 # # Duas colunas, a primeira retrata a geração, a segunda representa os gastos energéticos da casa 


# In[17]:


data1.shape


# In[18]:


np.vstack((1.5*gen,6*load))


# # Generation values
# Data analysis to understand the data of the PV's

# In[19]:


# Values of generation
env_data[1].value_counts()


# In[20]:


env_data[env_data[1] ==1.764][0] # Dias e horas para as quais a geração de energia é maior


# In[21]:


df_generation = 0


# In[22]:


# pd.options.mode.chained_assignment = None 

df_generation = env_data[[0,1]]
df_generation.columns = ['Date','Generation']


# In[23]:


df_generation['Date'] = pd.to_datetime(df_generation['Date'])


# In[24]:


df_generation['Year'] = df_generation['Date'].apply(lambda time: time.year)
df_generation['Month'] = df_generation['Date'].apply(lambda time: time.month)
dmap = {1:'January',2:'February',3:'March',4:'April',5:'May',6:'June',7:'July',8:'August',9:'September',10:'October',11:'November',12:'December'}
df_generation['Month'] = df_generation['Month'].map(dmap)
df_generation['Day'] = df_generation['Date'].apply(lambda time: time.day)
df_generation['Day of Week'] = df_generation['Date'].apply(lambda time: time.dayofweek)
dmap1 = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
df_generation['Day of Week'] = df_generation['Day of Week'].map(dmap1)
df_generation['Hour'] = df_generation['Date'].apply(lambda time: time.hour)
df_generation['Minutes'] = df_generation['Date'].apply(lambda time: time.minute)


# In[25]:


df_generation


# In[26]:


df_generation[df_generation['Generation']==1.764]['Hour'].value_counts()
# Determining the hours at which the generation is the biggest


# In[27]:


df_generation[df_generation['Generation']==1.764]['Month'].value_counts()
# Determining the months at which the generation is the biggest


# In[28]:


df_generation[df_generation['Generation']==0]['Hour'].value_counts()
# Determining the hours at which the generation is non existential


# # Stable Baselines ALGORITHMS
# 
# Uncomment the algorithm

# In[29]:


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
# Policy='LnMlpPolicy'

t1_start = perf_counter()

## Train model
model = DQN('MlpPolicy', env, learning_rate=learning_rate, verbose=1,batch_size=batch_size,exploration_fraction=exploration_fraction,)
model.learn(total_timesteps=int(9e5))

t1_stop = perf_counter()
print("\nElapsed time:", t1_stop, t1_start)
print("Elapsed time during the whole program in seconds:", t1_stop-t1_start)

##Proximal Policy Optimization
# model = PPO("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=4e5)


## Other algorithms
#SAC
# model = SAC('MlpPolicy', env, verbose=2).learn(total_timesteps=1e5)

#A2C
# model = A2C(MlpPolicy, env, verbose=1)
# model.learn(total_timesteps=25000)


# # Load a trained model 
# (saved model files must be in .zip )
# 

# In[30]:


#Load + Save Model
# Save model
# model.save('ppo_1day')

#Load Model
# filename='DQN_1day_6.7e5steps_2hours'
# filename = 'DQN_1day_100000_steps'
# model_trained = DQN.load(filename, env=env) 

# filename='ppo_1day_5e6steps_7910seconds' # Na verdade bastam bastante menos steps mas este dá sempre 10 de reward
# model_trained = PPO.load(filename, env=env) #

# filename='A2C_1day_6e5_steps_1088seconds'
# model_trained = A2C.load(filename, env=env) #

# model_trained = DQN.load('flexenv_dqn', env=env)

# model=model_trained


# # Evaluate the agent

# In[33]:


mean_reward, std_reward = evaluate_policy(model,env, n_eval_episodes=5)

## Enjoy trained agent
action_track=[]
state_track=[]
obs = env.reset()

for i in range(timesteps):
    action, _states = model.predict(obs, deterministic=True)
    action_track.append(action)
    state_track.append(obs)
    obs, rewards, dones, info = env.step(action)
    
    env.render()

state_track=np.array(state_track)

#translate actions into charging power
action_track=[env.get_charge(k) for k in action_track]


# # Modificação António do Evaluate Agent

# In[34]:


## Enjoy trained agent
action_track=[]
state_track=[]
obs = env.reset()
rewards_track = []
load_track = []

for i in range(timesteps):
    action, states = model.predict(obs)
    print(obs)
    print(action)
    print('')
#     print(states)
    action_track.append(int(action))
    state_track.append(obs)
    obs, rewards, done, info = env.step(action)
    rewards_track.append(rewards)
    load_track.append(load)
    
    env.render()

state_track=np.array(state_track)

#translate actions into charging power
action_numbers = action_track
action_track=[env.get_charge(k) for k in action_track]


# In[35]:


action_numbers


# In[36]:


action_track


# In[37]:


# State definition
        # 0-->t : time slot
        # 1-->gen : PV generation at timeslot t
        # 2-->load : Load at timeslot t
        # 3--> SOC : State of charge
        # 4-->SOC_1 : State of charge t-1
        # 5-->tar : Electricity tariff at timeslot t
        # 6-->R  : total reward per episode
        # 7 --> sc: self-consumption
        # 8 --> r :reward


# In[38]:


rewards_track


# In[39]:


sum(rewards_track)


# In[40]:


# load_track


# # Plots

# In[41]:


flex.makeplot(48,state_track[:,3],action_track,env.data[:,0],env.data[:,1],env) # O Tempo e o Env não estão a fazer nada
# makeplot(T,soc,sol,gen,load,env): Tempo, SOC, Bat_Charge, Generation, load, env


# In[42]:


flex.reward_plot(env.R_Total)


# In[43]:


evaluate_policy(model,env, n_eval_episodes=10)


# # Studying the battery

# In[44]:


SOC = state_track[:,3]
bat_charge = action_track # Bat_charge
gen = env.data[:,0]
load = env.data[:,1]


# In[45]:


state_track


# In[46]:


def myplot(x):
    plt.figure(figsize=(10,6))
    plt.plot(x)


# In[47]:


env.data


# In[48]:


myplot(gen)
plt.ylabel('gen')


# In[49]:


gen


# In[50]:


myplot(load)
plt.ylabel('load')


# In[51]:


myplot(SOC)
plt.ylabel('SOC')


# In[52]:


myplot(bat_charge)
plt.ylabel('Battery Charge')


# In[53]:


myplot(rewards_track)
plt.ylabel('Rewards track')


# In[54]:


plt.figure(figsize=(10,7))
plt.legend
plt.plot(bat_charge,label='bat_charge')
plt.plot(gen,label='gen')
plt.legend()
plt.show()


# In[55]:


eta = 0.95
dh=30*(1/60)


# In[56]:


SOC


# In[57]:


gen


# In[58]:


C=np.array(bat_charge)*eta*dh
Sum=[]
Sumi = 0
for i in range(len(np.array(bat_charge)*eta*dh)):
    Sumi+= C[i]
    Sum.append(Sumi)
Sum


# In[59]:


SOC


# In[60]:


# bat_charge

charge_steps=np.linspace(-3,0,int((3/0.2)+1))
