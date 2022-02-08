"""
Created on Thu Jan 13 16:48:57 2022

@author: GuiLuz
"""

from __future__ import absolute_import, division, print_function
import base64
import imageio
import IPython
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import pyvirtualdisplay
import reverb
import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import py_driver
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.environments import suite_gym
from tf_agents.environments import validate_py_environment
from tf_agents.environments import TFEnvironment
from tf_agents.environments import batched_py_environment

from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import sequential
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.utils import common



# import numpy as np
import pandas as pd
import os
# import matplotlib.pyplot as plt
import numpy.random as rnd
import time
# import random as rnd

from time import perf_counter

import gym
import flexenv as flex

# from Scripts Antonio import A_auxfunctions as fun

import sys

cwd=os.getcwd()
datafolder=cwd + '/Data'

sys.path.append(cwd+ '/Scripts Antonio')
import A_auxfunctions as fun


#Add this folder to path


#%% Create the environment

#impor the data csv
env_data=pd.read_csv(datafolder + '/env_data.csv', header = None)


#make and check the environment
# Select the number of timesteps to consider
# timestesps=141
timesteps=47

#Create environmentn from Gym
flexenv=fun.make_env(env_data, load_num=2, timestep=timesteps, soc_max=4, eta=0.95, charge_lim=2, min_charge_step=0.02, reward_type=2)

#Convert to TF-agents environment
flexenv_tf = suite_gym.wrap_env(flexenv)



# Usually two environments are instantiated: one for training and one for evaluation. 
train_env = tf_py_environment.TFPyEnvironment(flexenv_tf)
eval_env = tf_py_environment.TFPyEnvironment(flexenv_tf)

# train_env=tf_py_environment.batched_py_environment.BatchedPyEnvironment([flexenv_tf])
# eval_env=tf_py_environment.batched_py_environment.BatchedPyEnvironment([flexenv_tf])



# 





#%% Networks

# At the heart of a DQN Agent is a `QNetwork`, a neural network model that can learn to predict `QValues` (expected returns) for all actions, given an observation from the environment.
# 
# We will use `tf_agents.networks.` to create a `QNetwork`. The network will consist of a sequence of `tf.keras.layers.Dense` layers, where the final layer will have 1 output for each possible action.

#Hyperparameters

num_iterations = 100 # @param {type:"integer"}

initial_collect_steps = 100  # @param {type:"integer"}
collect_steps_per_iteration =   1# @param {type:"integer"}
replay_buffer_max_length = 100000  # @param {type:"integer"}

batch_size = 64  # @param {type:"integer"}
learning_rate = 1e-3  # @param {type:"number"}
log_interval = 200  # @param {type:"integer"}

num_eval_episodes = 10  # @param {type:"integer"}
eval_interval = 1000  # @param {type:"integer"}

#Network

fc_layer_params = (100, 50)
action_tensor_spec = tensor_spec.from_spec(flexenv_tf.action_spec())
num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1

# Define a helper function to create Dense layers configured with the right
# activation and kernel initializer.
def dense_layer(num_units):
  return tf.keras.layers.Dense(
      num_units,
      activation=tf.keras.activations.relu,
      kernel_initializer=tf.keras.initializers.VarianceScaling(
          scale=2.0, mode='fan_in', distribution='truncated_normal'))

# QNetwork consists of a sequence of Dense layers followed by a dense layer
# with `num_actions` units to generate one q_value per available action as
# its output.

dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]#A list with two dense layers

q_values_layer = tf.keras.layers.Dense(
    num_actions,
    activation=None,
    kernel_initializer=tf.keras.initializers.RandomUniform(
        minval=-0.03, maxval=0.03),
    bias_initializer=tf.keras.initializers.Constant(-0.2))


q_net = sequential.Sequential(dense_layers + [q_values_layer], name='qnetwork')


# Now use `tf_agents.agents.dqn.dqn_agent` to instantiate a `DqnAgent`. In addition to the `time_step_spec`, `action_spec` and the QNetwork, the agent constructor also requires an optimizer (in this case, `AdamOptimizer`), a loss function, and an integer step counter.

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

train_step_counter = tf.Variable(0)

agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter)

agent.initialize()

#A summary of the q-network
q_net.summary()


#%%Policies

 
eval_policy = agent.policy #The main policy that is used for evaluation and deployment.
collect_policy = agent.collect_policy #A second policy that is used for data collection.



random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                train_env.action_spec())


# To get an action from a policy, call the `policy.action(time_step)` method. The `time_step` contains the observation from the environment. This method returns a `PolicyStep`, which is a named tuple with three components:
# 
# -   `action` — the action to be taken (in this case, `0` or `1`)
# -   `state` — used for stateful (that is, RNN-based) policies
# -   `info` — auxiliary data, such as log probabilities of actions




#%% ## Metrics and Evaluation
# 
# The most common metric used to evaluate a policy is the average return. The return is the sum of rewards obtained while running a policy in an environment for an episode. Several episodes are run, creating an average return.
# 
# The following function computes the average return of a policy, given the policy, environment, and a number of episodes.
# 

def compute_avg_return(environment, policy, num_episodes=10):

  total_return = 0.0
  for _ in range(num_episodes):

    time_step = environment.reset()
    episode_return = 0.0

    while not time_step.is_last():
      action_step = policy.action(time_step)
      time_step = environment.step(action_step.action)
      episode_return += time_step.reward
    total_return += episode_return

  avg_return = total_return / num_episodes
  return avg_return.numpy()[0]


# See also the metrics module for standard implementations of different metrics.
# https://github.com/tensorflow/agents/tree/master/tf_agents/metrics


# Running this computation on the `random_policy` shows a baseline performance in the environment.

compute_avg_return(eval_env, random_policy, num_eval_episodes)



#%% ## Replay Buffer
# 
# In order to keep track of the data collected from the environment, we will use [Reverb](https://deepmind.com/research/open-source/Reverb), an efficient, extensible, and easy-to-use replay system by Deepmind. It stores experience data when we collect trajectories and is consumed during training.
# 
# This replay buffer is constructed using specs describing the tensors that are to be stored, which can be obtained from the agent using agent.collect_data_spec.
# 

table_name = 'uniform_table'
replay_buffer_signature = tensor_spec.from_spec(
      agent.collect_data_spec)
replay_buffer_signature = tensor_spec.add_outer_dim(
    replay_buffer_signature)

table = reverb.Table(
    table_name,
    max_size=replay_buffer_max_length,
    sampler=reverb.selectors.Uniform(),
    remover=reverb.selectors.Fifo(),
    rate_limiter=reverb.rate_limiters.MinSize(1),
    signature=replay_buffer_signature)

reverb_server = reverb.Server([table])

replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
    agent.collect_data_spec,
    table_name=table_name,
    sequence_length=2,
    local_server=reverb_server)

rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
  replay_buffer.py_client,
  table_name,
  sequence_length=1000)

# For most agents, `collect_data_spec` is a named tuple called `Trajectory`, containing the specs for observations, actions, rewards, and other items.


# ## Data Collection
# 
# Now execute the random policy in the environment for a few steps, recording the data in the replay buffer.
# 
# Here we are using 'PyDriver' to run the experience collecting loop. You can learn more about TF Agents driver in our [drivers tutorial](https://www.tensorflow.org/agents/tutorials/4_drivers_tutorial).



#@test {"skip": true}
# py_driver.PyDriver(
#     flexenv_tf,
#     py_tf_eager_policy.PyTFEagerPolicy(agent.collect_policy, use_tf_function=True), [rb_observer], max_steps=initial_collect_steps).run(flexenv_tf.reset())


# The replay buffer is now a collection of Trajectories.

driver = dynamic_episode_driver.DynamicEpisodeDriver(
    train_env, agent.collect_policy, [rb_observer], num_episodes=100)



# For the curious:
# Uncomment to peel one of these off and inspect it.
# iter(replay_buffer.as_dataset()).next()


# The agent needs access to the replay buffer. This is provided by creating an iterable `tf.data.Dataset` pipeline which will feed data to the agent.
# 
# Each row of the replay buffer only stores a single observation step. But since the DQN Agent needs both the current and next observation to compute the loss, the dataset pipeline will sample two adjacent rows for each item in the batch (`num_steps=2`).
# 
# This dataset is also optimized by running parallel calls and prefetching data.

# Dataset generates trajectories with shape [Bx2x...]
dataset = replay_buffer.as_dataset(
    num_parallel_calls=3,
    sample_batch_size=batch_size,
    num_steps=2).prefetch(3)

dataset


iterator = iter(dataset)
print(iterator)

# For the curious:
# Uncomment to see what the dataset iterator is feeding to the agent.
# Compare this representation of replay data 
# to the collection of individual trajectories shown earlier.

# iterator.next()


# ## Training the agent
# 
# Two things must happen during the training loop:
# 
# -   collect data from the environment
# -   use that data to train the agent's neural network(s)
# 
# This example also periodicially evaluates the policy and prints the current score.
# 
# The following will take ~5 minutes to run.

#@test {"skip": true}
try:
  get_ipython().run_line_magic('time', '')
except:
  pass

# (Optional) Optimize by wrapping some of the code in a graph using TF function.
agent.train = common.function(agent.train)

# Reset the train step.
agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
returns = [avg_return]

# Reset the environment.
time_step = train_env.reset()

# Create a driver to collect experience.
# collect_driver = py_driver.PyDriver(
#     train_env,
#     py_tf_eager_policy.PyTFEagerPolicy(agent.collect_policy, use_tf_function=True),[rb_observer], max_steps=collect_steps_per_iteration)


collect_driver= dynamic_episode_driver.DynamicEpisodeDriver(
    eval_env, agent.collect_policy, [rb_observer], num_episodes=10)


for _ in range(num_iterations):

  # Collect a few steps and save to the replay buffer.
  # time_step, _ = collect_driver.run(time_step)
  time_step, _ = collect_driver.run(time_step)
  print(time_step)
  # Sample a batch of data from the buffer and update the agent's network.
  experience, unused_info = next(iterator)
  train_loss = agent.train(experience).loss

  step = agent.train_step_counter.numpy()

  # if step % log_interval == 0:
  #   print('step = {0}: loss = {1}'.format(step, train_loss))

  # if step % eval_interval == 0:
  #   avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
  #   print('step = {0}: Average Return = {1}'.format(step, avg_return))
  #   returns.append(avg_return)

