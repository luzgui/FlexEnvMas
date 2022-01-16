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
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.environments import suite_gym
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

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy.random as rnd
import time
import random as rnd

from time import perf_counter

import gym
import flexenv as flex

import auxfunctions as fun

cwd=os.getcwd()
datafolder=cwd + '/Data'


#%% Create the environment

#impor the data csv
env_data=pd.read_csv(datafolder + '/env_data.csv', header = None)


#make and check the environment
# Select the number of timesteps to consider
# timestesps=141
timestesps=47

#Create environmentn from Gym
flexenv=fun.make_env(env_data, load_num=4, timestep=timestesps, soc_max=5, eta=0.95, charge_lim=3)


#Convert to TF-agents environment
flexenv_tf = suite_gym.wrap_env(flexenv)
# Usually two environments are instantiated: one for training and one for evaluation. 
train_env = tf_py_environment.TFPyEnvironment(flexenv_tf)
eval_env = tf_py_environment.TFPyEnvironment(flexenv_tf)


#%% Networks

# At the heart of a DQN Agent is a `QNetwork`, a neural network model that can learn to predict `QValues` (expected returns) for all actions, given an observation from the environment.
# 
# We will use `tf_agents.networks.` to create a `QNetwork`. The network will consist of a sequence of `tf.keras.layers.Dense` layers, where the final layer will have 1 output for each possible action.

#Hyperparameters

num_iterations = 20000 # @param {type:"integer"}

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

#`agent.policy` — The main policy that is used for evaluation and deployment.
#`agent.collect_policy` — A second policy that is used for data collection.
 
eval_policy = agent.policy
collect_policy = agent.collect_policy


