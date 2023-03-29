# this should be a class!!!


from ray.rllib.algorithms.ppo import PPO #trainer
from ray.rllib.algorithms.ppo import PPOConfig #config
import pandas as pd
import numpy as np
import random
from ray import tune
from auxfunctions_CC import *
from termcolor import colored

import cProfile

def trainable_mas(config):
    
    n_iters=2
    checkpoint_freq=50
    
    # trainer=PPO(config, env=config["env"])
    trainer=CentralizedCritic(config)
    print(colored('Trainer created...','red'))
    
    
    weights={}
    
    #set the seed
    seed=config['seed']
    np.random.seed(seed)
    random.seed(seed)    
    
    for i in range(n_iters):
        print(colored('training...','red'))
        train_results=trainer.train()
        # train_results=cProfile.run('trainer.train()')
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
        # for k, v in trainer.get_policy().get_weights().items():
        #             weights["FCC/{}".format(k)] = v
        
        #save checkpoint every checkpoint_freq
        # if i % checkpoint_freq == 0: 
        #     print(colored('Saving checkpoint...','green'))
        #     checkpoint=trainer.save(tune.get_trial_dir())
        
        
        
        checkpoint=trainer.save(tune.get_trial_dir())
        print(colored('Checkpoint saved...','green'))
        #evaluate agent
        # print('evaluating...')
        # # eval_results=trainer.evaluate()
        # eval_metrics={'episode_reward_max', 
        #       'episode_reward_mean',
        #       'episode_reward_min',}
        # eval_logs={'evaluation':{}}
        # eval_logs['evaluation']={k: eval_results['evaluation'][k] for k in eval_metrics}
        
        # results={**logs,**weights,**eval_logs}
        # print(colored('Results...','green'))
        results={**logs}
        print(colored('Reporting...','green'))
        tune.report(results)
        
    trainer.stop()







n_iters=200
checkpoint_freq=10

def trainable(config):
    
    trainer=PPO(config, env=config["env"])
    weights={}
    
    #set the seed
    seed=config['seed']
    np.random.seed(seed)
    random.seed(seed)    
    
    for i in range(n_iters):
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

        #save checkpoint every checkpoint_freq
        if i % checkpoint_freq == 0: 
            checkpoint=trainer.save(tune.get_trial_dir())
        
        #evaluate agent
        print('evaluating...')
        # eval_results=trainer.evaluate()
        eval_metrics={'episode_reward_max', 
              'episode_reward_mean',
              'episode_reward_min',}
        eval_logs={'evaluation':{}}
        # eval_logs['evaluation']={k: eval_results['evaluation'][k] for k in eval_metrics}
        
        results={**logs,**weights,**eval_logs}
        # results={**eval_logs}
        tune.report(results)
        
    trainer.stop()





