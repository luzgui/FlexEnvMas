import gymnasium as gym
from gymnasium.wrappers import *
import numpy as np



class NormalizeObs(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space
        self.action_space=env.action_space
        
    def observation(self, obs):
        
        # for aid in self.agents_id:
        #     stats=self.env_config['data'].loc[aid].describe() #get stats from data for nomalization
            
        #     obs_dict={}
            
        #     for v,val in zip(self.state_vars.keys(),obs[aid]['observations']):
                
        #         if v=='tstep':
        #             obs_dict[v]=self.normalize(val, 0, self.T)
                    
        #         # elif v=='minutes':
        #         #     obs_dict[v]=self.normalize(val,0,self.min_max)
                    
        #         # elif 'gen' in v:
        #         #     obs_dict[v]=self.normalize(val,stats.loc['min']['gen'],self.min_max)
                    
        #         else:
        #             obs_dict[v]=val
            
                    
        #     obs[aid]['observations']=np.fromiter(obs_dict.values(), dtype=np.float32)
        #     # print(obs[aid]['observations'])
            
        return obs
    

    def normalize(self,val,min_val,max_val):
        return (val-min_val)/(max_val-min_val)







# menv_w=RelativePosition(menv)   



# obs=menv.get_env_obs()
# obs['ag1']['observations']
# assert len(obs_orig['ag1']['observations'])==len(menv.state_vars.keys())



# obs_dict={}
# for v,val in zip(menv.state_vars.keys(),obs['ag1']['observations']):
    
#     if v=='tstep':
#         obs_dict[v]=normalize(val, 0, menv.T)
        
#     else:
#         obs_dict[v]=val
        

# obs['ag1']['observations']=np.fromiter(obs_dict.values(), dtype='float32')
    
 
# stats.loc['min'][xx]