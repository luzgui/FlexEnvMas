import pandas as pd
from auxfunctions_shiftenv import *


def make_env_config(datafolder):

    #%% Make Shiftable loads environment
    #import raw data
    # data_raw=pd.read_csv(datafolder + '/issda_data_halfyear.csv')
    
    # data_raw_cons=pd.read_excel(datafolder + '/Dataset_gecad.xlsx', 'Total Consumers')
    data_raw_prod=pd.read_excel(datafolder + '/Dataset_gecad.xlsx', 'Total Producers')
    
    
    data = get_raw_data('Dataset_gecad.xlsx', datafolder)
    # data=data_raw[['minutes','PV0','Ag0','Ag1']]
    
    
    tstep_size=15 # number of minutes in each timestep
    # %% convert to env data
    tstep_per_day=96 #number of timesteps per day
    num_days=7 #number of days
    # timesteps=tstep_per_day*num_days #number of timesteps to feed the agent
    timesteps=len(data)-1
    
    
    # load_id=['ag1','ag2']
    load_id=['ag1']
    
    
    #%% Make env data
    num_agents=len(load_id)
    agents_id=load_id
    
    # agents_id=['ag'+str(k) for k in range(num_agents)]
    #What are agents data?
    
    
    env_data=make_env_data_mas(data, timesteps, load_id, 4, num_agents,agents_id)
    
    ## Shiftable profile example
    AgentsProfiles=np.array([[1.2,1.2,1.2,1.2,1.2,1.2,1.2],
                       [1.5,1.5,1.5,1.5,1.5],
                       [0.6,0.6,0.6,0.6,0.6],
                       [0.9,0.9,0.9,0.9,0.9]], dtype=object)
    
    shiftprof={agent:profile for (agent,profile) in zip(agents_id,AgentsProfiles)}
    
    #Agents delivery times
    delivery_times={ag:37*tstep_size for ag in agents_id }
    
    
    #%% make env config
    reward_type='excess_cost_max'
    
    
    env_config={"step_size": tstep_size,
                'window_size':tstep_per_day,
                'tstep_per_day':tstep_per_day,
                "data": env_data,
                "reward_type": reward_type, 
                "profile": shiftprof, 
                "time_deliver": delivery_times, 
                'done_condition': 'mode_window',
                'init_condition': 'mode_window',
                'tar_type':'bi',
                'env_info': 'training environment',
                'num_agents':num_agents,
                'agents_id':agents_id}
    
    
    return env_config