import numpy as np

class Reward:
    def __init__(self, self_env):
        self.self_env = self_env  # This allows the subcomponent to access attributes and methods of ClassA
    

    def get_penalty(self,agent):
            if self.self_env.minutes == self.self_env.min_max-self.self_env.agents_params.loc[agent]['T_prof']*self.self_env.tstep_size and self.self_env.state.loc[agent]['y_s']  !=self.self_env.agents_params.loc[agent]['T_prof']:
                penalty=-5.0
                
            else: penalty=0
                
            return penalty
        
    def indicator(self,action):
        if action == 0:
            return 1
        else: 
            return 0
    
        
    def get_agent_cost(self, agent):
        
        agent_reward=-max(0,((self.self_env.action.loc[agent]['action']*self.self_env.com.agents[agent].apps[0].base_load*(self.self_env.tstep_size/60))-self.self_env.state.loc[agent]['excess0']))*self.self_env.state.loc[agent]['tar_buy']
                                
        return agent_reward+self.get_penalty(agent)
    
   
    def get_agent_cost_alpha(self, agent,alpha):
        
        agent_reward=-max(0,((self.self_env.action.loc[agent]['action']*self.self_env.com.agents[agent].apps[0].base_load*(self.self_env.tstep_size/60))-alpha*self.self_env.state.loc[agent]['excess0']))*self.self_env.state.loc[agent]['tar_buy']
                                                     
        return agent_reward+self.get_penalty(agent)
    

    
    
    
    def coop_sigma_reward(self):
        
        df=self.self_env.action.copy()
        excess=self.self_env.state.loc[self.self_env.agents_id[0]]['excess0']
        AgentTotalLoads=sum([self.self_env.action.loc[agent]['action']*self.self_env.com.agents[agent].apps[0].base_load*(self.self_env.tstep_size/60) for agent in self.self_env.agents_id])
        
        
        
        for ag in self.self_env.agents_id:
            
            if AgentTotalLoads != 0:
                df.loc[ag, 'alpha'] = (self.self_env.action.loc[ag, 'action'] * 
                                        self.self_env.com.agents[ag].apps[0].base_load * 
                                        (self.self_env.tstep_size / 60) / AgentTotalLoads)
            else:
                df.loc[ag, 'alpha'] = 0  # or handle it in another appropriate way

            df.loc[ag,'cost']=self.get_agent_cost_alpha(ag, df.loc[ag,'alpha'])
            
            df.loc[ag,'sigma_1']=self.self_env.action.loc[ag,'action']*(-np.exp(-1.7*df.loc[ag,'alpha']*excess)+1)
            
        
        R=df['cost'].sum()+df['sigma_1'].sum()
        
        return {aid: R for aid in self.self_env.agents_id} #this adds the term of individual reward
    
    
    def sigma(self,x):
        c=-0.4
        d=-3.2
        a=100
        f=-0.01
        y=(c)/(d-np.exp(-a*(x-f)))
        return y
        
    
    def coop_sigma_reward_2(self):
        df=self.self_env.action.copy()
        # excess=self.self_env.state.loc[self.self_env.agents_id[0]]['excess0']
        AgentTotalLoads=sum([self.self_env.action.loc[agent]['action']*self.self_env.com.agents[agent].apps[0].base_load*(self.self_env.tstep_size/60) for agent in self.self_env.agents_id])
        
        
        for ag in self.self_env.agents_id:
            action=self.self_env.action.loc[ag, 'action']
            base_load=self.self_env.com.agents[ag].apps[0].base_load
            
            if AgentTotalLoads != 0:
                df.loc[ag, 'alpha'] = (action*base_load*(self.self_env.tstep_size / 60) / AgentTotalLoads)
            else:
                df.loc[ag, 'alpha'] = 0  # or handle it in another appropriate way

            df.loc[ag,'alpha_cost']=self.get_agent_cost_alpha(ag, df.loc[ag,'alpha'])
            
            df.loc[ag,'new_r']=self.indicator(action)*df.loc[ag,'alpha_cost']+(1-self.indicator(action))*self.sigma(df.loc[ag,'alpha_cost'])
            
            # df.loc[ag,'sigma_1']=self.self_env.action.loc[ag,'action']*(-np.exp(-1.7*df.loc[ag,'alpha']*excess)+1)
            
        R=df['new_r'].sum()
        # R=df['cost'].sum()+df['sigma_1'].sum()
        
        return {aid: R for aid in self.self_env.agents_id} #this adds the term of individual reward
    
        

            




#     def get_agent_reward(self, agent):
        # "Computes the reward for each agent as a float"
        
#         if self.com.scenarios_conf['reward_type'] == 'excess_cost_max':
#             # The reward should be function of the action
#             if self.minutes == self.min_max-self.agents_params.loc[agent]['T_prof']*self.tstep_size and self.state.loc[agent]['y_s']  !=self.agents_params.loc[agent]['T_prof']:
#                 agent_reward=-5.0
# #This agent specific reward must variable according to the situation (machines invlved and time horizon)
    
#             else:
                
#                 if self.com.scenarios_conf['game_setup'] == 'cooperative_colective':
#                     agent_reward=0
                    
#                 else:
                
#                     agent_reward=-max(0,((self.action.loc[agent]['action']*self.com.agents[agent].apps[0].base_load*(self.tstep_size/60))-self.state.loc[agent]['excess0']))*self.state.loc[agent]['tar_buy']
                                
#             return agent_reward
        
    # def get_agent_reward_alpha(self, agent,alpha):
    #     "Computes the reward for each agent considering sharing coeff as a float"
        
    #     if self.com.scenarios_conf['reward_type'] == 'excess_cost_max':
    #         # The reward should be function of the action
    #         if self.minutes == self.min_max-self.agents_params.loc[agent]['T_prof']*self.tstep_size and self.state.loc[agent]['y_s']  !=self.agents_params.loc[agent]['T_prof']:
    #             agent_reward=-5.0

    
    #         else:
    #                 agent_reward=-max(0,((self.action.loc[agent]['action']*self.com.agents[agent].apps[0].base_load*(self.tstep_size/60))-alpha*self.state.loc[agent]['excess0']))*self.state.loc[agent]['tar_buy']
                                
    #         return agent_reward
        
    
#     # def reward_shapping(self,agent):
#     #     return self.action.loc[agent]['action']*10*self.state.loc[agent]['excess0']
    
    
    
#     def get_env_reward(self):
#         '''
#         Returns the reward for each agent in the environment as a dictionary for algorithm processing. 
        
        
#         inspect self.com.scenarios_conf['game_setup'] for actual setup
        
#         - Cooperative: All agents get the same reward given by the sum of all agents rewards "
        
#         - Competitive: each agent has an individual reward
        
#         - Cooperative_colective: all agents get the same reward given by the collective purchase of energy from the grid (all loads are summed and the excess is collective)
#         '''
        
        
#         if self.com.scenarios_conf['game_setup'] == 'cooperative':
#         #cooperative // common reward
#             R=sum([self.get_agent_reward(aid) for aid in self.agents_id])
#             return {aid: R for aid in self.agents_id}
        
#         #Competitive / individual rewards
#         elif self.com.scenarios_conf['game_setup'] == 'competitive':
#             return {aid:self.get_agent_reward(aid) for aid in self.agents_id}
        
#         elif self.com.scenarios_conf['game_setup'] == 'cooperative_colective':
#             # this is the real cost of collective energy consumption. it is centralized information since it sums all the loads and subtracts the excess            
#             AgentLoads=[self.action.loc[agent]['action']*self.com.agents[agent].apps[0].base_load*(self.tstep_size/60) for agent in self.agents_id]
            
#             #introduce a penalty for violating conditions
#             penalty_table=[]
#             for aid in self.agents_id:
#                 if self.minutes == self.min_max-self.agents_params.loc[aid]['T_prof']*self.tstep_size and self.state.loc[aid]['y_s']  != self.agents_params.loc[aid]['T_prof']: #if arrived at the last possible timeslot for connecting app and you havent connceted then there is a penalty
#                     penalty_table.append(True)
            
#             penalty=-5*any(penalty_table) #a common penalty -5 is imposed if any agent violates the constraints
                    
#             # this reward is considering that the excess infromation is the same for all agents!        
#             R=-max(0,(sum(AgentLoads)-self.state.loc[self.agents_id[0]]['excess0']))*self.state.loc[self.agents_id[0]]['tar_buy']
#             # print('pen:', penalty)
#             # print('R:', R)
#             # return {aid: R+self.get_agent_reward(aid) for aid in self.agents_id} #this adds the term of individual reward
            
#             return {aid: R+penalty for aid in self.agents_id} #this adds the term of individual reward
        
#         elif self.com.scenarios_conf['game_setup'] == 'cooperative_colective_boost':
#             # this is the real cost of collective energy consumption. it is centralized information since it sums all the loads and subtracts the excess            
#             AppLoads=[self.action.loc[agent]['action']*self.com.agents[agent].apps[0].base_load*(self.tstep_size/60) for agent in self.agents_id]
            
#             BaseLoads=[self.state.loc[agent]['load0'] for agent in self.agents_id]
            
#             Balance=(sum(AppLoads)+sum(BaseLoads))-self.state.loc[self.agents_id[0]]['gen0']
#             #introduce a penalty for violating conditions
#             penalty_table=[]
#             for aid in self.agents_id:
#                 if self.minutes == self.min_max-self.agents_params.loc[aid]['T_prof']*self.tstep_size and self.state.loc[aid]['y_s']  != self.agents_params.loc[aid]['T_prof']: #if arrived at the last possible timeslot for connecting app and you havent connceted then there is a penalty
#                     penalty_table.append(True)
            
#             penalty=-5*any(penalty_table) #a common penalty -5 is imposed if any agent violates the constraints
                    
#             # this reward is considering that the excess infromation is the same for all agents!        
#             R=-max(0,Balance)*self.state.loc[self.agents_id[0]]['tar_buy']
#             # print('pen:', penalty)
#             # print('R:', R)
#             # return {aid: R+self.get_agent_reward(aid) for aid in self.agents_id} #this adds the term of individual reward
            
#             return {aid: R+penalty for aid in self.agents_id} #this adds the term of individual reward
        
#         elif self.com.scenarios_conf['game_setup'] == 'cooperative_colective_sigma':          
#             AgentLoads=[self.action.loc[agent]['action']*self.com.agents[agent].apps[0].base_load*(self.tstep_size/60) for agent in self.agents_id]
            
#             #introduce a penalty for violating conditions
#             penalty_table=[]
#             for aid in self.agents_id:
#                 if self.minutes == self.min_max-self.agents_params.loc[aid]['T_prof']*self.tstep_size and self.state.loc[aid]['y_s']  != self.agents_params.loc[aid]['T_prof']: #if arrived at the last possible timeslot for connecting app and you havent connceted then there is a penalty
#                     penalty_table.append(True)
            
#             penalty=-5*any(penalty_table) #a common penalty -5 is imposed if any agent violates the constraints
                    
#             # this reward is considering that the excess infromation is the same for all agents!        
#             actions=sum([self.action.loc[agent]['action'] for agent in self.agents_id])
            
#             sigma=-np.exp(-1.7*self.state.loc[self.agents_id[0]]['excess0'])+1
            
#             R=-max(0,(sum(AgentLoads)-self.state.loc[self.agents_id[0]]['excess0']))*self.state.loc[self.agents_id[0]]['tar_buy']+actions*sigma
#             # print('pen:', penalty)
#             # print('R:', R)
#             # return {aid: R+self.get_agent_reward(aid) for aid in self.agents_id} #this adds the term of individual reward
            
#             return {aid: R+penalty for aid in self.agents_id} #this adds the term of individual reward
        
#         elif self.com.scenarios_conf['game_setup'] == 'cooperative_colective_scaled_sigma':          
#             df=self.action.copy()
#             excess=self.state.loc[self.agents_id[0]]['excess0']
#             AgentTotalLoads=sum([self.action.loc[agent]['action']*self.com.agents[agent].apps[0].base_load*(self.tstep_size/60) for agent in self.agents_id])
            
            
            
#             for ag in self.agents_id:
                
#                 if AgentTotalLoads != 0:
#                     df.loc[ag, 'alpha'] = (self.action.loc[ag, 'action'] * 
#                                             self.com.agents[ag].apps[0].base_load * 
#                                             (self.tstep_size / 60) / AgentTotalLoads)
#                 else:
#                     df.loc[ag, 'alpha'] = 0  # or handle it in another appropriate way

#                 df.loc[ag,'cost']=self.get_agent_reward_alpha(ag, df.loc[ag,'alpha'])
                
#                 df.loc[ag,'sigma_1']=self.action.loc[ag,'action']*(-np.exp(-1.7*df.loc[ag,'alpha']*excess)+1)
                
            
#             R=df['cost'].sum()+df['sigma_1'].sum()
            
#             return {aid: R for aid in self.agents_id} #this adds the term of individual reward
        
#         elif self.com.scenarios_conf['game_setup'] == 'cooperative_colective_scaled_sigma2':        
#             df=self.action.copy()
#             excess=self.state.loc[self.agents_id[0]]['excess0']
#             AgentTotalLoads=sum([self.action.loc[agent]['action']*self.com.agents[agent].apps[0].base_load*(self.tstep_size/60) for agent in self.agents_id])
            
            
            
#             for ag in self.agents_id:
                
#                 if AgentTotalLoads != 0:
#                     df.loc[ag, 'alpha'] = (self.action.loc[ag, 'action'] * 
#                                             self.com.agents[ag].apps[0].base_load * 
#                                             (self.tstep_size / 60) / AgentTotalLoads)
#                 else:
#                     df.loc[ag, 'alpha'] = 0  # or handle it in another appropriate way

#                 df.loc[ag,'cost']=self.get_agent_reward_alpha(ag, df.loc[ag,'alpha'])
                
#                 df.loc[ag,'sigma_1']=self.action.loc[ag,'action']*(-np.exp(-1.7*df.loc[ag,'alpha']*excess)+1)
                
            
#             R=df['cost'].sum()+df['sigma_1'].sum()
            
#             return {aid: R for aid in self.agents_id} #this adds the term of individual reward


