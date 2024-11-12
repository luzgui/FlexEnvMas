import numpy as np

class Reward:
    def __init__(self, self_env):
        self.self_env = self_env  
        
        for key, value in self.self_env.com.scenarios_conf['reward_params'].items():
            setattr(self, key, value)  
            
        self.reward_func_list = {
              'sigma_reward': self.coop_sigma_reward,
              'tar_sigma_reward':self.tar_sigma_reward,
              'simple_reward':self.simple_reward}
    
        self.reward_func=self.reward_func_list[self.self_env.com.scenarios_conf['reward_func']]

    def get_penalty(self,agent):
            # if self.self_env.minutes == self.self_env.min_max-self.self_env.agents_params.loc[agent]['T_prof']*self.self_env.tstep_size and self.self_env.state.loc[agent]['y_s']  !=self.self_env.agents_params.loc[agent]['T_prof']:
                
                
            if self.self_env.tstep >= (self.self_env.tstep_init+18*4) and self.self_env.state.loc[agent]['y_s']  !=self.self_env.agents_params.loc[agent]['T_prof']:
                penalty=-5.0
                
            else: penalty=0
                
            return penalty
        
    def indicator(self,action):
        if action == 0:
            return 0
        else: 
            return 1
    
        
    def get_agent_cost(self, agent):
        
        agent_reward=-max(0,((self.self_env.action.loc[agent]['action']*self.self_env.com.agents[agent].apps[0].base_load*(self.self_env.tstep_size/60))-self.self_env.state.loc[agent]['excess0']))*self.self_env.state.loc[agent]['tar_buy']
                                
        return agent_reward+self.get_penalty(agent)
    
   
    def get_agent_cost_alpha(self, agent,alpha):
        
        agent_reward=-max(0,((self.self_env.action.loc[agent]['action']*self.self_env.com.agents[agent].apps[0].base_load*(self.self_env.tstep_size/60))-alpha*self.self_env.state.loc[agent]['excess0']))*self.self_env.state.loc[agent]['tar_buy']
                                                     
        return agent_reward+self.get_penalty(agent)
    

    
    def exponent(self,x):
        a=10
        b=1
        c=-2.3
        d=8.3
        
        return (a/(b+np.exp(x+c)))+d

    def custom_elu(self,x, c):
        # a=0.3
        # b=10
        # g=0.1
        if x >=c:
            r=self.g*(-(1/c) * x + 1)
        elif x<c:
            r=0            
            # r=a*(1/(1+np.exp(-b*(x-c)))-0.5) 
        return r
    
    def line(self,x,m,b):
        return m*x+b
    
    
    def beta(self,x):
        
        if 0 <= x <= 1:
            return 1
        
        elif x>1:
            return self.line(x, -1.5, 1)
    
    def coop_sigma_reward(self):
        
        df=self.self_env.action.copy()

        AgentTotalLoads=sum([self.self_env.action.loc[agent]['action']*self.self_env.com.agents[agent].apps[0].base_load*(self.self_env.tstep_size/60) for agent in self.self_env.agents_id])
        
        
        for ag in self.self_env.agents_id:
            action=self.self_env.action.loc[ag, 'action']
            base_load=self.self_env.com.agents[ag].apps[0].base_load
            df.loc[ag,'tar_min']=min(self.self_env.com.agents[ag].make_tariff())
            df.loc[ag,'c_min']=-df.loc[ag,'tar_min']*base_load*(self.self_env.tstep_size / 60)
            
            if AgentTotalLoads != 0:
                df.loc[ag, 'alpha'] = (action*base_load*(self.self_env.tstep_size / 60) / AgentTotalLoads)
            else:
                df.loc[ag, 'alpha'] = 0  # or handle it in another appropriate way

            df.loc[ag,'alpha_cost']=self.get_agent_cost_alpha(ag, df.loc[ag,'alpha'])
            
            
            df.loc[ag,'new_r']=df.loc[ag,'alpha_cost']+self.indicator(action)*self.custom_elu(df.loc[ag,'alpha_cost'],df.loc[ag,'c_min'])
            
        R=df['new_r'].sum()
        # print(df)
        return {aid: R for aid in self.self_env.agents_id} 
    
    
    
    def tar_sigma_reward(self):
        
        df=self.self_env.action.copy()

        AgentTotalLoads=sum([self.self_env.action.loc[agent]['action']*self.self_env.com.agents[agent].apps[0].base_load*(self.self_env.tstep_size/60) for agent in self.self_env.agents_id])
        
        
        for ag in self.self_env.agents_id:
            action=self.self_env.action.loc[ag, 'action']
            base_load=self.self_env.com.agents[ag].apps[0].base_load
            df.loc[ag,'tar_min']=min(self.self_env.com.agents[ag].make_tariff())
            df.loc[ag,'tar_max']=max(self.self_env.com.agents[ag].make_tariff())
            df.loc[ag,'c_min']=-df.loc[ag,'tar_min']*base_load*(self.self_env.tstep_size / 60)
            
            if AgentTotalLoads != 0:
                df.loc[ag, 'alpha'] = (action*base_load*(self.self_env.tstep_size / 60) / AgentTotalLoads)
            else:
                df.loc[ag, 'alpha'] = 0  # or handle it in another appropriate way

            df.loc[ag,'alpha_cost']=self.get_agent_cost_alpha(ag, df.loc[ag,'alpha'])
            
            df.loc[ag,'d_tar']=(self.self_env.state.loc[ag]['tar_buy']-df.loc[ag,'tar_min'])/(df.loc[ag,'tar_max']-df.loc[ag,'tar_min'])
            df.loc[ag,'x_ratio']=self.self_env.state.loc[ag]['pv_sum']/self.self_env.agents_params['E_prof'].sum()
            
            
            
            
            
            df.loc[ag,'new_r']=df.loc[ag,'alpha_cost']*self.line(df.loc[ag,'d_tar'],2,-1)*self.exponent(df.loc[ag,'x_ratio'])+self.indicator(action)*self.custom_elu(df.loc[ag,'alpha_cost'],df.loc[ag,'c_min'])
            
        R=df['new_r'].sum()
        print(df)
        return {aid: R for aid in self.self_env.agents_id} 
    
    
    
    
    def simple_reward(self):
        
        df=self.self_env.action.copy()

        AgentTotalLoads=sum([self.self_env.action.loc[agent]['action']*self.self_env.com.agents[agent].apps[0].base_load*(self.self_env.tstep_size/60) for agent in self.self_env.agents_id])
        
        
        for ag in self.self_env.agents_id:
            action=self.self_env.action.loc[ag, 'action']
            base_load=self.self_env.com.agents[ag].apps[0].base_load
            df.loc[ag,'tar_min']=min(self.self_env.com.agents[ag].make_tariff())
            df.loc[ag,'tar_max']=max(self.self_env.com.agents[ag].make_tariff())
            df.loc[ag,'c_min']=-df.loc[ag,'tar_min']*base_load*(self.self_env.tstep_size / 60)
            
            if AgentTotalLoads != 0:
                df.loc[ag, 'alpha'] = (action*base_load*(self.self_env.tstep_size / 60) / AgentTotalLoads)
            else:
                df.loc[ag, 'alpha'] = 0  # or handle it in another appropriate way

            df.loc[ag,'alpha_cost']=self.get_agent_cost_alpha(ag, df.loc[ag,'alpha'])
            
            df.loc[ag,'d_tar']=(self.self_env.state.loc[ag]['tar_buy']-df.loc[ag,'tar_min'])/(df.loc[ag,'tar_max']-df.loc[ag,'tar_min'])
            df.loc[ag,'x_ratio']=self.self_env.state.loc[ag]['pv_sum']/self.self_env.agents_params['E_prof'].sum()
            
            
            
            
            
            df.loc[ag,'new_r']=(df.loc[ag,'alpha_cost']+0.01)*self.indicator(action)
            
        R=df['new_r'].sum()
        # print(df)
        return {aid: R for aid in self.self_env.agents_id} 
    
    
