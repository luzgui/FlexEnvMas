#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 17:26:54 2024

@author: omega
"""
from __future__ import division
from pyomo.environ import *
from pyomo.opt import SolverFactory
import pandas as pd
import numpy as np
import random

from utilities import utilities
from dataprocessor import OptiDataPostProcessor


class CommunityOptiModel():
    def __init__(self,env):
        """
        Environment must be the testing environment 
        """
        self.env=env
        self.processor=OptiDataPostProcessor(env)
    
    
    def solve_model(self):
        opt = SolverFactory('gurobi')
        self.results=opt.solve(self.model,tee=True)
        return self.get_solution()
        
    def solve_model_yearly(self):
        # total_num_days=len(self.env.allowed_inits)
        total_num_days=10
        solutions=[]
        for day in range(total_num_days):
            self.make_model(day)
            times=self.get_one_day_env_data(day).loc['ag1']['minutes']
            sol=self.solve_model()
            sol['day']=day
            # import pdb
            # pdb.pdb.set_trace()
            # sol['minutes']=times.values
            sol.insert(0,'minutes',times.values)
            sol = sol.set_index(times.index)
            solutions.append(sol)
        result_df = pd.concat(solutions, ignore_index=True)
        # save it to the correct place
        return result_df
            
    
    def make_model(self,day_num):
        """
        Centralized shiftable loads allocation model
        
        H: number of timeslots per day ( Horizon)
        nI: Number of agents/shiftable loads
        d0: Duration of each agent/load (dict of the format 0 : val, 1:val ... nI-1:val)
        p0: Constant load of each agent/load (dict of the format 0 : val, 1:val ... nI-1:val)
        c: tariff vector (vector)
        miu: power-energy convertion factor    
        Viol: Allowed violation of PV usage per timestep (vector)
        Ppv: vector with PV production (vector)
        """
        
        H=self.env.Tw
        nI=self.env.com.num_agents
        
        d={}
        l={}
        for index, agent_id in enumerate(self.env.com.agents):
            agent=self.env.com.agents[agent_id]
            utilities.print_info('convertion from power to energy is hardcoded: solve this')
            l[index]=agent.apps[0].base_load*0.25
            d[index]=int(agent.apps[0].duration/self.env.tstep_size)
            # import pdb
            # pdb.pdb.set_trace()
        
        
        
        utilities.print_info('Theres a BUG here when using the tariff for agent 1 for all agents. Need to solve it')
        c=self.env.com.agents['ag1'].tariff #bug
        
        #data only for the day we are instatiating the model
        data=self.get_one_day_env_data(day_num)
        
        Ppv=data.loc['ag1']['gen'].values #bug
        
        baseload=data.groupby(level=1).sum()
        baseload=baseload['load'].values
        
        
        
        m = ConcreteModel()
        # SETS
        m.T = RangeSet(0,H-1)
        
        m.I = RangeSet(0,nI-1)
        
        
        m.c=Param(m.T, initialize=c)
        
        m.pv=Param(m.T, initialize=Ppv)
        
        m.baseload=Param(m.T, initialize=baseload)
        
        m.l=Param(m.I,initialize=l)
        
        m.d=Param(m.I,initialize=d)
        
        m.H=Param(initialize=H)
        
        # m.miu=Param(m.I, initialize=miu)
        
        def BuildTs(model,nI):
            for i in range(nI):
                    # import pdb
                    # pdb.pdb.set_trace()
                    return range(int(model.d[i]),int(H-model.d[i]))
                
        m.Ts = Set(initialize=BuildTs(m,nI))
        # m.Ts = RangeSet(d0,H-d0)
                
        
        # VARIABLES
        
        # Starting variable
        m.y = Var(m.I,m.T,domain=Binary, initialize=0)
        # Activity variable
        m.x = Var(m.I,m.T,domain=Binary, initialize=0)
        
        m.P = Var(m.I,m.T,domain=PositiveReals, initialize=1)
        
        m.Px = Var(m.T,domain=Reals, initialize=1)
        
        m.C = Var(m.T,domain=Reals, initialize=1)
        
        m.C_pos = Var(m.T,domain=NonNegativeReals, initialize=0)
        m.C_neg = Var(m.T,domain=NonPositiveReals, initialize=0)
        # m.C_neg = Var(m.T,domain=NegativeReals, initialize=-0.1)
        m.sign =  Var(m.T,domain=Binary, initialize=0)
        
        ## CONSTRAINTS
        def Consty(m,i,t):
            return sum(m.y[i,t] for t in m.Ts) == 1
        m.y_constraint = Constraint(m.I,m.Ts,rule=Consty)
        
        #
        def Constxy(m,i,t):
            # for t in m.Ts:
        #        if t >= m.d[i] or t <= H-m.d[i]:
                    return sum(m.x[i,t+k] for k in range(0,m.d[i]))\
                >=m.d[i]*m.y[i,t]
            
        m.xy_constraint = Constraint(m.I,m.Ts,rule=Constxy)
        
        #shiftable load per agent per tstep
        def ConstP(m,i,t):
            return m.P[i,t] == m.x[i,t]*m.l[i]
        m.lConstraint = Constraint(m.I,m.T, rule=ConstP) 
        
        
        #compute the total PV excess per timestep
        def Excess(m,t):
            # import pdb
            # pdb.pdb.set_trace()
            return m.Px[t] == max(0,m.pv[t]-m.baseload[t])
        m.excess_constraint = Constraint(m.T, rule=Excess)
        
        # def P_T(m,t):
            
        
        #constraints total shiftable load to be under the pv production
        # it must be relaxed to allow for shedulling when there is no PV production
        # def ConstTotal(m,t):
        #     return sum(m.x[i,t]*m.p[i] for i in m.I) <= m.pv[t]+Viol[t]
        # m.TotalConstraint = Constraint(m.T, rule=ConstTotal) 
        
        
        def Constx(m,i,t):    
            return sum(m.x[i,t] for t in m.T) == m.d[i]
        m.x_constraint = Constraint(m.I,m.T,rule=Constx) 
        
        
        def Cost(m,t):
            return m.C[t]==(sum(m.P[i,t] for i in m.I)-m.Px[t])*m.c[t]
        m.cost_constraint = Constraint(m.T,rule=Cost)
        
        def Cost_decompose(m,t):
            return m.C[t] == m.C_pos[t]*m.sign[t]+m.C_neg[t]*(1-m.sign[t])
        m.pos_cost_constraint = Constraint(m.T,rule=Cost_decompose)
             
        
        #OBJECTIVE
        # def MinCost(m):
        #     return sum(sum(m.x[i,t]*m.c[t]*m.l[i] for t in m.T)\
        #                for i in m.I) 
        # m.objective = Objective(rule=MinCost)
        
        def MinCost(m):
            # R=sum((sum(m.P[i,t] for i in m.I)-m.Px[t])*m.c[t] for t in m)
            # return max(0,m.C)
            # return m.C
            # return sum((sum(m.P[i,t] for i in m.I)-m.Px[t])*m.c[t] for t in m.T)
            return sum(m.C_pos[t] for t in m.T)
        m.objective = Objective(rule=MinCost)


        self.model=m
    
    
    def get_solution(self):
        #importing values
        df=pd.DataFrame()
        for (aid,n) in zip(self.env.agents_id,range(len(self.model.I))):
            
            P=np.array([value(self.model.P[n,t]) for t in self.model.T])
            x=np.array([value(self.model.x[n,t]) for t in self.model.T])
            y=np.array([value(self.model.y[n,t]) for t in self.model.T])
            
            df['shift_'+aid]=P
            df['action_'+aid]=x
            df['y_act_'+aid]=y
        
        #global    
        pv=np.array([value(self.model.pv[t]) for t in self.model.T])
        px=np.array([value(self.model.Px[t]) for t in self.model.T])
        tar=np.array([value(self.model.c[t]) for t in self.model.T])
        base=np.array([value(self.model.baseload[t]) for t in self.model.T])
        
        C_pos = np.array([value(self.model.C_pos[t]) for t in self.model.T])
        C_neg = np.array([value(self.model.C_neg[t]) for t in self.model.T])
        C = np.array([value(self.model.C[t]) for t in self.model.T])
        sign = np.array([value(self.model.sign[t]) for t in self.model.T])                   
        
        df['gen0_ag1']=pv
        df['tar_buy']=tar
        
        df['shift_T']=df[[k for k in df.columns if 'shift_ag' in k]].sum(axis=1)
        df['Excess']=px
        df['baseload_T']=base
        df['C_pos']=C_pos
        df['Cost_shift_T']=C_pos
        df['C_neg']=C_neg
        df['C']=C
        df['sign']=sign
            
            # c=np.array([value(model.c[t]) for t in model.T])
            # H=len(c)
            # nI=len(model.I)
            # agents=np.array([value(model.I[t]) for t in model.I])
            
        return df
    
    
    def get_one_day_timeslot(self,day_num):
            """reurn the inital timeslot for a day number in the year"""
            w=self.env.Tw #this is hardcoded for 15 minutes resolution 1 day horizon
            # return self.env.data.iloc[day_num*w:(day_num+1)*w]
            return w*day_num
        
    def get_one_day_env_data(self,day_num):
         t_init=self.get_one_day_timeslot(day_num)
         t_end=t_init+self.env.Tw
         return self.env.com.com_data.loc[:, slice(t_init, t_end-1), :]
         