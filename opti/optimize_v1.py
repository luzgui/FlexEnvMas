#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 16:22:51 2026

@author: omega
"""
from __future__ import division
from pyomo.environ import *
from pyomo.opt import SolverFactory
import pandas as pd
import numpy as np
import random

from utils.utilities import utilities
from utils.dataprocessor import OptiDataPostProcessor

import os
from os import path
from termcolor import colored
import math

from opti.optimize import CommunityOptiModel

class CommunityOptiModelV1(CommunityOptiModel):
    def __init__(self,env,folder):
        super().__init__(env,folder)
        
        
    def make_model(self,t_init):
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
        
        # H=self.env.Tw
        _, data=self.get_one_day_env_data(t_init)
        H=len(data.loc['ag1'])
        nI=self.env.com.num_agents
        
        d={}
        l={}
        for index, agent_id in enumerate(self.env.com.agents):
            agent=self.env.com.agents[agent_id]
            utilities.print_info('convertion from power to energy of shiftable appliance is hardcoded: solve this')
            l[index]=agent.apps[0].base_load*0.25
            d[index]=int(agent.apps[0].duration/self.env.tstep_size)
            # import pdb
            # pdb.pdb.set_trace()
        
        
        
        utilities.print_info('Theres a BUG here when using the tariff for agent 1 for all agents. Need to solve it')
        c=self.env.com.agents['ag1'].tariff[0:H] #bug
        
        #data only for the day we are instatiating the model
        
        
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
        # def Constxy(m,i,t):
        #     # for t in m.Ts:
        # #        if t >= m.d[i] or t <= H-m.d[i]:
        #             return sum(m.x[i,t+k] for k in range(0,m.d[i]))\
        #         >=m.d[i]*m.y[i,t]
            
        # m.xy_constraint = Constraint(m.I,m.Ts,rule=Constxy)
        
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
        