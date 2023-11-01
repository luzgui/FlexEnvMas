from __future__ import division
from pyomo.environ import *
from pyomo.opt import SolverFactory
import pandas as pd
import numpy as np
import random


## alocating shiftable loads ##

def Agent(H,d,c,miu,profile,base):
    """
    Alocation of individual shiftable loads with variable profile
    """
    
    
    m = ConcreteModel()

    m.d=Param(initialize=len(profile))
    m.Td=RangeSet(0,m.d-1) # cicle set
    # SETS
    m.T = RangeSet(0,H-1)
    # m.Ts = RangeSet(0,H-m.d-1)
    
    
    
    m.b=Param(m.T, initialize=base)
    
    m.c=Param(m.T, initialize=c)
    
    # m.p=Param(initialize=p)
    
    
    #shiftable device
    
    # m.Td=RangeSet(0,m.d-1)
    m.profile=Param(m.Td,initialize=profile)
    
    
    # VARIABLES
    # Starting variable
    m.y = Var(m.T,domain=Binary, initialize=0) 
    # Activity variable
    m.w = Var(m.T,m.Td, domain=Binary, initialize=0)
    #Power variable
    # m.p = Var(m.T,within=NonNegativeReals, initialize=0)
    m.p = Var(m.T,within=Reals, initialize=0)
    
    ## CONSTRAINTS
    # def Consty(m,t):
    #     return sum(m.y[t] for t in m.Ts) == 1
    # m.y_constraint = Constraint(m.Ts,rule=Consty)
    
    
    # def Constxy(m,t):
    
    #     return sum(m.x[t+i]for i in range(0,int(m.d))) >= m.d*m.y[t]
    # m.xy_constraint = Constraint(m.Ts,rule=Constxy)
    
    # def Constx(m,t):
        
    #     return sum(m.x[t] for t in m.T) == m.d
    # m.x_constraint = Constraint(m.T,rule=Constx) 
    
    
    # def Constxp(m,t,td):
    #     return m.p[t]== m.x[t]*m.profile[td]*miu
    # m.p_constraint = Constraint(m.T,m.Td,rule=Constxp)
    
    
    #Hegeller formulation
    
    # def Constwp(m,t):
    #     return m.p[t] == sum(m.w[t,td]*m.profile[td] for td in m.Td)
    # m.p_constraint = Constraint(m.T,rule=Constwp)
    
    
    def Constp(m,t):
        return sum(m.profile[td]*m.w[t,td] for td in m.Td) == m.p[t]
    m.p_constraint  = Constraint(m.T, rule=Constp)
        
    
    def Constw1(m,t):
        return sum(m.w[t,td] for td in m.Td) == m.y[t]
    m.w1_constraint = Constraint(m.T,rule=Constw1)
    
    
    def Constw2(m,t):
        return sum(m.w[t,td] for td in m.Td) <= 1
    m.w2_constraint = Constraint(m.T,rule=Constw2)

    
    def Constw3(m,td):
        return sum(m.w[t,td] for t in m.T) == 1
    m.w3_constraint = Constraint(m.Td,rule=Constw3)


    def Constw4(m):
        return sum(m.w[t,0] for t in m.T if t <= m.T.last()-m.d+1) == 1
    m.w4_constraint = Constraint(rule=Constw4)
    
    
    def Constw(m,t,td):
        if td > m.Td.last()-1 or t > m.T.last()-1:
            return Constraint.Skip
        return m.w[t+1,td+1] >= m.w[t,td]
            
    m.w_constraint = Constraint(m.T,m.Td, rule=Constw)
    

    
    
    #OBJECTIVE
    
    def MinCost(m):
        return sum((m.p[t]+m.b[t]*2)*m.c[t]*miu for t in m.T)
    m.objective = Objective(rule=MinCost)
    

    return m




#Maintain the constant consumption

def Agent_2(H,d,c,miu,profile,base,p_max,p_min):   
    """
    Alocation of individual shiftable loads with variable profile with total power constraints
    """
    
    m = ConcreteModel()

    m.d=Param(initialize=len(profile))
    m.Td=RangeSet(0,m.d-1) # cicle set
    # SETS
    m.T = RangeSet(0,H-1)
    # m.Ts = RangeSet(0,H-m.d-1)
    
    
    
    m.b=Param(m.T, initialize=base)
    
    m.c=Param(m.T, initialize=c)
    
    # m.p=Param(initialize=p)
    
    
    #shiftable device
    
    # m.Td=RangeSet(0,m.d-1)
    m.profile=Param(m.Td,initialize=profile)
    
    m.p_max=Param(initialize=p_max)
    m.p_min=Param(initialize=p_min)
    
    # VARIABLES
    # Starting variable
    m.y = Var(m.T,domain=Binary, initialize=0) 
    # Activity variable
    m.w = Var(m.T,m.Td, domain=Binary, initialize=0)
    #Power variable
    # m.p = Var(m.T,within=NonNegativeReals, initialize=0)
    m.p = Var(m.T,within=Reals, initialize=0)
    
    
    
    def Constp(m,t):
        return sum(m.profile[td]*m.w[t,td] for td in m.Td) == m.p[t]
    m.p_constraint  = Constraint(m.T, rule=Constp)
        
    
    def Constw1(m,t):
        return sum(m.w[t,td] for td in m.Td) == m.y[t]
    m.w1_constraint = Constraint(m.T,rule=Constw1)
    
    
    def Constw2(m,t):
        return sum(m.w[t,td] for td in m.Td) <= 1
    m.w2_constraint = Constraint(m.T,rule=Constw2)

    
    def Constw3(m,td):
        return sum(m.w[t,td] for t in m.T) == 1
    m.w3_constraint = Constraint(m.Td,rule=Constw3)


    def Constw4(m):
        return sum(m.w[t,0] for t in m.T if t <= m.T.last()-m.d+1) == 1
    m.w4_constraint = Constraint(rule=Constw4)
    
    
    def Constw(m,t,td):
        if td > m.Td.last()-1 or t > m.T.last()-1:
            return Constraint.Skip
        return m.w[t+1,td+1] >= m.w[t,td]
            
    m.w_constraint = Constraint(m.T,m.Td, rule=Constw)
    
    def Const_p_max(m,t):
        return m.p[t]+m.b[t] <= m.p_max
    m.pmax_constraint = Constraint(m.T, rule=Const_p_max)
    
    def Const_p_min(m,t):
        return m.p[t]+m.b[t] >= m.p_min
    m.pmin_constraint = Constraint(m.T, rule=Const_p_min)
        
    
    
    #OBJECTIVE
    
    def MinCost(m):
        return sum((m.p[t]+m.b[t])*m.c[t] for t in m.T)
    m.objective = Objective(rule=MinCost)
    

    return m



def Agent_3(H,d,c,miu,profiles,baseload,p_max):   
    
    
    m = ConcreteModel()
    

    
    apps=profiles.keys()
    m.n_ap=Param(initialize=len(apps))# Number of apps
    
    
    
    

    # SETS
    m.T = RangeSet(0,H-1)
    # m.Ts = RangeSet(0,H-m.d-1)
    # m.N = RangeSet(0,m.n_ap-1)
    m.N=Set(initialize=apps, ordered=False)

    
    duration={}
    Td_index={}
    for k in apps:
        
        duration[k]=len(profiles[k])
        Td_index[k]=range(0,len(profiles[k]))
        
    m.d=Param(m.N,initialize=duration) # duration of each app
    
    
    # m.Td=Set(m.N,initialize=Td_index) # cicle set
    
    
    m.b=Param(m.T, initialize=baseload)
    
    m.c=Param(m.T, initialize=c)
    
    # m.p=Param(initialize=p)
    
    
    #shiftable device
    

    m.profile=Param(m.N, initialize=profiles)
    
    m.p_max=Param(initialize=p_max)
    
    # VARIABLES
    # Starting variable
    m.y = Var(m.N,m.T,domain=Binary, initialize=0) 
    
    def Td_rule(m):
        return [(n,td) for n in m.N for td in range(m.d[n])]
    
    
    m.Td = Set(initialize=Td_rule)

    
    
    # Activity variable
    # m.w = Var(m.N,m.T,m.Td, domain=Binary, initialize=0)
    m.w = Var(m.Td,m.T, domain=Binary, initialize=0)
    #Power variable
    # m.p = Var(m.T,within=NonNegativeReals, initialize=0)
    m.p = Var(m.N,m.T,within=Reals, initialize=0)
    
    
    
    def Constp(m,t):
        return sum(m.profile[m.N,td]*m.w[t,td] for td in m.Td for n in m.N) == m.p[t]
    m.p_constraint  = Constraint(m.T, rule=Constp)
        
    
    def Constw1(m,t,n):
        return sum(m.w[t,td,n] for td in m.Td) == m.y[t,n]
    m.w1_constraint = Constraint(m.T,m.N,rule=Constw1)
    
    
    def Constw2(m,t,n):
        return sum(m.w[t,td,n] for td in m.Td) <= 1
    m.w2_constraint = Constraint(m.T,m.N,rule=Constw2)

    
    def Constw3(m,td,n):
        return sum(m.w[t,td,n] for t in m.T) == 1
    m.w3_constraint = Constraint(m.Td,m.N,rule=Constw3)


    def Constw4(m,n):
        return sum(m.w[t,0,n] for t in m.T if t <= m.T.last()-m.d+1) == 1
    m.w4_constraint = Constraint(m.N,rule=Constw4)
    
    
    def Constw(m,t,td,n):
        if td > m.Td.last()-1 or t > m.T.last()-1:
            return Constraint.Skip
        return m.w[t+1,td+1,n] >= m.w[t,td,n]
            
    m.w_constraint = Constraint(m.T,m.Td,m.N,rule=Constw)
    
    def Const_p_max(m,t):
        return m.p[t]+m.b[t] <= m.p_max
    m.pmax_constraint = Constraint(m.T, rule=Const_p_max)
    
    def Const_p_min(m,t):
        return m.p[t]+m.b[t] >= 0
    m.pmin_constraint = Constraint(m.T, rule=Const_p_min)
        
    
    
    #OBJECTIVE
    
    def MinCost(m):
        return sum((m.p[t]+m.b[t]*2)*m.c[t]*miu for t in m.T)
    m.objective = Objective(rule=MinCost)
    

    return m



# Customer load strategies for load shifting and curtailment

def LoadShift(H,c,base,p_max,y_max): 
    """Customer load strategies for load shifting and curtailment"""
    
    m = ConcreteModel()

    # SETS
    m.T = RangeSet(0,H-1)

    
    #Parameters    
    #base load
    m.b=Param(m.T, initialize=base)
    #electricity cost
    m.c=Param(m.T, initialize=c)
    #maximum increase/decrease variation
    m.p_max=Param(initialize=p_max)
    m.y_max=Param(initialize=y_max)
    
    
    # VARIABLES
    # increase/decrease variable
    m.y = Var(m.T,domain=Binary, initialize=0) 
    #Power varation variable
    m.p = Var(m.T,within=Reals, initialize=0)
    # Total power
    m.P_T = Var(m.T,within=Reals, initialize=0)
    
    
    # Constraints
    def Const_Total(m,t):
        return m.P_T[t] == m.b[t] + m.p[t] 
    m.Total_constraint  = Constraint(m.T, rule=Const_Total)
    # ::
    def Const_increase_max(m,t):
        return m.p[t] <= m.p_max*m.y[t]
    m.increase_max  = Constraint(m.T, rule=Const_increase_max)
    # ::
    def Const_decrease_min(m,t):
        return m.p[t] >= -m.p_max*(1-m.y[t])
    m.decrease_min  = Constraint(m.T, rule=Const_decrease_min)
    # ::
    def p_balance(m,t):
        return sum(m.p[t] for t in m.T)==0
    m.balance  = Constraint(m.T, rule=p_balance)
    
    def Const_y_max(m):
        return sum(m.y[t] for t in m.T)==m.y_max
    m.const_y_max = Constraint(rule=Const_y_max)
    
    #OBJECTIVE
    def MinCost(m):
        return sum(m.P_T[t]*m.c[t] for t in m.T)
    m.objective = Objective(rule=MinCost)
        
    
    return m






## CENTRALIZED PROBLEM ##

def Agent_C(H,nI,d0,p0,c,miu,Viol,Ppv):
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

    prosumer = ConcreteModel()
    # SETS
    prosumer.T = RangeSet(0,H-1)
    
    prosumer.I = RangeSet(0,nI-1)
    
    
    prosumer.c=Param(prosumer.T, initialize=c)
    
    prosumer.p=Param(prosumer.I,initialize=p0)
    
    prosumer.d=Param(prosumer.I,initialize=d0)
    
    prosumer.H=Param(initialize=H)
    
    # prosumer.miu=Param(prosumer.I, initialize=miu)
    
    def BuildTs(model,nI):
        for i in range(nI):
                return range(model.d[i],H-model.d[i])
            
    prosumer.Ts = Set(initialize=BuildTs(prosumer,nI))
    # prosumer.Ts = RangeSet(d0,H-d0)
            
    
    # VARIABLES
    
    # Starting variable
    prosumer.y = Var(prosumer.I,prosumer.T,domain=Binary, initialize=0)
    # Activity variable
    prosumer.x = Var(prosumer.I,prosumer.T,domain=Binary, initialize=0)
    
    prosumer.P = Var(prosumer.I,prosumer.T,domain=PositiveReals, initialize=1)
    
    
    
    ## CONSTRAINTS
    def Consty(prosumer,i,t):
        return sum(prosumer.y[i,t] for t in prosumer.Ts) == 1
    prosumer.y_constraint = Constraint(prosumer.I,prosumer.Ts,rule=Consty)
    
    #
    def Constxy(prosumer,i,t):
        # for t in prosumer.Ts:
    #        if t >= prosumer.d[i] or t <= H-prosumer.d[i]:
                return sum(prosumer.x[i,t+k] for k in range(0,prosumer.d[i]))\
            >=prosumer.d[i]*prosumer.y[i,t]
        
    prosumer.xy_constraint = Constraint(prosumer.I,prosumer.Ts,rule=Constxy)
    
    
    def ConstP(prosumer,i,t):
        return prosumer.P[i,t] == prosumer.x[i,t]*prosumer.p[i]
    prosumer.PConstraint = Constraint(prosumer.I,prosumer.T, rule=ConstP) 
    
    
    def ConstTotal(prosumer,t):
        return sum(prosumer.x[i,t]*prosumer.p[i] for i in prosumer.I) <= Ppv[t,0]+Viol[t]
    prosumer.TotalConstraint = Constraint(prosumer.T, rule=ConstTotal) 
    
    
    def Constx(prosumer,i,t):    
        return sum(prosumer.x[i,t] for t in prosumer.T) == prosumer.d[i]
    prosumer.x_constraint = Constraint(prosumer.I,prosumer.T,rule=Constx) 
    
    
    #OBJECTIVE
    def MinCost(prosumer):
        return sum(sum(prosumer.x[i,t]*prosumer.c[t]*prosumer.p[i]*miu for t in prosumer.T)\
                   for i in prosumer.I) 
    prosumer.objective = Objective(rule=MinCost)


    return prosumer



def Agent_C1(H,nI,d,l,c,Viol,Ppv,base):
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

    m = ConcreteModel()
    # SETS
    m.T = RangeSet(0,H-1)
    
    m.I = RangeSet(0,nI-1)
    
    
    m.c=Param(m.T, initialize=c)
    
    m.pv=Param(m.T, initialize=Ppv)
    
    m.b=Param(m.T, initialize=base)
    
    m.l=Param(m.I,initialize=l)
    
    m.d=Param(m.I,initialize=d)
    
    m.H=Param(initialize=H)
    
    # m.miu=Param(m.I, initialize=miu)
    
    def BuildTs(model,nI):
        for i in range(nI):
                return range(model.d[i],H-model.d[i])
            
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
        return m.Px[t] == max(0,m.pv[t]-m.b[t])
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


    return m



def get_opti_vars(model):
    
    w_sol=pd.DataFrame([[value(model.w[i,j]) for i in model.T] for j in model.Td])
    
    y=pd.DataFrame([value(model.y[i]) for i in model.T])

    p=pd.DataFrame([value(model.p[i]) for i in model.T])
    
    c=model.objective()
    
    b=pd.DataFrame([value(model.b[i]) for i in model.T])

    return w_sol,y,p,c,b



def get_opti_vars_2(model):
    

    
    y=pd.DataFrame([value(model.y[i]) for i in model.T])

    p=pd.DataFrame([value(model.p[i]) for i in model.T])
    
    P_T=pd.DataFrame([value(model.P_T[i]) for i in model.T])
    
    c=model.objective()
    
    b=pd.DataFrame([value(model.b[i]) for i in model.T])

    return y,p,P_T,c,b
