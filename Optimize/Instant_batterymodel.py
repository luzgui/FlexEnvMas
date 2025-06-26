from __future__ import division

import numpy as np
import csv
import matplotlib.pyplot as plt
import math
import random
import pandas as pd
from pyomo.opt import SolverFactory
from pyomo.core import Var
import pyomo.environ as en
import seaborn as sns
import time

# Since using the data-driven data for the testing, use their battery class
from battery import Battery


def battmodel_(H,batt_obj, load, PV, p_sell, b_buy, dt, capacity, current_SOC):
    
    "H: time horizon, length of data \
     batt_obj: object conatinig the battery instance (instance of class battery.py)   "
    
    
    net = load-PV
    # posload = np.copy(net)
    # negload = np.copy(net)
    # for j,e in enumerate(net):
    #     if e>=0:
    #         negload[j]=0 #load > PV
    #     else: # e < 0
    #         posload[j]=0 #load < PV
    # split net into positive demand and (negative) exports

    posload = np.where(net >= 0, net, 0)
    negload = np.where(net <  0, net, 0)

    batt = batt_obj
    # Pyomo Param initializers prefer dicts:
    posLoadDict = dict(enumerate(posload))
    
    negLoadDict = dict(enumerate(negload))
    
    priceDict1=p_sell
    priceDict2=b_buy
    current_SOC = current_SOC # Adição minha
    
    
    # now set up the pyomo model
    m = en.ConcreteModel()
    
    # we use rangeset to make a sequence of integers
    # time is what we will use as the model index
    m.Time = en.RangeSet(0, H-1) # H é adição minha. O -1 é por causa de python começar em 0
    
    
    # #### Variables
    # Now we define the variables that we are interested in: <br>
    # We formulate the problem such that our decision variables are *posNetLoad* and *negNetLoad* <br>
    # See the objective fn.
    
    # In[75]:
    
    
    # variables (all indexed by Time)
    # m.SOC = en.Var(m.Time, bounds=(0.1*batt.capacity,batt.capacity), initialize=0.1*batt.capacity) #0 implements limits in soc
    
    m.SOC = en.Var(m.Time, bounds=(0,batt.capacity), initialize=0) #0 implements limits in soc
    
    
    m.posDeltaSOC = en.Var(m.Time, initialize=0) #1
    m.negDeltaSOC = en.Var(m.Time, initialize=0) #2
    m.posEInGrid = en.Var(m.Time, bounds=(0,batt.charging_power_limit), initialize=0) #3
    m.posEInPV = en.Var(m.Time, bounds=(0,batt.charging_power_limit), initialize=0) #4
    m.negEOutLocal = en.Var(m.Time, bounds=(batt.discharging_power_limit,0), initialize=0) #5
    m.negEOutExport = en.Var(m.Time, bounds=(batt.discharging_power_limit,0), initialize=0) #6
    m.posNetLoad = en.Var(m.Time, initialize=posLoadDict) #7
    m.negNetLoad = en.Var(m.Time, initialize=negLoadDict) #8
    
    
    # The numbers commented after are the indices that we will use when looping through the model components afterwards to get the final values of the variables after the optimisation has been completed
    
    # The Boolean variables are what we will use to denote whether the battery is charging or discharging at a particular period
    
    # In[76]:
    
    
    # Boolean variables (again indexed by Time)
    m.Bool_char=en.Var(m.Time,within=en.Boolean) #9
    m.Bool_dis=en.Var(m.Time,within=en.Boolean,initialize=0) #10
    
    
    # In[77]:
    
    
    # parameters (indexed by time)
    m.priceSell = en.Param(m.Time, initialize=priceDict1)
    m.priceBuy = en.Param(m.Time, initialize=priceDict2)
    m.posLoad = en.Param(m.Time, initialize=posLoadDict)
    m.negLoad = en.Param(m.Time, initialize=negLoadDict)
    
    
    # In[78]:
    
    
    # single value parameters
    m.etaChg = en.Param(initialize = batt.charging_efficiency)
    m.etaDisChg = en.Param(initialize = batt.discharging_efficiency)
    m.ChargingLimit = en.Param(initialize = batt.charging_power_limit)
    m.DischargingLimit = en.Param(initialize = batt.discharging_power_limit)
    
    
    # #### Objective function 
    # Now define the objective function that we are going to minimise (the cost of the site's electricity)
    
    # In[79]:
    
    
    # objective function
    def Obj_fn(m):
        return sum((m.priceBuy[i]*m.posNetLoad[i]) + (m.priceSell[i]*m.negNetLoad[i]) for i in m.Time)  
    
        # return sum(()  for i in m.Time) 
    m.total_cost = en.Objective(rule=Obj_fn,sense=en.minimize)
    
    
    # In the above posNetLoad and negNetLoad are variables, indexed by time that will change dependent on the action of the battery <br>
    # They have initially been assigned using posLoad and negLoad, which correspond to no battery action
    
    # We now need to think about the constraints on the model. First of all, we add a constraint which represents the finite physical capacity of the battery, which cannot be above the maximum and cannot fall below zero
    
    # In[80]:
    
    
    # constraints
    # first we define the constraint at each time period
    def SOC_rule(m,t):
        # if H==0:
        #     # return (m.SOC[t] == m.posDeltaSOC[t]+m.negDeltaSOC[t])
        #     return m.SOC[t] == 0.03*capacity # An initial value
        # else:
        #     # return (m.SOC[t] == m.SOC[t-1]+m.posDeltaSOC[t]+m.negDeltaSOC[t])   
        if t == 0:
            return (m.SOC[t] == current_SOC +m.posDeltaSOC[t]+m.negDeltaSOC[t])  # Adição minha
        
        else:
            return m.SOC[t] == m.SOC[t-1] + m.posDeltaSOC[t] + m.negDeltaSOC[t]
        #FP - SOC dinâmico: o estado de carga no tempo t depende do SOC no tempo t-1
        
    # then we specify that this constraint is indexed by time
    m.Batt_SOC = en.Constraint(m.Time,rule=SOC_rule)
    
    
    # #### boolean constraints - the integers
    # The next set of constraints is the "Integer" part in the Mixed Integer Linear Program formulation. <br>
    # These constraints explicitly constrain that the battery can only charge OR discharge during one time period <br>
    # The observant might notice that in this specific example, these constraints aren't actually required, since in our objective function there will never be an economic benefit to this type of action <br>
    # However, it is good to see how they are set up, they can make the optimisation faster and many cases they are required
    
    # In[81]:
    
    
    # we use bigM to bound the problem
    # boolean constraints
    
    def Bool_char_rule_1(m,i):
        bigM=500000
        return((m.posDeltaSOC[i])>=-bigM*(m.Bool_char[i]))
    m.Batt_ch1=en.Constraint(m.Time,rule=Bool_char_rule_1)
    # if battery is charging, charging must be greater than -large
    # if not, charging geq zero
    
    
    def Bool_char_rule_2(m,i):
        bigM=500000
        return((m.posDeltaSOC[i])<=0+bigM*(1-m.Bool_dis[i]))
    m.Batt_ch2=en.Constraint(m.Time,rule=Bool_char_rule_2)
    # if batt discharging, charging must be leq zero
    # if not, charging leq +large
    
    
    def Bool_char_rule_3(m,i):
        bigM=500000
        return((m.negDeltaSOC[i])<=bigM*(m.Bool_dis[i]))
    m.Batt_cd3=en.Constraint(m.Time,rule=Bool_char_rule_3)
    # if batt discharge, discharge leq POSITIVE large
    # if not, discharge leq 0
    
    
    def Bool_char_rule_4(m,i):
        bigM=500000
        return((m.negDeltaSOC[i])>=0-bigM*(1-m.Bool_char[i]))
    m.Batt_cd4=en.Constraint(m.Time,rule=Bool_char_rule_4)
    # if batt charge, discharge geq zero
    # if not, discharge geq -large
    
    
    def Batt_char_dis(m,i):
        return (m.Bool_char[i]+m.Bool_dis[i],1)
    m.Batt_char_dis=en.Constraint(m.Time,rule=Batt_char_dis)
    
    
    # bigM is a big number to bound the problem...
    # Here is a link from an MIT open course: https://ocw.mit.edu/courses/sloan-school-of-management/15-053-optimization-methods-in-management-science-spring-2013/tutorials/MIT15_053S13_tut09.pdf
    
    # #### battery efficiency
    # The next constraints deal with the battery efficiency: <br>
    # We ensure that any change in the battery in the battery's state of charge at a particular period due to charging is reduced by the charging efficieny <br>
    # Similarly, we ensure that the energy output from the battery is reduced when it is converted to an output
    
    # In[82]:
    
    
    #ensure charging efficiency is divided
    def pos_E_in_rule(m,i):
        return (m.posEInGrid[i]+m.posEInPV[i]) == m.posDeltaSOC[i]/m.etaChg/(dt/60.)
    m.posEIn_cons = en.Constraint(m.Time, rule=pos_E_in_rule)
    # ensure discharging eff multiplied
    def neg_E_out_rule(m,i):
        return (m.negEOutLocal[i]+m.negEOutExport[i]) == m.negDeltaSOC[i]*m.etaDisChg/(dt/60.)
    m.negEOut_cons = en.Constraint(m.Time, rule=neg_E_out_rule)
    
    
    # #### Charging and discharging power limits
    # Now ensure that the charging and discharging power limits of the battery are respected. <br>
    # Note that we have opted to split the energy into that coming-from the grid (posEInGrid), going-to the grid (negEOutExport), coming from local PV (posEInPV) and being used locally (negEOutLocal)
    
    # In[83]:
    
    
    # ensure charging rate obeyed
    def E_charging_rate_rule(m,i):
        return (m.posEInGrid[i]+m.posEInPV[i])<=m.ChargingLimit
    m.chargingLimit_cons = en.Constraint(m.Time, rule=E_charging_rate_rule)
    # ensure DIScharging rate obeyed
    def E_discharging_rate_rule(m,i):
        return (m.negEOutLocal[i]+m.negEOutExport[i])>=m.DischargingLimit
    m.dischargingLimit_cons = en.Constraint(m.Time, rule=E_discharging_rate_rule)
    
    
    
    # #### Further constraints to ensure physical sense
    
    # In[84]:
    
    
    # ensure that posEInPV cannot exceed local PV (note gui: cannot exceed pv excess)
    def E_solar_charging_rule(m,i):
        return m.posEInPV[i]<=-m.negLoad[i]
    m.solarChargingLimit_cons = en.Constraint(m.Time, rule=E_solar_charging_rule)
    # ensure that negEOutLocal cannot exceed local demand
    def E_local_discharge_rule(m,i):
        return m.negEOutLocal[i]>=-m.posLoad[i]
    m.localDischargingLimit_cons = en.Constraint(m.Time, rule=E_local_discharge_rule)
    
    
    # ensure that there are no exports from the abttery to avoid arbitrage
    def E_No_export_Battery(m,i):   # Adição minha
        return m.negEOutExport[i] == 0
    m.NoExportDischarging_cons = en.Constraint(m.Time, rule=E_No_export_Battery)
    
    # #### Rules for actually calculating the main decision variables
    
    # In[85]:
    
    
    # calculate the net positive demand
    def E_pos_net_rule(m,i):
        return m.posNetLoad[i] == m.posLoad[i]+m.posEInGrid[i]+m.negEOutLocal[i]
    m.E_posNet_cons = en.Constraint(m.Time,rule=E_pos_net_rule)
    
    # calculate export
    def E_neg_net_rule(m,i):
        return m.negNetLoad[i] == m.negLoad[i]+m.posEInPV[i]+m.negEOutExport[i]
    m.E_negNet_cons = en.Constraint(m.Time,rule=E_neg_net_rule)
    
    
    return m
    
    
def make_plot(df,t0,t1):
    
    T0=t0
    T=t1
    
    df_sol=df

    df_sol['load'][T0:T]

    # newNetLoad = outputVars[7]+outputVars[8] #positive + negative net load
    newNetLoad = df_sol['posNetLoad']+df_sol['negNetLoad']
    Action=df_sol[['posEInGrid', 'posEInPV','negEOutLocal','negEOutExport']].sum(axis=1)
    soc=df_sol['SOC']


    colors = sns.color_palette()
    hrs = np.arange(t0,t1)
    fig = plt.figure(figsize=(14,14))
    ax1 = fig.add_subplot(3,1,1)
    l1, = ax1.plot(hrs,df_sol['load'][T0:T],color=colors[0], label='load') #Load
    l2, = ax1.plot(hrs,df_sol['PV'][T0:T],color=colors[1],label='PV') #PV
    l3, = ax1.plot(hrs, -newNetLoad[T0:T],color=colors[3],label='NetLoad') ##positive + negative net load
    l4, = ax1.plot(hrs,Action[T0:T],color=colors[5],label='BatteryAction') # sum of the 4 balance variables 
    ax1.legend(loc='upper left')
    ax1.set_xlabel('hour'), ax1.set_ylabel('kWh')
    ax2=ax1.twinx()
    l5, = ax2.plot(hrs,soc[T0:T],color=colors[4], label='SOC')
    ax2.legend()
    fig.tight_layout()
    # ax1.grid(axis='x',linewidth = 1.5, color='black',linestyle = '--')
    ax1.grid()


    #2nd plot
    ax3 = fig.add_subplot(3,1,2)

    l1, = ax3.plot(hrs,soc[T0:T],color=colors[4], label='SOC')
    l2, = ax3.plot(hrs,Action[T0:T],color=colors[5],label='BatteryAction')
    ax3.legend(loc='upper left')
    ax3.grid()
    ax4=ax3.twinx()
    l3, = ax4.plot(hrs,df_sol['tar'][T0:T],color=colors[3],linestyle = '--',label='buy price')
    # l4, = ax4.plot(hrs,df_sol['tar_sell'][T0:T],color=colors[4])



    ax4.legend(loc='upper right')
    ax4.grid(axis='x',linewidth = 1.5, color='black',linestyle = 'dashed')
    ax3.set_ylabel('kWh'), ax4.set_ylabel('price')