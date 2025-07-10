from __future__ import division

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyomo.environ as en
from pyomo.opt import SolverFactory
from pyomo.core import Var
import seaborn as sns
import time
import os
import yaml

from battery import Battery
from Instant_batterymodel import battmodel_ as battmodel
from Instant_batterymodel import make_plot

#%% Setup
cwd = os.getcwd()
DataFolder = os.path.join(cwd, 'data')

sns.set_style({'axes.linewidth': 1, 'axes.edgecolor': 'black',
               'xtick.direction': 'out', 'xtick.major.size': 4.0,
               'ytick.direction': 'out', 'ytick.major.size': 4.0,
               'axes.facecolor': 'white', 'grid.color': '.8',
               'grid.linestyle': '-', 'grid.linewidth': 0.5})

#%% Parameters
ti = 0
H = 5 * 24 * 2 * 2
dt = 15

#%% Load Batteries from YAML
with open("batteries.yaml", "r") as f:
    battery_data = yaml.safe_load(f)

batteries = []
for batt_spec in battery_data["batteries"]:
    percent = batt_spec.pop("initial_soc_percent", 0.0)
    capacity = batt_spec["capacity"]
    batt_spec["current_charge"] = percent * capacity  
    batteries.append(Battery(**batt_spec))

#%% Load Data
g_df = pd.read_csv(os.path.join(DataFolder, 'pv_gen.csv'), header=None)
num_batteries = len(batteries)
PV_total = 4 * g_df.iloc[:, :num_batteries].clip(lower=0).sum(axis=1).values[ti:H]

loads = pd.read_csv(os.path.join(DataFolder, 'load_cons.csv'), header=None)

fixed_buy = 0.10 

fixed_sell = 0.01

tar = pd.Series(fixed_buy, index=range(H), name='tar')
sell = pd.Series(fixed_sell, index=range(H), name='tar_sell')
priceDict1 = dict(enumerate(sell))
priceDict2 = dict(enumerate(tar))

#%% Dynamic Alpha

loads_matrix = loads.iloc[ti:H, :num_batteries].astype(float).values  # shape: (H, n_batteries)
load_sums = loads_matrix.sum(axis=1).reshape(-1, 1)  # soma total de carga por timestep (shape: Hx1)

alpha_matrix = np.divide(loads_matrix, load_sums, where=load_sums!=0)  # shape: H x n_batteries

#To set all alphas as 0.5
#alpha_matrix = np.full_like(loads_matrix, 1.0 / loads_matrix.shape[1])

PV_matrix = alpha_matrix * PV_total[:, np.newaxis]


#%% Run model per battery
all_results = []
for i, batt in enumerate(batteries):
      
    load = loads_matrix[:, i]
    PV = PV_matrix[:, i]
    
    model = battmodel(H, batt, load, PV, priceDict1, priceDict2, dt, batt.capacity, batt.current_charge)
    opt = SolverFactory("glpk")
    print(f"Solving for battery {i+1}...")
    results = opt.solve(model, tee=False)
    print(f"Battery {i+1} solved.")

    # Extract variables
    outputVars = np.zeros((9, H))
    varnames = []
    j = 0
    for v in model.component_objects(Var, active=True):
        varnames.append(v.getname())
        varobject = getattr(model, str(v))
        for index in varobject:
            outputVars[j, index] = varobject[index].value
        j += 1
        if j >= 9:
            break

    df_param = pd.DataFrame({'load': load, 'PV': PV})
    df_sol = pd.merge(pd.DataFrame(outputVars.T, columns=varnames), df_param, left_index=True, right_index=True)
    df_sol = pd.merge(df_sol, tar, left_index=True, right_index=True)
    df_sol = pd.merge(df_sol, sell, left_index=True, right_index=True)
    df_sol['battery_id'] = i
    all_results.append(df_sol)

#%% Combine and Plot
for df_sol in all_results:
    make_plot(df_sol, 0, H)
