from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyomo.environ as en
from pyomo.opt import SolverFactory
import seaborn as sns
import os
import yaml

from env.community import Community
from env.state import StateVars
from utils.dataprocessor import YAMLParser
from utils.utilities import ConfigsParser

from battery import Battery
from batmodel import battmodel_
from ploty import make_plot

# === Initialization ===
start_time = time.time()
cwd = Path.cwd()
datafolder = cwd.parent / 'Data'
configs_folder = cwd.parent / 'configs'
resultsfolder = cwd.parent / 'Results'

# === Load Configs ===
exp_name = YAMLParser().load_yaml(configs_folder / 'exp_name.yaml')['exp_name']
configs = ConfigsParser(configs_folder, exp_name)
file_ag_conf, file_apps_conf, file_scene_conf, file_prob_conf, file_vars, file_experiment, ppo_config = configs.get_configs()

# === Load Dataset ===
gecad_dataset = datafolder / 'dataset_gecad_clean.csv'

# === Create Community ===
com = Community(file_ag_conf, file_scene_conf, file_prob_conf, gecad_dataset)
com_vars = StateVars(file_vars)

# === Set Parameters ===
ti = 0
H = 4 * 24 * 4

# === Build and Solve Model ===
model = battmodel_(com, H)
opt = SolverFactory("glpk")

print("Solving model...")
results = opt.solve(model, tee=False)
print(results.solver.termination_condition)
print("Model solved.")

# === Extract Results ===
data = {}
dt = com.agent_processor.step_size
load_df = pd.DataFrame({aid: agent.data['load'] for aid, agent in com.agents.items()})
pv_df =  pd.DataFrame({aid: agent.data['gen'] for aid, agent in com.agents.items()})
tar = next(iter(com.agents.values())).tariff[:H]

# Compute normalized alpha values
alpha_values = {}
for t in model.Time:
    total_allocated = sum(model.pv[b, t].value for b in model.BATTS)
    for b in model.BATTS:
        alpha_values[(b, t)] = model.pv[b, t].value / total_allocated if total_allocated > 0 else 0.0

for b in com.agents:
    for var in ['load', 'PV', 'SOC', 'posEInGrid', 'posEInPV', 'posNetLoad', 'negNetLoad', 'negEOutLocal', 'negEOutExport']:

        if var == 'load':
            data[(b, var)] = load_df[b].values[:H]
        elif var == 'PV':
            data[(b, var)] = pv_df[b].values[:H]
        else:
            var_obj = getattr(model, var)
            data[(b, var)] = [var_obj[b, t].value for t in range(H)]

    data[(b, 'tar')] = tar[:H]
    data[(b, 'alpha')] = [alpha_values[(b, t)] for t in range(H)]
    data[(b, 'allocatedPV')] = [model.pv[b, t].value for t in range(H)]

# Add community unused PV
data[('community', 'unusedPV')] = [model.unusedPV[t].value for t in range(H)]

# === Combine Results ===
df_results = pd.DataFrame(data)

# === Plot ===
for b in com.agents:
    make_plot(df_results, 0, H, battery_id=b)

def plot_alpha_distribution(alpha_values, agents, time_steps):
    df_alpha = pd.DataFrame(0.0, index=range(time_steps), columns=agents)

    for (b, t), alpha in alpha_values.items():
        df_alpha.loc[t, b] = alpha

    # Plot de linhas
    ax = df_alpha.plot(
        kind='line',
        figsize=(14, 5),
        linewidth=2
    )
    ax.set_title("Alpha Distribution per Agent Over Time")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Alpha Value")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Agent")
    plt.tight_layout()
    plt.show()

# Chamada
plot_alpha_distribution(alpha_values, list(model.BATTS), len(model.Time))

