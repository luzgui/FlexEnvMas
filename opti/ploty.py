import pyomo.environ as en
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from battery import Battery


def make_plot(df, t0, t1, battery_id):
    hrs = np.arange(t0, t1)
    colors = sns.color_palette()
    cols = lambda vars: [(battery_id, var) for var in vars]

    # ler as séries
    load = df[(battery_id, 'load')]
    PV = df[(battery_id, 'allocatedPV')]
    posNL = df[(battery_id, 'posNetLoad')]    # importações
    negNL = df[(battery_id, 'negNetLoad')]    # exportações
    Action = df[cols(['posEInGrid', 'posEInPV', 'negEOutLocal', 'negEOutExport'])].sum(axis=1)
    soc = df[(battery_id, 'SOC')]
    tar = df[(battery_id, 'tar')]
    alpha = df[(battery_id, 'alpha')]

    fig = plt.figure(figsize=(14, 18))
    fig.suptitle(f'Agent: {battery_id}', fontsize=16)

    # Primeiro plot: Load, PV, Grid Import e Grid Export, Battery Action
    ax1 = plt.subplot2grid((10, 1), (0, 0), rowspan=3)
    ax1.plot(hrs, load[t0:t1],           color=colors[0], label='Load')
    ax1.plot(hrs, PV[t0:t1],             color=colors[1], label='Allocated PV')
    ax1.plot(hrs, posNL[t0:t1],          color=colors[2], label='Grid+Battery Usage')
    ax1.plot(hrs, -negNL[t0:t1],         color=colors[3], label='Energy Excedente')
    ax1.plot(hrs, Action[t0:t1],         color=colors[5], label='BatteryAction')
    ax1.legend(loc='upper left')
    ax1.set_ylabel('kWh')
    ax1.grid(True)

    # desenhar SOC a direito
    ax2 = ax1.twinx()
    ax2.plot(hrs, soc[t0:t1], color=colors[4], label='SOC')
    ax2.legend(loc='upper right')
    ax2.set_ylabel('SOC (kWh)')

    # resto dos subplots mantém-se igual…
    ax3 = plt.subplot2grid((10, 1), (3, 0), rowspan=3)
    ax3.plot(hrs, soc[t0:t1], color=colors[4], label='SOC')
    ax3.plot(hrs, Action[t0:t1], color=colors[5], label='BatteryAction')
    ax3.legend(loc='upper left')
    ax3.set_ylabel('kWh')
    ax3.grid()
    ax4 = ax3.twinx()
    ax4.plot(hrs, tar[t0:t1], color=colors[3], linestyle='--', label='Buy Price')
    ax4.legend(loc='upper right')
    ax4.set_ylabel('Price')
    ax4.grid(axis='x', linewidth=1.5, color='black', linestyle='dashed')

    ax5 = plt.subplot2grid((10, 1), (6, 0), rowspan=1)
    ax5.plot(hrs, alpha[t0:t1], color='black', linewidth=1)
    ax5.set_ylabel('α', rotation=0, labelpad=10)
    ax5.set_yticks([0, 0.5, 1])
    ax5.set_ylim(0, 1)
    ax5.set_xlabel('15-min intervals')
    ax5.grid()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
