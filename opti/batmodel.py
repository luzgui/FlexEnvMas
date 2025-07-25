import pyomo.environ as en
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from battery import Battery

def battmodel_(com, H):
    """
    com: Community instance
    H: Time horizon (number of time steps)
    """
    m = en.ConcreteModel()

    dt = 15

    battery_dict = {}
    for aid, agent in com.agents.items():
        batt_params = agent.conf['battery']
        battery = Battery(**batt_params)
        battery.set_initial_charge()
        battery_dict[aid] = battery

    load_df = pd.DataFrame({aid: agent.data['load'] for aid, agent in com.agents.items()})
    pv_df = pd.DataFrame({aid: agent.data['gen'] for aid, agent in com.agents.items()})
    
    total_pv = pv_df.sum(axis=1)
    
    
    buy_prices = {(b, t): com.agents[b].tariff[t] for b in battery_dict for t in range(H)}
    sell_prices = {t: 0.0 for t in range(H)}

    m.BATTS = en.Set(initialize=list(battery_dict.keys()))
    m.Time = en.RangeSet(0, H - 1)
    
    posload = load_df.copy()
    negload = pd.DataFrame(0, index=load_df.index, columns=load_df.columns)

    m.load_demand = en.Param(m.BATTS, m.Time, initialize={(b, t): load_df[b].iloc[t] for b in battery_dict for t in range(H)})
    total_pv_dict = {t: total_pv.iloc[t] for t in range(H)}
    m.total_pv = en.Param(m.Time, initialize=total_pv_dict)

    m.unusedPV = en.Var(m.Time, within=en.NonNegativeReals, initialize=0)
    m.pv = en.Var(m.BATTS, m.Time, within=en.NonNegativeReals, initialize=0)
    m.posLoad = en.Param(m.BATTS, m.Time, initialize={(b, t): posload[b].iloc[t] for b in battery_dict for t in range(H)})
    m.negLoad = en.Param(m.BATTS, m.Time, initialize={(b, t): negload[b].iloc[t] for b in battery_dict for t in range(H)})
    m.priceSell = en.Param(m.Time, initialize=sell_prices)
    m.priceBuy = en.Param(m.BATTS, m.Time, initialize=buy_prices)


    m.SOC = en.Var(m.BATTS, m.Time, bounds=lambda m, b, t: (0, battery_dict[b].capacity))
    m.posDeltaSOC = en.Var(m.BATTS, m.Time, bounds=lambda m, b, t: (0, battery_dict[b].charging_power_limit * dt / 60.0), initialize=0)
    m.negDeltaSOC = en.Var(m.BATTS, m.Time, bounds=lambda m, b, t: (-abs(battery_dict[b].discharging_power_limit) * dt / 60.0, 0), initialize=0)

    m.posEInGrid = en.Var(m.BATTS, m.Time, initialize=0)
    m.posEInPV = en.Var(m.BATTS, m.Time, initialize=0)
    m.negEOutLocal = en.Var(m.BATTS, m.Time, initialize=0)
    m.negEOutExport = en.Var(m.BATTS, m.Time, initialize=0)
    
    m.posNetLoad = en.Var(m.BATTS, m.Time, initialize=0)
    m.negNetLoad = en.Var(m.BATTS, m.Time, initialize=0)
    
    m.Bool_char = en.Var(m.BATTS, m.Time, within=en.Boolean)
    m.Bool_dis = en.Var(m.BATTS, m.Time, within=en.Boolean, initialize=0)

    def pv_usage_limit(m, b, t):
        return m.posEInPV[b, t] <= m.pv[b, t]
    m.PV_usage_limit = en.Constraint(m.BATTS, m.Time, rule=pv_usage_limit)

    def pv_balance(m, t):
        return sum(m.posEInPV[b, t] for b in m.BATTS) + m.unusedPV[t] == m.total_pv[t]    
    m.PV_balance = en.Constraint(m.Time, rule=pv_balance)

    def soc_rule(m, b, t):
        if t == 0:
            return m.SOC[b, t] == battery_dict[b].current_charge + m.posDeltaSOC[b, t] + m.negDeltaSOC[b, t]
        return m.SOC[b, t] == m.SOC[b, t - 1] + m.posDeltaSOC[b, t] + m.negDeltaSOC[b, t]
    m.Batt_SOC = en.Constraint(m.BATTS, m.Time, rule=soc_rule)

    m.etaChg = en.Param(m.BATTS, initialize=lambda m, b: battery_dict[b].charging_efficiency)
    m.etaDisChg = en.Param(m.BATTS, initialize=lambda m, b: battery_dict[b].discharging_efficiency)
    m.ChargingLimit = en.Param(m.BATTS, initialize=lambda m, b: battery_dict[b].charging_power_limit)
    m.DischargingLimit = en.Param(m.BATTS, initialize=lambda m, b: battery_dict[b].discharging_power_limit)

    penalty_weight = 0.5  # Penalty for unused PV
    
    def Obj_fn(m):
        return sum(m.priceBuy[b, t] * m.posNetLoad[b, t] + m.priceSell[t] * m.negNetLoad[b, t]
                   for b in m.BATTS for t in m.Time ) + penalty_weight * sum(m.unusedPV[t] for t in m.Time)
    
    m.total_cost = en.Objective(rule=Obj_fn, sense=en.minimize)

    def Bool_char_rule_1(m, b, t):
        return m.posDeltaSOC[b, t] >= -500000 * m.Bool_char[b, t]
    m.Batt_ch1 = en.Constraint(m.BATTS, m.Time, rule=Bool_char_rule_1)

    def Bool_char_rule_2(m, b, t):
        return m.posDeltaSOC[b, t] <= 500000 * (1 - m.Bool_dis[b, t])
    m.Batt_ch2 = en.Constraint(m.BATTS, m.Time, rule=Bool_char_rule_2)

    def Bool_char_rule_3(m, b, t):
        return m.negDeltaSOC[b, t] <= 500000 * m.Bool_dis[b, t]
    m.Batt_cd3 = en.Constraint(m.BATTS, m.Time, rule=Bool_char_rule_3)

    def Bool_char_rule_4(m, b, t):
        return m.negDeltaSOC[b, t] >= -500000 * (1 - m.Bool_char[b, t])
    m.Batt_cd4 = en.Constraint(m.BATTS, m.Time, rule=Bool_char_rule_4)

    def soc_discharge_limit(m, b, t):
        if t == 0:
            return m.negDeltaSOC[b, t] >= -battery_dict[b].current_charge
        return m.negDeltaSOC[b, t] >= -m.SOC[b, t - 1]
    m.SOC_discharge_limit = en.Constraint(m.BATTS, m.Time, rule=soc_discharge_limit)

    def soc_charge_limit(m, b, t):
        if t == 0:
            return m.posDeltaSOC[b, t] <= battery_dict[b].capacity - battery_dict[b].current_charge
        return m.posDeltaSOC[b, t] <= battery_dict[b].capacity - m.SOC[b, t - 1]
    m.SOC_charge_limit = en.Constraint(m.BATTS, m.Time, rule=soc_charge_limit)

    def Batt_char_dis(m, b, t):
        return m.Bool_char[b, t] + m.Bool_dis[b, t] <= 1
    m.Batt_char_dis = en.Constraint(m.BATTS, m.Time, rule=Batt_char_dis)

    def pos_E_in_rule(m, b, t):
        return (m.posEInGrid[b, t] + m.posEInPV[b, t] == m.posDeltaSOC[b, t] / m.etaChg[b] * (60.0 / dt))
    m.posEIn_cons = en.Constraint(m.BATTS, m.Time, rule=pos_E_in_rule)

    def neg_E_out_rule(m, b, t):
        return (m.negEOutLocal[b, t] + m.negEOutExport[b, t] == m.negDeltaSOC[b, t] * m.etaDisChg[b] * (60.0 / dt))
    m.negEOut_cons = en.Constraint(m.BATTS, m.Time, rule=neg_E_out_rule)

    def E_charging_rate_rule(m, b, t):
        return (m.posEInGrid[b, t] + m.posEInPV[b, t]) <= m.ChargingLimit[b]
    m.chargingLimit_cons = en.Constraint(m.BATTS, m.Time, rule=E_charging_rate_rule)

    def E_discharging_rate_rule(m, b, t):
        return (m.negEOutLocal[b, t] + m.negEOutExport[b, t]) >= m.DischargingLimit[b]
    m.dischargingLimit_cons = en.Constraint(m.BATTS, m.Time, rule=E_discharging_rate_rule)

    def E_solar_charging_rule(m, b, t):
        return m.posEInPV[b, t] <= m.pv[b, t]
    m.solarChargingLimit_cons = en.Constraint(m.BATTS, m.Time, rule=E_solar_charging_rule)

    def E_local_discharge_rule(m, b, t):
        return m.negEOutLocal[b, t] >= -m.posLoad[b, t]
    m.localDischargingLimit_cons = en.Constraint(m.BATTS, m.Time, rule=E_local_discharge_rule)

    def E_No_export_Battery(m, b, t):
        return m.negEOutExport[b, t] == 0
    m.NoExportDischarging_cons = en.Constraint(m.BATTS, m.Time, rule=E_No_export_Battery)

    def E_pos_net_rule(m, b, t):
        return m.posNetLoad[b, t] == m.posLoad[b, t] + m.posEInGrid[b, t] + m.negEOutLocal[b, t]
    m.E_posNet_cons = en.Constraint(m.BATTS, m.Time, rule=E_pos_net_rule)

    def E_neg_net_rule(m, b, t):
        return m.negNetLoad[b, t] == m.negLoad[b, t] + m.posEInPV[b, t] + m.negEOutExport[b, t]
    m.E_negNet_cons = en.Constraint(m.BATTS, m.Time, rule=E_neg_net_rule)


    return m
