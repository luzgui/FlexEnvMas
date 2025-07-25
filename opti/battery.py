import numpy as np

class Battery:
    def __init__(self, battery_id=None,
                 current_charge=None,
                 capacity=0.0,
                 charging_power_limit=0.0,
                 discharging_power_limit=-0.0,
                 charging_efficiency=1.00,
                 discharging_efficiency=1.00,
                 initial_soc_percent=0.0):
        self.battery_id = battery_id
        self.capacity = capacity
        self.charging_power_limit = charging_power_limit
        self.discharging_power_limit = discharging_power_limit
        self.charging_efficiency = charging_efficiency
        self.discharging_efficiency = discharging_efficiency
        self.initial_soc_percent = initial_soc_percent

        if current_charge is not None:
            self.current_charge = current_charge
        else:
            self.set_initial_charge()

    def set_initial_charge(self):
        self.current_charge = self.capacity * self.initial_soc_percent


class Battery_ob:
    def __init__(self,
                 cycles=3000.0,
                 maxLi=30.0,
                 installedCap=10.0,  # kWh
                 maxDOD=0.85,         # % equivalent full cycle (EFC)
                 EoL=0.8,             # end of life % equivalent full charge
                 minSOC=0.0,         # kWh
                 maxChg=5.0,         # kW
                 maxDisChg=-5.0,     # kW
                 etaChg=0.9,
                 etaDisChg=0.9):

        self.cycles = cycles
        self.maxLi = maxLi
        self.installedCap = installedCap  # kWh
        self.maxDOD = maxDOD              # % equivalent full cycle (EFC)
        self.EoL = EoL                     # end of life % equivalent full charge
        self.maxSOC = installedCap        # kWh
        self.minSOC = minSOC              # kWh
        self.maxChg = maxChg              # kW
        self.maxDisChg = maxDisChg        # kW
        self.etaChg = etaChg
        self.etaDisChg = etaDisChg
