from optimize_battery import BatteryOptiModel
import pandas as pd
import os
import yaml
from pathlib import Path


cwd = Path.cwd()
configs_folder = cwd / 'configs'
datafolder = cwd / 'Data'

with open(configs_folder / 'exp_name.yaml', 'r') as f:
    exp_name = yaml.safe_load(f)['exp_name']


battery_config_path = os.path.join(configs_folder, exp_name, 'battery_config.yaml')


dataset_path = os.path.join(datafolder, 'dataset_gecad_clean.csv')
df = pd.read_csv(dataset_path, index_col=[0, 1])
df.index = df.index.set_levels(df.index.levels[1].astype(int), level=1)

battery_model = BatteryOptiModel(config_path=battery_config_path, env=None)

# Atribui manualmente o dataset e parâmetros mínimos do ambiente
battery_model.env = type("DummyEnv", (), {})()  # cria um objeto vazio tipo classe
battery_model.env.data = df
battery_model.env.Tw = 6  # ou 4, 8, consoante os teus blocos por dia
battery_model.env.allowed_inits = list(range(0, len(df.index.levels[1])))  # ou outro critério
