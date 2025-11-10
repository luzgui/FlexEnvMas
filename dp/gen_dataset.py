#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 09:35:26 2025

@author: omega

This script generates Differentially private datasets using the DPEnergyData class

"""
from pathlib import Path
from dp.diffpriv import DPEnergyData


cwd=Path.cwd()
datafolder=cwd.parent / 'Data'
configs_folder=cwd.parent / 'configs'
algos_config = configs_folder / 'algos_configs'
resultsfolder=cwd.parent / 'Results'

config=configs_folder / 'dp' / 'dp_config.yaml'
file=datafolder / 'dataset_gecad_clean.csv'



dp=DPEnergyData(file, config)
dp.save_data(num_files=20)
#%% 

# dp.laplace_mechanism_trunc_day()

# daily_stats=dp.get_daily_data(dp.data)

# data_Adj=dp.get_adjancent_data()
# data=dp.data
# noisy_data=dp.apply_dp_noise()

# noise=dp.laplace_mechanism()
# noise=dp.laplace_mechanism_year()

# noise=dp.laplace_mechanism_day()

# daily_stats=dp.get_daily_data(dp.data)
# dp.plots_compare(52, 'ag1')


# dataset_gecad_clean_laplace_day_sens_s_max_eps_0.06.csv
# dataset_gecad_clean_laplace_day_sens_s_max_eps_0.08.csv
# dataset_gecad_clean_laplace_day_sens_s_max_eps_0.1.csv
# dataset_gecad_clean_laplace_day_sens_s_max_eps_0.2.csv
# dataset_gecad_clean_laplace_day_sens_s_max_eps_0.4.csv
# dataset_gecad_clean_laplace_day_sens_s_max_eps_0.6.csv
# dataset_gecad_clean_laplace_day_sens_s_max_eps_0.8.csv
# dataset_gecad_clean_laplace_day_sens_s_max_eps_10.0.csv
# dataset_gecad_clean_laplace_day_sens_s_max_eps_1.0.csv
# dataset_gecad_clean_laplace_day_sens_s_max_eps_20.0.csv
# dataset_gecad_clean_laplace_day_sens_s_max_eps_2.0.csv
# dataset_gecad_clean_laplace_day_sens_s_max_eps_3.0.csv
# dataset_gecad_clean_laplace_day_sens_s_max_eps_4.0.csv
# dataset_gecad_clean_laplace_day_sens_s_max_eps_5.0.csv
# dataset_gecad_clean_laplace_day_sens_s_max_eps_8.0.csv

# files=['dataset_gecad_clean.csv',
#        'dataset_gecad_clean_laplace_day_sens_s_max_eps_10.0.csv',
#        'dataset_gecad_clean_laplace_day_sens_s_max_eps_5.0.csv',
#        'dataset_gecad_clean_laplace_day_sens_s_max_eps_4.0.csv',
#        'dataset_gecad_clean_laplace_day_sens_s_max_eps_3.0.csv',
#        'dataset_gecad_clean_laplace_day_sens_s_max_eps_2.0.csv']

# files=['dataset_gecad_clean.csv',
#        'dataset_gecad_clean_laplace_day_sens_s_max_eps_10.0.csv',
#        'dataset_gecad_clean_laplace_day_sens_s_max_eps_5.0.csv']


# files=['dataset_gecad_clean.csv',
#        'dataset_gecad_clean_laplace_day_sens_s_max_eps_1.0.csv',
#        'dataset_gecad_clean_laplace_day_sens_s_max_eps_0.8.csv',
#        'dataset_gecad_clean_laplace_day_sens_s_max_eps_0.6.csv',
#        'dataset_gecad_clean_laplace_day_sens_s_max_eps_0.4.csv',
#        'dataset_gecad_clean_laplace_day_sens_s_max_eps_0.2.csv']


# files=['dataset_gecad_clean.csv',
#        'dataset_gecad_clean_laplace_day_sens_s_max_eps_10.0_clip.csv',
#        'dataset_gecad_clean_laplace_day_sens_s_max_eps_5.0_clip.csv',
#        'dataset_gecad_clean_laplace_day_sens_s_max_eps_4.0_clip.csv',
#        'dataset_gecad_clean_laplace_day_sens_s_max_eps_3.0_clip.csv',
#        'dataset_gecad_clean_laplace_day_sens_s_max_eps_2.0_clip.csv']

# files=['dataset_gecad_clean.csv',
#        'dataset_gecad_clean_laplace_day_sens_s_max_eps_10.0_clip.csv',
#        'dataset_gecad_clean_laplace_day_sens_s_max_eps_5.0_clip.csv']

# files=['dataset_gecad_clean.csv',
#        'dataset_gecad_clean_laplace_day_sens_s_max_eps_5.0_clip.csv',
#        'dataset_gecad_clean_laplace_day_sens_s_max_eps_1.0_clip.csv',
#        'dataset_gecad_clean_laplace_day_sens_s_max_eps_0.2_clip.csv']


# files=['dataset_gecad_clean.csv',
#        'dataset_gecad_clean_laplace_day_sens_s_max_eps_1.0_clip.csv',
#        'dataset_gecad_clean_laplace_day_sens_s_max_eps_0.8_clip.csv',
#        'dataset_gecad_clean_laplace_day_sens_s_max_eps_0.6_clip.csv',
#        'dataset_gecad_clean_laplace_day_sens_s_max_eps_0.4_clip.csv',
#        'dataset_gecad_clean_laplace_day_sens_s_max_eps_0.2_clip.csv']


# dp.plots_multi_compare(0, 'ag1', files)




# day_d_a=dp.get_daily_data(dp.get_adjancent_data())

# ag1=dp.data.loc[0:96]['ag1']
# ag1_a=data_Adj.loc[0:96]['ag1']

# np.random.laplace(0.0,0.0)


