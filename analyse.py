#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 18:13:37 2024

@author: omega
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 11:05:42 2024

@author: omega
"""



#math + data
import pandas as pd
import numpy as np

#Plotting
import matplotlib.pyplot as plt
import seaborn as sns


#System
import os
from os import path
from pathlib import Path
import sys
import time
import datetime
from datetime import datetime

#Custom functions
from plotutils import *



import random


#pyomo
from auxfunctions_opti import *
from pyomo.environ import *
from pyomo.opt import SolverFactory
import scipy.io as sio
import re 
from itertools import compress
from rich import print
from rich.console import Console
from rich.syntax import Syntax

from experiment_test import ExperimentTest
from testenv import TestEnv
from trainable import Trainable
from utilities import ConfigsParser, FolderUtils
from dataprocessor import YAMLParser
from community import Community
from state import StateVars
from environment import FlexEnv
from analyzer import Analyzer
#paths

cwd=Path.cwd()
datafolder=cwd / 'Data'
raylog=cwd / 'raylog'
prof_folder=raylog / 'profiles'
resultsfolder=cwd / 'Results'
storage_path='/home/omega/Downloads/ShareIST'

configs_folder=cwd / 'configs'

#%% 
from analyzer import Analyzer
from plots import Plots
folder_name= resultsfolder / YAMLParser().load_yaml(configs_folder / 'results_config.yaml')['folder_name']
analyse=Analyzer(folder_name)

one_day=analyse.get_one_day_data(day_num=0)

metrics=analyse.metrics
state=analyse.state

#%%5
plot=Plots()
# plot.plot_energy_usage(one_day,filename_save=folder_name / 'pic2') #
plot.makeplot_bar(one_day,filename_save=folder_name / 'pic_bar')

#%%

