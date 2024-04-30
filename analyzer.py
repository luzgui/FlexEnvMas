#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 18:23:50 2024

@author: omega
"""

from plots import Plots
from utilities import FolderUtils
import pandas as pd


class Analyzer():
    def __init__(self,results_folder):
        self.folder=results_folder
        self.plot=Plots()
        self.get_full_test_data() #import data
        
    def get_full_test_data(self):
        """
        Scans the results folder and gets metrics and solutions for post analysis and visualization
        """

        files=FolderUtils().get_file_in_folder(self.folder,'.csv')
        for f in files:
            if 'metrics' in f:
                self.metrics=pd.read_csv(f)
            elif 'env_state' in f:
                self.state=pd.read_csv(f,index_col=0)
    
    def get_one_day_data(self,day_num):
        """extract one day from env_state"""
        w=96 #this is hardcoded for 15 minutes resolution 1 day horizon
        return self.state.iloc[day_num*w:(day_num+1)*w]
        
        
class AnalyzerMulti():
    def __init__(self,analyzer_list):
        """gets a list of analyzer experiment objects and performs comparison between experiments"""
        self.analyzer_list=analyzer_list
                
        
        
        
        
