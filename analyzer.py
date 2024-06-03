#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 18:23:50 2024

@author: omega
"""

from plots import Plots
from utilities import FolderUtils, utilities
import pandas as pd
from os import path
from pathlib import Path
from dataprocessor import YAMLParser
import seaborn as sns
import matplotlib.pyplot as plt

class Analyzer():
    """
    From results_folder produces an object with all the needed data and methods to perform analysis 
    """
    
    def __init__(self,results_folder,baseline_folder):
        self.folder=results_folder
        self.baseline_folder=baseline_folder
        self.plot=Plots()
        self.get_full_test_data() #import data
        self.baseline=self.get_baseline_costs()
        print(f'Experiment in analysis: {self.folder}')
        print(f'Baseline in analysis: {self.baseline_folder}')
        
    def get_full_test_data(self):
        """
        Scans the results folder and gets metrics and solutions for post analysis and visualization
        """

        files=FolderUtils().get_file_in_folder(self.folder,'.csv')
        for f in files:
            if 'metrics' in f:
                self.metrics=pd.read_csv(f,index_col=0,decimal=',')
                self.metrics['day'] = self.metrics['day'].astype(float).astype(int)
            elif 'env_state' in f:
                self.state=pd.read_csv(f,index_col=0)
            elif 'solutions' in f:
                self.opti_sol=pd.read_csv(f)
            elif 'objectives' in f:
                self.opti_objective=pd.read_csv(f)
            elif 'progress' in f:
                self.progress=pd.read_csv(f)
                
    
    def get_one_day_data(self,day_num,source):
        """extract one day from env_state"""
        utilities.print_info('this is hardcoded for 15 minutes resolution 1 day horizon')
        w=96 #this is hardcoded for 15 minutes resolution 1 day horizon
        
        if source=='opti':
            data=self.opti_sol
        elif source=='rl':
            data=self.state
        elif source=='baseline':
            data=self.baseline_state
            
        t_init=w*day_num
        t_end=t_init+w
        # import pdb
        # pdb.pdb.set_trace()
        return data[(data['tstep'] >= t_init) & (data['tstep'] < t_end)]   
        # return data.iloc[day_num*w:(day_num+1)*w]
        # return self.state.iloc[day_num*w:(day_num+1)*w]
       
    def get_baseline_costs(self):
        baseline_files=FolderUtils().get_file_in_folder(self.baseline_folder,'.csv')
        baseline_costs=[]
        for f in baseline_files:
            if 'metrics' in f:
                self.baseline_metrics=pd.read_csv(f,index_col=0,decimal=',')
                self.baseline_metrics=self.baseline_metrics.drop(columns='season')
                self.baseline_metrics=self.baseline_metrics.astype(float)
                self.baseline_metrics['day'] = self.baseline_metrics['day'].astype(float).astype(int)
                baseline_costs.append(self.baseline_metrics.loc['com'][['day','cost']])
                # df=self.baseline_metrics.join(self.baseline_metrics)
        day=self.baseline_metrics.loc['com']['day']

        df=pd.concat(baseline_costs, axis=1)
        df_costs=df.drop(columns='day')
        df['cost_mean_base']=df_costs.mean(axis=1)
        df=df['cost_mean_base']
        # df=pd.merge(df,day,left_index=df.index)
        df_final=pd.concat([df,day],axis=1)
        
        return df_final
        
    def get_cost_compare(self):
        """
        creates a dataframe that stores cost values between optimal solution 
        and rl agent
        """

        df = pd.merge(self.metrics.loc['com'], self.opti_objective, 
                         on='day', 
                         how='inner')
        
        df = pd.merge(df,self.baseline, on='day', how='inner')
        
        df_compare=df[['day','cost','objective','cost_mean_base','x_ratio']]
        df_compare=df_compare.copy()
        df_compare=df_compare.astype(float)
        
        df_compare.loc[:, 'dif']=((df_compare['cost'].astype(float)-df_compare['objective'])/(df_compare['cost_mean_base']))*100
        

        utilities().print_info('Random baseline cost is used in indicators')
        
        df_compare.loc[:, 'save_rate']=((df_compare['cost_mean_base']-df_compare['cost'])/(df_compare['cost_mean_base']-df_compare['objective']))*100
        
        df_compare.loc[:, 'sigma2_0']=abs((df_compare['objective']-df_compare['cost']))/df_compare['cost_mean_base']
        df_compare.loc[:, 'sigma2']=(1-df_compare['sigma2_0'])*100

        
        
        
        return df_compare
        
    
    def get_per_agent_costs(self):
        data = []
        indexes=self.metrics.index.unique()
        
        for i in indexes:
            # Extract data for each unique index value
            df = self.metrics.loc[i, ['cost', 'test_epi']].copy()
            # Rename the 'cost' column to the unique index value
            df.rename(columns={'cost': i}, inplace=True)
            # Set 'test_epi' as index
            df.set_index('test_epi', inplace=True)
            data.append(df)
        
        # Concatenate all DataFrames along the columns axis
        df_final = pd.concat(data, axis=1)
        
        #erase the com column
        df_final.drop(columns=['com'], inplace=True)
        df_final=df_final.astype(float)
        df_final = df_final.round(2)
        
        utilities().print_info('hardcoded for the 3 agents and the specific testing community// Need to get community info in analyse phase')
        
        # import pdb
        # pdb.pdb.set_trace()
        w=pd.DataFrame([3.6,3.0,3.0],index=df_final.columns,columns=['En'])
        
        new_cols=[]
        for ag in df_final.columns:
            new_col=ag+'_E'
            new_cols.append(new_col)
            df_final[new_col]=df_final[ag]/w.loc[ag]['En']
        
        
        # file_name=str(self.folder / f'cost_compare_{len(indexes)}')
        # self.plot.make_compare_plot2(df_final,filename_save=file_name)
        
        return df_final
    
    
    
    def plot_joint(self,x,y, save=False):
        data=self.get_cost_compare()
        file_name=None
        if save:
            file_name=str(self.folder / f'Joint_Plot_Indicator_{x}_{len(data)}')
        
        self.plot.make_jointplot(data,x,y,filename_save=file_name)
        
    
    
    def plot_one_day(self,day_num,source):
        data=self.get_one_day_data(day_num, source)
        
        file_name=str(self.folder / ('profile_' + f'day_{day_num}_'+f'source_{source}'))
        self.plot.makeplot_bar(data, file_name)
        


    def plot_cost_hist(self,save=False):
        data=self.get_per_agent_costs()
        
        for k in data.columns:
            if 'E' not in k:
                data=data.drop(columns=k)
                
        file_name=None
        if save:
            file_name=str(self.folder / 'per_agent_cost_histogram.png')
        
        
        self.plot.make_histogram(data, file_name)
        

            
            
            
        
        # data=[]
        # for i in self.metrics.index.unique():
        #     df=self.metrics.loc[i, ['cost', 'test_epi']]
        #     df.rename(columns={'cost': i}, inplace=True)
        #     data.append(df)
    
        # for k in data:
        #     k.set_index('test_epi', inplace=True)
            
        # df_final=pd.concat(data, axis=1)
        # return df_final


 

   
class AnalyzerMulti():
    "Class to compare experiments"
    def __init__(self,results_config,name):
        """gets a list of analyzer experiment objects and performs comparison between experiments"""
        self.results_folder=Path.cwd() / 'Results'
        self.name=name
        self.config=YAMLParser().load_yaml(results_config)[self.name]
        self.descript=self.config['description']
        self.baseline=self.results_folder / self.config['baseline_name']
        self.analyser_objs=self.get_analyser_objects()
        self.plot=Plots()
        self.compare_results_folder=self.results_folder / self.name
        FolderUtils().make_folder(self.compare_results_folder)
        
    def get_experiments_folders(self):
        exp_folders={}
        exps=self.config['experiments']
        for k in exps:
            exp_folders[exps[k]['label']]=exps[k]['folder']
        return exp_folders
    
    def get_analyser_objects(self):
        exp_dict=self.get_experiments_folders()
        multi_analyser={}
        for label in exp_dict:
            exp_dict[label]=Analyzer(self.results_folder / exp_dict[label],self.baseline)
        
        return exp_dict
    
    def get_multi_cost_compare(self):
        data={}
        for label in self.analyser_objs:
            data[label]=self.analyser_objs[label].get_cost_compare()
            
        return data
        
    
    
    def plot_multi_joint(self,x,y, save=False):        
        data=self.get_multi_cost_compare()
 
        file_name=None
        if save:
            file_name=str(self.compare_results_folder / f'Joint_Plot_Indicator_{x}_{len(data)}')
        
        self.plot.make_multi_jointplot(data,x,y,filename_save=file_name)
        
    
    def plot_per_agent_cost_multi_hist(self,save=False):
        data={}
        for obj_id in self.analyser_objs:
            
            data_agent=self.analyser_objs[obj_id].get_per_agent_costs()
            
            for k in data_agent.columns:
                if 'E' not in k:
                    data_agent=data_agent.drop(columns=k)
                    
            data[obj_id]=data_agent

        file_name=None
        if save:
            file_name=str(self.compare_results_folder / 'per_agent_cost_histogram.png')
        
        
        self.plot.make_multi_histogram(data, file_name)
        
      
    def plot_year_cost(self,save=False):
        data=self.get_multi_cost_compare()
        
        df=pd.DataFrame()
        for obj_id in data:
            df[obj_id]=data[obj_id]['cost']
        df['opti']=data[obj_id]['objective']
        df['rand']=self.analyser_objs[obj_id].get_baseline_costs()['cost_mean_base'].values
        
        data_sorted=df.mean().sort_values()
        
        file_name=None
        if save:
            file_name=str(self.compare_results_folder / 'models_barplot.png')
        
        self.plot.plot_barplot(data_sorted,file_name)
        
            
        
        
                
        
        
        
        
