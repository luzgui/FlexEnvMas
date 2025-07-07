#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 14:59:38 2024

@author: omega
"""
import os
import inspect
from pathlib import Path
from termcolor import colored



class utilities:
    @staticmethod
    def print_info(message):
        frame = inspect.currentframe().f_back
        line_number = frame.f_lineno
        function_name = frame.f_code.co_name
        
        print(colored(message,'red'))
        print(colored(f"@ '{function_name}' at line {line_number}",'green'))
        
    @staticmethod    
    def get_exp_from_results_name(folder_name: str) -> dict:
        parts = folder_name.split('_')
        result={'train_exp': parts[1],
                'test_exp':parts[3]}
        
        return result
        
        
        

class ConfigsParser():
    def __init__(self,configs_folder, exp_name):
        self.folder=configs_folder
        self.exp_name=exp_name
        self.make_configs()
        print(colored(f'Experiment Name ---> {self.exp_name}','red'))
        
    def traverse_folder(self):
        result = {}
        for root, dirs, files in os.walk(self.folder):
            # Initialize nested dictionary for subfolder
            current_dict = result
            for subdir in os.path.relpath(root, self.folder).split(os.path.sep):
                if subdir not in current_dict:
                    current_dict[subdir] = {}
                current_dict = current_dict[subdir]
            # Add files to subfolder dictionary
            for file in files:
                file_name, file_extension = os.path.splitext(file)
                current_dict[file_name] = file
        return result
     
    
    def make_configs(self):
        """
        Output order
        - agents_config, 
        - apps_config
        - scene_config
        - problem_config
        - vars
        - experiment_config
        - self.algo_config_file
        
        exp_name must be a folder name or wwill return an error
        """
        from utils.dataprocessor import YAMLParser #import here due to circular error
        # import pdb
        # pdb.pdb.set_trace()
        files=self.traverse_folder()
        files=files[self.exp_name]
        self.exp_folder=Path(self.folder) / self.exp_name
        
        self.experiment_config=self.exp_folder / files['experiment_config']
        algo_config_file=YAMLParser().load_yaml(self.experiment_config)['algorithm']['config']
        self.algo_config_file=self.exp_folder /'algos_configs' / algo_config_file
        self.agents_config=self.exp_folder / files['agents_config']
        self.apps_config=self.exp_folder / files['apps_config']
        self.scene_config=self.exp_folder / files['scenario_config']
        self.problem_config=self.exp_folder / files['problem_config']
        self.state_vars=self.exp_folder / files['state_vars']
        
        
    def get_configs(self):
        """
        Returns the config files existing in the config folder of the experiment
        """
        return self.agents_config, self.apps_config, self.scene_config, self.problem_config, self.state_vars, self.experiment_config,self.algo_config_file
    
    def print_experiment_info(self):
        from dataprocessor import YAMLParser #import here due to circular error
        msg=YAMLParser().load_yaml(self.experiment_config)['info']
        print(colored(msg,'red'))
        
        

class FolderUtils():
    @staticmethod
    def get_file_in_folder(folder_path, file_type):
        """
        Scan a folder and return a list of CSV files in it.
        
        Args:
        - folder_path (str): The path to the folder to scan.
        
        Returns:
        - csv_files (list): A list of CSV files found in the folder.
        """
        csv_files = []
        for file in os.listdir(folder_path):
            if file.endswith(file_type):
                csv_files.append(os.path.join(folder_path, file))
        return csv_files
    
    @staticmethod
    def make_folder(folder):
        if not os.path.exists(folder) and not os.path.isdir(folder):
            os.makedirs(folder)
            print(colored('folder created' ,'red'),folder)
    
    @staticmethod
    def get_csv_files_in_subfolders(folder_path, file_type='.csv'):
        """Scan a folder and its subfolders, returning a dictionary of CSV files found,
        using folder names as part of the unique keys.
        
        Args:
        - folder_path (str): The path to the folder to scan.
        - file_type (str): The file extension to look for (default is '.csv').
        
        Returns:
        - csv_files_dict (dict): A dictionary where the key is a unique identifier 
          based on the folder name, and the value is the full path to the CSV file.
        """
        csv_files_dict = {}
    
        # Walk through the folder and its subfolders
        for root, _, files in os.walk(folder_path):
            # Check if the current directory is not the base folder
            if root != folder_path:
                # Extract the folder name from the path
                folder_name = os.path.basename(root)
                
                for file in files:
                    if file.endswith(file_type):
                        # Create a unique key using the folder name and file name
                        unique_key = f'{folder_name}'
                        csv_files_dict[unique_key] = os.path.join(root, file)
    
        return csv_files_dict
    
    @staticmethod
    def get_subfolders(folder_path):
        """
        Scan a folder and return a list of all its subfolders.
        
        Args:
        - folder_path (str): The path to the folder to scan.
        
        Returns:
        - subfolders (list): A list of paths to the subfolders found in the folder.
        """
        subfolders = []
        
        # Iterate through the directory entries
        for entry in os.listdir(folder_path):
            full_path = os.path.join(folder_path, entry)
            if os.path.isdir(full_path):  # Check if it's a directory
                subfolders.append(full_path)
    
        return subfolders


# parser=YAMLParser()
# parser.write_yaml(file_experiment,'exp_name','cenas')

# import os






        
    





    
 