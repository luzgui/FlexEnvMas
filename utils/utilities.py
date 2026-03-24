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
import re

# from env.environment import FlexEnv
# from env.environment_v1 import FlexEnvV1



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
        """
        Extracts train_exp, train_env and test_exp from a folder name.
        Structure: Train_{exp-name}_{env-name}_Test_{test-name}
        Assumes env names follow the pattern 'env' followed by digits e.g. env000.
        
        Returns a dictionary with keys: 'train_exp', 'train_env', 'test_exp'
        """
        parts = folder_name.split('_')

        try:
            train_index = parts.index('Train')
            test_index  = parts.index('Test')
        except ValueError:
            train_index = 0
            test_index  = 2

        train_parts = parts[train_index + 1:test_index]

        # Find the env segment by matching the 'env' + digits pattern
        env_index = next((i for i, p in enumerate(train_parts) if 'env' in p), None)
        # import pdb
        # pdb.pdb.set_trace()
        if env_index is None:
            raise ValueError(f"No env name found in '{folder_name}'. Expected pattern: envXXX")
        
        train_exp = '_'.join(train_parts[:env_index])
        train_env = train_parts[env_index]
        test_exp  = '_'.join(parts[test_index + 1:])

        return {'train_exp': train_exp, 'train_env': train_env, 'test_exp': test_exp}

    
    @staticmethod
    def get_num_from_str(text):
        # This regex matches integers and floats (e.g., 123, 45.67)
        pattern = r'\d+\.\d+|\d+'
        numbers = re.findall(pattern, text)
        # Convert matched strings to float if they contain a dot, else int
        result = [float(num) if '.' in num else int(num) for num in numbers]
        return result
        
        
        

class ConfigsParser():
    
    ENV_CONFIG_FILES = {'agents_config', 
                        'apps_config', 
                        'scenario_config', 
                        'problem_config', 
                        'state_vars'}
    
    EXP_CONFIG_FILES = {'experiment_config'}
    
    def __init__(self,configs_folder, exp_name):
        self.folder=configs_folder
        self.exp_name=exp_name
        self._check_exp_exists()
        self.type=self._folder_type()
        self.make_configs()
        
        
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
     
        
     
        
    def _folder_type(self):
        """
        Identify whether a folder is an 'environment', 'experiment', or 'unknown'
        config folder by checking which expected config files are present.
        """
        
        if 'env' in self.exp_name:
            return 'environment'
        if 'eval' in self.exp_name:
            return 'environment'
        else:
            return 'experiment'

    def _check_exp_exists(self):
        """Locate a named subfolder within the experiment folder."""
        path = self.folder / self.exp_name
        if not path.exists():
            raise FileNotFoundError(
                f"Expected '{self.exp_name}' subfolder not found in '{self.folder}'.")
    
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
        
        
        
        if self.type=='experiment':
            self.experiment_config=self.exp_folder / files['experiment_config']
            algo_config_file=YAMLParser().load_yaml(self.experiment_config)['algorithm']['config']
            self.algo_config_file=self.exp_folder /'algos_configs' / algo_config_file
            print(colored(f'Experiment Name ---> {self.exp_name}','red'))
            
        elif self.type=='environment':
            self.agents_config=self.exp_folder / files['agents_config']
            self.apps_config=self.exp_folder / files['apps_config']
            self.scene_config=self.exp_folder / files['scenario_config']
            self.problem_config=self.exp_folder / files['problem_config']
            self.state_vars=self.exp_folder / files['state_vars']
            print(colored(f'Environment Name ---> {self.exp_name}','red'))
        
        
    def get_configs(self):
        """
        Returns the config files existing in the config folder of the experiment
        """
        
        if self.type=='experiment':
            return self.experiment_config,self.algo_config_file
            
        elif self.type=='environment':
            return self.agents_config, self.apps_config, self.scene_config, self.problem_config, self.state_vars
        
        
    
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
    
    @staticmethod
    def get_subfolders_containing(folder_path: str, keywords: list[str]) -> list[str]:
        """
        Scan subfolders and return those whose names contain all the given keywords.
    
        Args:
        - folder_path (str): The path to the folder to scan.
        - keywords (list[str]): A list of strings to look for in subfolder names.
    
        Returns:
        - matches (list[str]): A list of subfolder paths whose names contain all keywords.
        """
        matches = []
        for entry in os.listdir(folder_path):
            full_path = os.path.join(folder_path, entry)
            if os.path.isdir(full_path):
                if all(keyword in entry for keyword in keywords):
                    matches.append(entry)
                    
        if len(matches) == 0:
            raise FileNotFoundError(f"No subfolder containing {keywords} found in '{folder_path}'.")
        if len(matches) > 1:
            raise ValueError(f"Expected a unique match for {keywords} but found {len(matches)}: {matches}")

        return matches[0]


# parser=YAMLParser()
# parser.write_yaml(file_experiment,'exp_name','cenas')

# import os






        
    





    
 