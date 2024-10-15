#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 18:26:40 2024

@author: omega
"""

import os
from pathlib import Path
import yaml
import argparse
from deepdiff import DeepDiff
from tabulate import tabulate
import colorama
from colorama import Fore, Style

colorama.init()

cwd = Path.cwd()
configs_folder = cwd / 'configs'
algos_config = configs_folder / 'algos_configs'

files = ['agents_config.yaml', 'problem_config.yaml', 'scenario_config.yaml', 
         'apps_config.yaml', 'experiment_config.yaml', 'algos_configs/ppo_config.yaml']

# Function to load a YAML file and return a dictionary
def load_yaml(file_path):
    try:
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        return {}  # Return empty dict if the file doesn't exist
    except yaml.YAMLError:
        print(f"{Fore.RED}Error loading {file_path}{Style.RESET_ALL}")
        return {}

# Function to flatten a nested dictionary
def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

# Helper function to convert non-hashable types like lists and dicts to a string for comparison
def make_hashable(value):
    if isinstance(value, (list, dict)):
        return str(value)  # Convert lists or dictionaries to a string
    return value

# Function to compare multiple YAML files and print differences, handling missing keys and nested dictionaries
def compare_multiple_yaml(file_list):
    yaml_contents = [load_yaml(file) for file in file_list]
    
    # Flatten the nested dictionaries
    flat_yaml_contents = [flatten_dict(content) for content in yaml_contents]

    # Collect all keys across all files (flattened keys)
    all_keys = set()
    for content in flat_yaml_contents:
        if content:
            all_keys.update(content.keys())

    table_data = []
    
    for key in all_keys:
        row = [key]
        values = []
        for yaml_content in flat_yaml_contents:
            value = yaml_content.get(key, 'N/A')
            values.append(make_hashable(value))  # Use make_hashable to handle lists and dicts
            row.append(value)
        
        # Only add the row to the table if there are differences between values
        if len(set(values)) > 1:  # If values differ, add the row to the table
            table_data.append(row)

    # Define table headers based on filenames
    headers = [f"{Fore.GREEN}Variable{Style.RESET_ALL}"] + [f"{Fore.BLUE}{Path(*file.parts[-2:])}{Style.RESET_ALL}" for file in file_list]

    # Print the table only if there are differences
    if table_data:
        print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))
    else:
        print(f"{Fore.YELLOW}\nNo differences found between {', '.join([str(Path(*file.parts[-2:])) for file in file_list])}.{Style.RESET_ALL}")

# Main function to parse arguments and run comparisons
def main():
    print(f"{Fore.RED}This script must be run inside the configs folders{Style.RESET_ALL}")
    parser = argparse.ArgumentParser(description="Compare YAML files in different folders")
    parser.add_argument('folders', nargs='+', help="Experiment names (equivalent to names of folders containing YAML files in the configs folder")

    args = parser.parse_args()

    for file in files:
        files_to_comp = [cwd / Path(folder) / file for folder in args.folders]
        print(f"\n{Fore.GREEN}Comparing {file} across {len(args.folders)} folders:{Style.RESET_ALL}")
        compare_multiple_yaml(files_to_comp)

if __name__ == "__main__":
    main()