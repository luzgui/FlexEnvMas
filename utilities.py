#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 14:59:38 2024

@author: omega
"""

import inspect
from termcolor import colored



class utilities:
    @staticmethod
    def print_info(message):
        frame = inspect.currentframe().f_back
        line_number = frame.f_lineno
        function_name = frame.f_code.co_name
        
        print(colored(message,'red'))
        print(colored(f"@ '{function_name}' at line {line_number}",'green'))





    
 