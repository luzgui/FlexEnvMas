#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 15:25:14 2022

@author: omega
"""
from shiftenvRLlib_mas import ShiftEnvMas
from ray.rllib.utils.pre_checks import env


def make_env(environment_config):
    menv=ShiftEnvMas(environment_config) 
    env.check_env(menv)
    
    
    return menv


