#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 14:37:41 2022

@author: omega
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy.random as rnd
import time
import random as rnd


def make_cyclical(series, max_val): # transforma valores como dia e hora em valores cíclicos de sin e cos para remover efeitos indesejados
    data_sin = np.sin( 2 * np.pi * series / max_val )
    data_cos = np.cos( 2 * np.pi * series / max_val )
    return list(data_sin), list(data_cos)


s=np.sin( minutes 1410)





