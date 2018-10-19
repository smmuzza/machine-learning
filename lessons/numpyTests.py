# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 22:53:38 2018

@author: shane
"""

import numpy as np
import pandas as pd

test = np.array([np.arange(100), np.random.randint(1, 11, 100)]).transpose()
#test[:10, :]
display(test[-10:,])
