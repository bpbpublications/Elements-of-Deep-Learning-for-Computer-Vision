# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 20:13:44 2020

@author: bhara
"""

import matplotlib.pyplot as plt 
import numpy as np 
import math 
  
x = np.linspace(-10, 10, 100) 
r = np.maximum(x,0)
plt.plot(x, r) 
plt.xlabel("x") 
plt.ylabel("Relu(x) or r") 
  
plt.show()