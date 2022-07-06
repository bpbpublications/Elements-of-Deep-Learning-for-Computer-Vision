# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 19:16:15 2020

@author: bhara
"""

import matplotlib.pyplot as plt 
import numpy as np 
import math 
  
x = np.linspace(-10, 10, 100) 
b = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
plt.plot(x, b) 
plt.xlabel("x") 
plt.ylabel("tanh(x) or z") 
  
plt.show()