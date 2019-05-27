# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 18:24:53 2018

@author: abhis
"""
import numpy as np
list=[[1,2,3,4,5,6,7,8,9],[1,1,1,1,1,1,1,1,1]]

x=np.array(list)
x, y = x[0: 1], x[1: 2]
result=np.abs(x-y)
#x=np.sub(x)
#x=np.abs(x)