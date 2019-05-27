# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 20:25:16 2018

@author: abhis
"""

#given height of father predict height of son

# reading dataset
import pandas as pd

#mathematical computation 
import numpy as np

#plot graph
from matplotlib import pyplot as plt

#linear regression model
from sklearn.linear_model import LinearRegression

#for data regularization 
import math


#import csv file
df=pd.read_csv('insurance.csv')    

#replace string data with boolean data 
df['sex'].replace('female',0,inplace=True)
df['sex'].replace('male',1,inplace=True)

df['smoker'].replace('no',0,inplace=True)
df['smoker'].replace('yes',1,inplace=True)


#print("For male=1 female=0 ")
#print("For smoker=1 non smoker=0 \n")

#if data missing put median 
median_age=math.floor(df.age.median())
df.age=df.age.fillna(median_age)

reg=LinearRegression()
reg.fit(df[['age','sex','bmi','children','smoker']],df.charges)

a=[[18,0,25.5,0,0]]
x=float(reg.predict(a))
print('The insurance premium for you is ${}'.format(x))


plt.scatter(df.age,df.charges,color='r')

plt.plot(a,x,color='blue',linewidth=3)
plt.xlabel('Age')
plt.ylabel('Insurance Premium')
plt.show()


