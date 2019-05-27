# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 17:30:43 2018

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


#import csv file

df=pd.read_csv('father_son.csv')    
#prints first 10 rows


x_train=df[['a']]
y_train=df[['b']]


#train the model 
lm=LinearRegression()
lm.fit(x_train,y_train)



#test the model
no=[[65],[63.2],[62.5],[66.5],[70.4],[60]]
predictions=lm.predict(no)
print(predictions)

#plot the best fit line
plt.scatter(x_train,y_train,color='r')

plt.plot(no,predictions,color='black',linewidth=3)
plt.xlabel('Father height in inches')
plt.ylabel('Son height in inches')
plt.show()


# prepare training data


#x=people.father_stature
#y=people.son_stature
#x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=1)





