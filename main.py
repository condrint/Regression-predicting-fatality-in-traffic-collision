# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 20:46:16 2018

@author: Trenton
"""

import sklearn
import numpy as np
import matplotlib.pyplot as plot
import pandas as pd
import os

os.chdir('E:\\Documents\\programs\\traffic fatality')

data = pd.read_csv('NYPD_Motor_Vehicle_Collisions.csv', nrows = 1000)


#label columns
counter = 0
columns = list(data.columns.values)
for i in range(len(columns)):
    columns.insert(i+counter, i)
    counter += 1

#print(columns) #candidates for features: 0, 1, 2, 10, y is 11

X = data[['DATE', 'TIME', 'BOROUGH', 'NUMBER OF PERSONS INJURED', 'CONTRIBUTING FACTOR VEHICLE 1']].copy()
Y = data[['NUMBER OF PERSONS KILLED']].copy()
#print(X)

#pre processing y value
Y=Y.rename(columns = {'NUMBER OF PERSONS KILLED':'fatality'})
#remove multiple fatalities so only a binary prediction exists (one or more fatalities or no fatalities)
for index, row in Y.iterrows():
    if row.loc['fatality'] > 1:
        #print('corrected fatality ' + str(index))
        row.loc['fatality'] = 1

