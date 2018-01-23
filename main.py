# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 20:46:16 2018

@author: Trenton
"""

import sklearn
import numpy as np
import matplotlib.pyplot as plot
import pandas as pd
from os import chdir
from datetime import datetime

chdir('E:\\Documents\\programs\\Python\\traffic fatality')

data = pd.read_csv('NYPD_Motor_Vehicle_Collisions.csv', nrows = 100000)
"""
wrangling/preprocessing data
"""

#label columns to aid pre processing
counter = 0
columns = list(data.columns.values)
for i in range(len(columns)):
    columns.insert(i+counter, i)
    counter += 1

#print(columns) #candidates for features: 0, 1, 2, 10, y is 11
X = data[['TIME', 'BOROUGH', 'NUMBER OF PERSONS INJURED', 'CONTRIBUTING FACTOR VEHICLE 1']].copy()
Y = data[['NUMBER OF PERSONS KILLED']].copy()


#pre processing y value
Y=Y.rename(columns = {'NUMBER OF PERSONS KILLED':'fatality'})

#remove multiple fatalities so only a binary prediction exists (one or more fatalities or no fatalities)
for index, row in Y.iterrows():
    if row.loc['fatality'] > 1:
        #print('corrected fatality ' + str(index))
        row.loc['fatality'] = 1


#convert borough to numeric - Bronx = 0, Brooklyn = 1, Manhattan = 2, Queens = 3, Empty = 4
for column in X:
    if column != 'BOROUGH':
        continue
    for i in range(len(X[column])):
        if X[column][i] == 'BRONX':
            X.set_value(i, 'BOROUGH', 0)
        elif X[column][i] == 'BROOKLYN':
            X.set_value(i, 'BOROUGH', 1)
        elif X[column][i] == 'MANHATTAN':
            X.set_value(i, 'BOROUGH', 2)
        elif X[column][i] == 'QUEENS':
            X.set_value(i, 'BOROUGH', 3)
        else:
            X.set_value(i, 'BOROUGH', 4)

#convert contributing factor to numeric
for column in X:
    if column != 'CONTRIBUTING FACTOR VEHICLE 1':
        continue
    for i in range(len(X[column])):
        if X[column][i] == 'Passing or Lane Usage Improper':
            X.set_value(i, 'CONTRIBUTING FACTOR VEHICLE 1', 0)
        elif X[column][i] == 'Unspecified':
            X.set_value(i, 'CONTRIBUTING FACTOR VEHICLE 1', 1)
        elif X[column][i] == 'Alcohol Involvement':
            X.set_value(i, 'CONTRIBUTING FACTOR VEHICLE 1', 2)
        elif X[column][i] == 'Glare':
            X.set_value(i, 'CONTRIBUTING FACTOR VEHICLE 1', 3)
        elif X[column][i] == 'Driver Inattention/Distraction':
            X.set_value(i, 'CONTRIBUTING FACTOR VEHICLE 1', 4)
        elif X[column][i] == 'Other Vehicular':
            X.set_value(i, 'CONTRIBUTING FACTOR VEHICLE 1', 5)
        elif X[column][i] == 'Following Too Closely':
            X.set_value(i, 'CONTRIBUTING FACTOR VEHICLE 1', 6)
        elif X[column][i] == 'Backing Unsafely':
            X.set_value(i, 'CONTRIBUTING FACTOR VEHICLE 1', 7)    
        elif X[column][i] == 'Passing or Lane Usage Improper':
            X.set_value(i, 'CONTRIBUTING FACTOR VEHICLE 1', 8)  
        elif X[column][i] == 'Failure to Yield Right-of-Way':
            X.set_value(i, 'CONTRIBUTING FACTOR VEHICLE 1', 9)
        elif X[column][i] == 'Traffic Control Disregarded':
            X.set_value(i, 'CONTRIBUTING FACTOR VEHICLE 1', 10)  
        else:
            X.set_value(i, 'CONTRIBUTING FACTOR VEHICLE 1', 11)
            
#late night = 1, otherwise = 0            
for column in X:
    if column != 'TIME':
        continue
    for i in range(len(X[column])):
        t = datetime.strptime(X[column][i], '%H:%M')
        if int(t.hour) >= 22 or int(t.hour) <= 6:
            X.set_value(i, 'TIME', 1)
        else:
            X.set_value(i, 'TIME', 0)

print(X)