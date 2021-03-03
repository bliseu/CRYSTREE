#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 18:48:05 2020

@author: bli
"""
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV


ESTIMATORS = {
    "Extra_trees": ExtraTreesRegressor(n_estimators=100,
                                       random_state=0),
    #"Extra_trees": ExtraTreesRegressor(n_estimators=10, max_features=32,
    #                                   random_state=0),
    "K-nn": KNeighborsRegressor(),
    "Linear_regression": LinearRegression(),
    "Ridge": RidgeCV(),
    "Multirf": MultiOutputRegressor(RandomForestRegressor(n_estimators=100,
                                                          max_depth=500,
                                                          random_state=550))
}

if __name__ == '__main__':
    try:
        error = 521314159.0
        max_depth = 500
        
        # use sorted_entire_struct.csv 
        data1 = pd.read_csv('sorted_entire_struct.csv', header=0,index_col=0)
        data = data1.dropna()
        row = data.shape[0]
        column = data.shape[1]
        #print (data.shape)
        content = np.array(data)
        content = content.reshape (row, column)
        
        nX = int((column-1)/2+1) 
        X = content[:,:nX]
        y = content[:,nX:]
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)  
        finput = "input.txt"
        for lnum, line in enumerate(open(finput)):
             if 'Popsize' in line:
                 popsize = int(line.strip().split()[2])
             if 'Ratio' in line:
                 ratio = float(line.strip().split()[2])
        NN=int(popsize*ratio) 
        MM=int(popsize*0.9) 
        X_train=X[0:MM]
        X_test=X[0:MM]
        y_train=y[0:MM]
        y_test=y[0:MM]
        #NN=30
        if row<NN:
            X_top=X[0:row]
            y_top=y[0:row]
        else:
            X_top=X[0:NN]
            y_top=y[0:NN]
        
        delta=(max(X_top[:,0])-min(X_top[:,0]))/len(X_top[:,0])
        #X_tmp=np.array(X_top[:,0]-delta*10)
        X_tmp=np.array(X_top[:,0]-delta*0.1)
        #print(X_tmp.shape)
        X_energy = X_tmp.reshape(X_tmp.shape[0], 1)
        new_X = np.hstack((X_energy, y_top))
        #print('new_X shape:', new_X.shape)
        #predicterMulti(np.array(new_X), X_train, X_test, y_train, y_test, 'results_entire_Multirf.csv')

        y_test_predict = dict()
        for name, estimator in ESTIMATORS.items():
            estimator.fit(X_train, y_train)
            y_test_predict[name] = estimator.predict(new_X)
            print (name, ' score is: ', estimator.score(X_test, y_test))
            data = pd.DataFrame(y_test_predict[name])
            data.to_csv('results_entire_'+name+'.csv', header=None, index=None)

        #predicterMulti(np.array(X_top), X_train, X_test, y_train, y_test, 'results_entire_Multirf.csv')
    except:
        print("Please check sorted_entire_struct.csv!")
        
