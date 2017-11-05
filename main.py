###############################################################################
##
## Assignment 3: Neural Networks
##
## @author: Matthew Cline
## @version: 20171104
##
## Description: Neural network program to use on the Flu Data from class and
## the WINE dataset.
##
###############################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def splitData(data, trainingSplit=0.6, validationSplit=0.8):
    training, validation, test = np.split(data, [int(trainingSplit*len(data)), int(validationSplit*len(data))])
    return training, validation, test

def shuffleData(data):
    data = data.reindex(np.random.permutation(data.index))
    data = data.reset_index(drop = True)
    return data

def standardizeFeatures(data, exceptions):
    for i in range(0, data.shape[1]):
        if i in exceptions:
            continue
        data.iloc[:,i] = (data.iloc[:,i] - np.mean(data.iloc[:,i]) / np.std(data.iloc[:,i]))
    return data

def scaleFeatures(data, exceptions):
    for i in range(0, data.shape[1]):
        if i in exceptions:
            continue
        data.iloc[:,i] = (data.iloc[:,i] - np.min(data.iloc[:,i])) / (np.max(data.iloc[:,i]) - np.min(data.iloc[:,i]))
    return data

wineData = pd.read_csv('wine.data')
wineData = shuffleData(wineData)
wineData = standardizeFeatures(wineData, [0])
wineData = scaleFeatures(wineData, [0])
wineTrain, wineVal, wineTest = splitData(wineData)

print("Wine Training Data: \n\n")
print(wineTrain, "\n\n")
print("Wine Validation Data: \n\n")
print(wineVal, "\n\n")
print("Wine Test Data: \n\n")
print(wineTest, "\n\n")
