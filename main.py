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

def sigmoid(value):
    return 1 / (1 + np.exp(-value))

def sigmoidPrime(value):
    ''' Derivative of the sigmoid function '''
    return np.exp(-value)/((1+np.exp(-value)**2))

class NeuralNet(object):
    def __init__(self, topology):
        tempWeights = []
        self.topology = topology
        for i in range(1, len(topology)):
            tempWeights.append(np.random.random_sample((topology[i-1], topology[i])))
        self.weights = np.array(tempWeights)

    def forwardProp(self, data):
        curLayer = 1
        self.a = []
        self.a.append(data)
        for i in range(len(self.topology)-1):
            self.a.append(sigmoid(np.dot(self.a[i], self.weights[i])))
        return self.a[-1]

    def calculateGradient(self, data, labels):
        ''' calculate the gradient of the current weights '''
        self.predictions = self.forwardProp(data)



wineData = pd.read_csv('wine.data')
wineData = shuffleData(wineData)
wineData = standardizeFeatures(wineData, [0])
wineData = scaleFeatures(wineData, [0])
wineTrain, wineVal, wineTest = splitData(wineData)

####### DEBUG: PRINT DATA PREPROCESSING RESULTS #######
# print("Wine Training Data: \n\n")
# print(wineTrain, "\n\n")
# print("Wine Validation Data: \n\n")
# print(wineVal, "\n\n")
# print("Wine Test Data: \n\n")
# print(wineTest, "\n\n")

dummyData = [[3,4], [1,5], [1,1]]
dummyLabels = [10, 6, 3]
dummyNet = NeuralNet([2,3,1])
predictions = dummyNet.forwardProp(dummyData)
print("Dummy Data Predictions: ", predictions)

wineNet = NeuralNet([13, 5, 3, 1])
wineClass = wineNet.forwardProp(wineTrain.iloc[:,1:])
print("Wine Data Predicitons: ", wineClass)

