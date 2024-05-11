# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 13:49:30 2022

@author: Omer Gamliel
"""
from sklearn import datasets 
import numpy as np
dataSet = datasets.load_iris()
data = dataSet.data
targets = dataSet.target
perm = np.random.permutation(len(data))
trainRows = int(0.8*len(data));
trainingSet = data[perm][0:trainRows]
trainingTargets = targets[perm][0:trainRows]
testSet = data[perm][trainRows:]
testTargets =  targets[perm][trainRows:]
print(set(targets))
print(dataSet.target_names)

def max_min_Scalling(trainingSet, testSet):
    data = np.vstack([trainingSet,testSet])
    min_col = np.min(data,axis = 0)
    max_col = np.max(data,axis = 0)
    Ndata = (data - min_col)/(max_col - min_col)
    NtrainingSet = Ndata[:len(trainingSet)]
    NtestSet = Ndata[len(trainingSet):]
    return NtrainingSet,NtestSet

def euclidean_distance(NtrainingSet,vec):
    diff = NtrainingSet - vec
    diff_square = diff**2
    distance_square = np.sum(diff_square, axis = 1)
    distance = np.sqrt(distance_square)
    return distance


def predict(k, distance, trainingTargets):
    sorted_index = np.argsort(distance)
    knn_index = sorted_index[:k]
    knn_target = trainingTargets[knn_index]
    targets,count = np.unique(knn_target,return_counts= True)
    max_index = np.argmax(count)
    prediction = targets[max_index]
    return prediction

def main_knn(k):
    NtrainingSet, NtestSet = max_min_Scalling(trainingSet,testSet)
    result = []
    for vec in NtestSet:
        distance = euclidean_distance(NtrainingSet, vec)
        pred = predict(k,distance,trainingTargets)
        result.append(pred)
    result = np.array(result)
    check = result == testTargets
    accuracy = 100* np.sum(check)/len(result)
       
    
#    print("The Classification Labels for k= ", " is:\n ", )
    print("The Classification Accuracy for k = " ,k, " is: \n",accuracy )
    

main_knn(11)
