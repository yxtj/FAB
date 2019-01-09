# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 23:18:15 2018

@author: Tian
"""

import numpy as np

def sigmoid(z):
    return 1./(1.+np.exp(-z))
    
def predict(X, w):
    return sigmoid(np.dot(X,w))

__classify=np.vectorize(lambda pred: 1 if pred>=0.5 else 0)

def classify(pred):
    return __classify(pred)
    
def loss(X, y, w):
    pred=predict(X,w)
    class1_cost = y*np.log(pred)
    class2_cost = (1-y)*np.log(1-pred)
    cost = -class1_cost.sum() - class2_cost.sum()
    return cost/len(y)

def gradient(X, y, w):
    n = X.shape[0] if X.ndim != 1 else X.size
    predictions = predict(X, w)
    gradient = np.dot(X.T, predictions - y)
    return gradient / n
    
def train(X, y, w, lrate, iters=1):
    if y.ndim == 1:
        y.resize([len(y),1])
    for _ in range(iters):
        grad=gradient(X, y, w)
        w -= grad*lrate
    return w, grad

def trainBatch(X, y, w, lrate, pf, pl):
    return train(X[pf:pl,:], y[pf:pl], w, lrate, 1)

    
def accuracy(predicted_labels, actual_labels):
    diff = predicted_labels - actual_labels
    return 1.0 - (float(np.count_nonzero(diff)) / len(diff))


def dis_gradient(Xs,ys,w,lid,idx,cnt):
    return gradient(Xs[lid][idx:idx+cnt,:], ys[lid][idx:idx+cnt], w)

