# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 14:11:39 2019

@author: Tian
"""

import numpy as np

def sigmoid(x):
    return 1./(1.+np.exp(-x))

def sigmoidPrime(x):
    v = sigmoid(x)
    return v * (1 - v)

def sigmoidDerivative(x, y):
    return y * (1 - y)


def tanh(x):
    return np.tanh(x);

def tanhPrime(x):
    t=tanh(x);
    return 1-t^2;

def tanhDerivative(x, y):
    return 1-y^2;


def relu(x):
    return np.maximum(x, 0);

def reluPrime(x):
    return np.where(x>0, 1, 0)

def reluDerivative(x, y):
    return np.where(y, 1, 0)


def softplus(x):
	return np.log1p(np.exp(x))

def softplusPrime(x):
	return sigmoid(x)

def softplusDerivative(x, y):
	return sigmoid(y)

