# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 02:24:30 2018

@author: Tian
"""
import numpy as np
import pandas


def normalize(v):
    upper=v.max(0)
    lower=v.min(0)
    v=2*(v-lower)/(upper-lower) -1
    return v;


def load_data(fn, do_normalize=False):
    d=pandas.read_csv(fn, header=None)
    (n,m)=d.shape
    y=d[m-1].values
    y.resize(n,1)
    v=d.iloc[:,0:m-1].values
    if do_normalize:
        v=normalize(v)
    X=np.hstack((v, np.ones([n,1])))
    return (X,y)


def sep_data(X,y,n):
    Xs=[]
    ys=[]
    l=len(y)
    rng=np.arange(0,l,n)
    for i in range(n):
        r=rng+i
        Xs.append(X[r,:])
        ys.append(y[r])
    return Xs,ys

'''
return a matrix (w*n), each column is a weight vector
'''
def load_param_from_record(fn):
    d=pandas.read_csv(fn, header=None)
    (n,m)=d.shape
    w=d.iloc[:,2:].values
    return w.transpose()

def load_param(fn):
    d=pandas.read_csv(fn, header=None)
    (n,m)=d.shape
    w=d.values
    return w.transpose()


def dump_param(fn, weight):
    f=open(fn, 'w')
    s=','.join([str(v) for v in weight.flatten()])
    f.write(s)
    f.close()
