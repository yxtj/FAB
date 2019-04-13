# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 22:49:11 2019

@author: yanxi
"""

import sys,os,re
import struct
import numpy as np
import pandas
import matplotlib.pyplot as plt

def loadGradBinaryOne(fn, n, type='f'):
    assert(type in ['f','d','float','double'])
    if type == 'f' or type == 'float':
        t='f'
        s=4
    else:
        t='d'
        s=8
    fin=open(fn, 'rb')
    l=[]
    d=fin.read(n*s)
    while d:
        v=struct.unpack(t*n, d)
        l.append(v)
        d=fin.read(n*s)
    fin.close()
    return l
    

def loadGradBinary(fname, ndp, ndim, type='f'):
    n = ndim*ndp
    l=loadGradBinaryOne(fname, n, type)
    x=np.array(l, 'float32')
    x.resize([len(x), ndp, ndim])
    return x


def loadGradCsv(fname, ndp, ndim):
    l=pandas.read_csv(fname, header=None)
    x=np.array(l, 'float32').ravel()
    nk=x.shape[0]//(ndp*ndim)
    assert(nk*ndp*ndim == x.shape[0])
    x.resize(nk,ndp,ndim)
    return x


def mergeGrad(glist):
    return np.array(v for f in glist for v in f)


def calcPointGrad(grad):
    return np.sum(grad*grad,2)


def loadParameter(fname):
    d=pandas.read_csv(fname, skiprows=0, header=None)
    iteration=np.array(d[0])
    time=np.array(d[1])
    d=d.drop(columns=[0,1]) # drop iteration and time
    return iteration, time, np.array(d)


def loadData(fname,ylist,skiplist=None):
    d=pandas.read_csv(fname, skiprows=0, header=None)
    y=np.array(d[:][ylist])
    skip=ylist
    if skiplist is not None and len(skiplist) != 0:
        for s in skiplist:
            skip.append(s)
    d=d.drop(columns=skip)
    X=np.array(d)
    return X,y


def drawGradDistribution(x, n, cumulative=-1, histtype='bar', logscale=False):
    assert(cumulative in [True, False, 0, 1, -1])
    plt.figure()
    plt.hist(x.ravel(), n, density=True, histtype=histtype, cumulative=cumulative);
    if logscale:
        plt.yscale('log')

#g1 is the data-point gradient of LR
#plt.hist(g1[[1,3,5,7],:].transpose(),100,cumulative=1,density=True,histtype='step');
#plt.legend(['iteration-%d'%i for i in [1,3,5,7]],loc='lower right')

def getTopKIndex(x,k):
    assert(x.ndim==1)
    ind=np.argpartition(x,-k)[-k:]
    return ind
    
def diffScore(x,y):
    sx=set(x)
    sy=set(y)
    sxy=sx.union(sy)
    s=sx.intersection(sy)
    return len(s)/len(sxy)
    
    
def checkDiff(x,y,k):
    ix=getTopKIndex(x.ravel(),k)
    iy=getTopKIndex(y.ravel(),k)
    return diffScore(ix,iy)


def drawTopKDiff(x,ref,ncol=1):
    assert(x.ndim>=2)
    n = x.shape[0]
    v=np.zeros([n,99])
    for i in range(n):
        v[i]=[checkDiff(x[ref],x[i],k) for k in range(1,100)]
    plt.figure()
    plt.plot(v.transpose());
    plt.legend(list(range(n)),ncol=ncol)
    plt.xlabel('top-percentage')
    plt.ylabel('Jacobi Score')
    
    
    
    
    
