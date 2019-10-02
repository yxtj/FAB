# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 23:43:40 2019

@author: yanxi
"""

import pandas
import numpy as np
import struct

# -------- score --------

def getIdxByVer(ver):
    if ver == 0:
        idx1,idx2=0,1
    elif ver == 1:
        idx1,idx2=1,2
    elif ver == 2:
        idx1,idx2=0,2
    elif ver == 3:
        idx1,idx2=1,3
    elif ver == 4:
        idx1,idx2=0,3
    elif ver == 5:
        idx1,idx2=0,4
    elif ver == 6:
        idx1,idx2=1,4
    else:
        return ver[0], ver[1]
    return idx1, idx2


__HEADER4__=['time(s)', 'loss', 'difference', 'delta']
__HEADER5__=['time(s)', 'loss', 'accuracy', 'difference', 'delta']
__HEADER6__=['iteration', 'time(s)', 'loss', 'accuracy', 'difference', 'delta']
def getxyLabel(idx1, idx2, ncol=6):
    if ncol == 4 and idx1 < 4 and idx2 < 4:
        return __HEADER4__[idx1], __HEADER4__[idx2]
    elif ncol == 5 and idx1 < 5 and idx2 < 5:
        return __HEADER5__[idx1], __HEADER5__[idx2]
    elif ncol == 6 and idx1 < 6 and idx2 < 6:
        return __HEADER6__[idx1], __HEADER6__[idx2]
    return None, None


'''load score file'''
def loadScore(fn, n=None, ver=1, **kwargs):
    if fn.endswith('.txt'):
        d = pandas.read_csv(fn,header=None)
    else:
        d = pandas.read_csv(fn+'.txt',header=None)
    if 'idx1' in kwargs and 'idx2' in kwargs:
        idx1=kwargs['idx1']
        idx2=kwargs['idx2']
    elif 'idx' in kwargs and hasattr(kwargs['idx'],'__iter__') and len(kwargs['idx'])==2:
        idx1,idx2=kwargs['idx']
    else:
        idx1,idx2=getIdxByVer(ver)
    xr,yr = getxyLabel(idx1, idx2, d.shape[1])
    return np.array(d[:n][idx1]), np.array(d[:n][idx2]), xr, yr


def renameLegend(lgd):
    for i in range(len(lgd)):
        s=lgd[i]
        s=s.replace('async','tap').replace('sync','bsp')
        s=s.replace('fsb','fsp').replace('fab','aap')
        lgd[i]=s
    return lgd

# -------- parameter --------

''' load parameter file'''
def loadParameterCsv(fname):
    d=pandas.read_csv(fname, skiprows=0, header=None)
    iteration=np.array(d[0])
    time=np.array(d[1])
    d=d.drop(columns=[0,1]) # drop iteration and time
    return iteration, time, np.array(d)

def loadParameterBinary(fname, ndim):
    n=4+8+8*ndim
    iteration = []
    time = []
    param = []
    with open(fname, 'rb') as fin:
        d=fin.read(n)
        while d:
            iteration.append(struct.unpack('i', d[0:4]))
            time.append(struct.unpack('d',d[4:12]))
            v=struct.unpack('d'*ndim, d[12:])
            param.append(v)
            d=fin.read(n)
    return np.array(iteration), np.array(time), np.array(d)


'''load data file'''
def loadData(fname, ylist, skiplist=None):
    d=pandas.read_csv(fname, skiprows=0, header=None)
    y=np.array(d[:][ylist])
    skip=ylist
    if skiplist is not None and len(skiplist) != 0:
        for s in skiplist:
            skip.append(s)
    d=d.drop(columns=skip)
    X=np.array(d)
    return X,y

# -------- gradient --------

def type2size(type):
    if type == 'f' or type == 'float':
        t='f'
        s=4
    else:
        t='d'
        s=8
    return (t,s)

'''load gradient files'''
def loadGradBinary(fname, ndp, ndim, type='f'):
    n = ndim*ndp
    t,s = type2size(type)
    x=[]
    with open(fname, 'rb') as fin:
        d=fin.read(n*s)
        while d:
            v=struct.unpack(t*n, d)
            x.append(v)
            d=fin.read(n*s)
    x=np.array(x, 'float32')
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


# -------- priority --------

'''load priority file'''   
def loadPriority(fn, ndp, type='f'):
    x=[]
    t,s=type2size(type)
    with open(fn, 'rb') as fin:
        d=fin.read(ndp*s)
        while d:
            v=struct.unpack(t*ndp, d)
            x.append(v)
            d=fin.read(ndp*s)
    x=np.array(x, 'float32')
    x.resize([len(x), ndp])
    return x


def dumpPriority(fn, priority):
    with open(fn, 'wb') as f:
        np.save(f, priority)


