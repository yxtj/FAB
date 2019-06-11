# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 09:56:19 2019

@author: yanxi
"""

import pandas
import numpy as np


def genFL(pre, l, post=''):
    return [str(pre)+str(i)+post for i in l]


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


''' load parameter file'''
def loadParameter(fname, binary=False):
    d=pandas.read_csv(fname, skiprows=0, header=None)
    iteration=np.array(d[0])
    time=np.array(d[1])
    d=d.drop(columns=[0,1]) # drop iteration and time
    return iteration, time, np.array(d)

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


def renameLegend(lgd):
    for i in range(len(lgd)):
        s=lgd[i]
        s=s.replace('async','tap').replace('sync','bsp')
        s=s.replace('fsb','fsp').replace('fab','aap')
        lgd[i]=s
    return lgd


def whenReachValue(fn, value, est=True, ver=1):
    x, v, _, _ = loadData(fn, None)
    # find the first index where v[p]<=value
    p=np.argmax(v<=value)
    if p==0 and v[0]>value: # failed
        if not est:
            return np.nan
        else:
            r1 = (v[-4]-v[-2])/(x[-4]-x[-2])
            r2 = (v[-3]-v[-1])/(x[-3]-x[-1])
            a=min(r2/r1,1)
            r=(r1+r2)/2
            if est != 'exp' or a==1:
                return  x[-1] + (value-v[-1])/r
            else:
                dx=(x[-3]-x[-1] + x[-4]-x[-2])/2
                # value=v[-1] + a*r2*dx + a*a*r2*dx + ...
                # a*r2*dx/(1-a)*(1-a^n)=value-v[-1]
                n=np.log(1-(value-v[-1])/a/r2/dx*(1-a))/np.log(a)
                return x[-1] + dx*n
    else:
        return x[p]


def score2progress(score, p0=None, pinf=None):
    if isinstance(score, np.ndarray):
        # 1-D and 2-D np.array
        if p0 is None:
            p0=score.max()
        if pinf is None:
            pinf=score.min()
        return (p0-score)/(p0-pinf)
    else:
        # list of np.array (length of each np.arry is different)
        n=len(score)
        if p0 is None or pinf is None:
            tp0=0
            tpinf=np.inf
            for i in range(n):
                tp0=max(tp0, score[i].max())
                tpinf=min(tpinf, score[i].min())
            if p0 is None:
                p0=tp0
            if pinf is None:
                pinf=tpinf
        res=[]
        for i in range(n):
            res.append((p0-score[i])/(p0-pinf))
        return res
