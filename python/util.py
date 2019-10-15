# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 09:56:19 2019

@author: yanxi
"""

import numpy as np
import myio
from scipy.interpolate import interp1d

####
# reload module for loading new modifications
import sys, os
sys.path.insert(0, os.getcwd())
import imp
imp.reload(myio)
####

def genFL(pre, l, post=''):
    return [str(pre)+str(i)+post for i in l]


def whenReachValue(fn, value, est=True, ver=1):
    x, v, _, _ = myio.loadScore(fn, None)
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


def smooth(x, y, n):
    if len(x) >= n:
        return (x, y)
    #f=interp1d(x,y,'cubic')
    f=interp1d(x,y,'quadratic')
    #f=interp1d(x,y,'slinear')
    n = max(n, len(x)*2)
    x2=np.arange(x[0], x[-1], (x[-1] - x[0])/n)
    y2=f(x2)
    return (x2, y2)

'''find the index of the first element in <line> which is <less> than <value>'''
def findIdxOfValue(line, value, less=True):
    if less:
        p=np.argmax(line<=value)
    else:
        p=np.argmax(line>=value)
    return p

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
