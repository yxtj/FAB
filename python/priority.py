# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 22:49:11 2019

@author: yanxi
"""

import sys,os,re
import numpy as np
import matplotlib.pyplot as plt
import myio

def calcPrioritySquare(grad):
    return np.sum(grad*grad,2)

def calcPriorityProjection(grad):
    m = grad.mean(1)
    res = np.zeros(grad.shape[0:2])
    for i in range(grad.shape[0]):
        res[i] = np.dot(grad[i], m[i])
    return res

def isIncreasing(l):
    return all(x <= y for x, y in zip(l, l[1:]))
def isDecreasing(l):
    return all(x >= y for x, y in zip(l, l[1:]))

def isIncreasingScale(l, f):
    return all(x*(1-f) <= y for x, y in zip(l, l[1:]))
def isDecreasingScale(l, f):
    return all(x*(1-f) >= y for x, y in zip(l, l[1:]))

def findNonlinearIdx(priority, factor):
    n,m=priority.shape
    res=[]
    for i in range(m):
        l=priority[:,i]
        if l[0] > 0:
            if isDecreasingScale(l, factor):
                res.append(i)
        else:
            if isIncreasingScale(l, factor):
                res.append(i)
    return res


#p10=myio.loadPriority('../../grad/lr-1000-10k-10000-0.01.priority',10000)
#l2=findNonlinearIdx(p10,0.2)
#plt.plot(p10[:,l2])
#plt.plot(p10[:,np.random.randint(0,10000,10)])
#plt.grid(True)
#plt.xlabel('epoch')
#plt.ylabel('priority')
#plt.tight_layout()

# distribution show

def drawCdfOne(x, bins, noTrailingZero=None):
    counts, bin_edges=np.histogram(x, bins)
    y = np.cumsum(counts)
    plt.plot(bin_edges, np.insert(y/y[-1], 0, 0.0))

def drawPdfOne(x, bins, noTrailingZero=False):
    counts, bin_edges=np.histogram(x, bins)
    idx = findLastNonZero(counts)
    s = np.sum(counts)
    plt.plot(bin_edges[:idx+2], np.insert(counts/s, 0, 0.0)[:idx+2])

def findLastNonZero(l):
    for i in range(len(l)-1,0,-1):
        if l[i]!=0:
            return i
    return 0

# hist on the first dim
def drawDistribution(data, nbins, noTrailingZero=False, cumulative=True, lgd=None):
    assert(data.ndim == 2)
    n, m = data.shape
    assert(n>m)
    funone = drawCdfOne if cumulative else drawPdfOne
    low = data.min()
    high = data.max()
    bins = np.linspace(low, high, nbins+1)
    plt.figure()
    for i in range(m):
        x = data[:,i]
        funone(x, bins, noTrailingZero)
    plt.grid(True)
    if lgd is not None:
        plt.legend(lgd)
    plt.show()

def drawContribution(data, nbins, cumulative=True, lgd=None):
    assert(data.ndim == 2)
    n, m = data.shape # n->point, m->iter
    assert(n>m)
    x = np.arange(0, nbins+1)/nbins*100
    bins = np.linspace(0, n-1, nbins+1, dtype=int)
    s = np.cumsum(np.sort(data, 0), 0)[bins, :]
    s[0,:] = 0
    plt.figure()
    if cumulative:
        y = s / s[-1,:]
    else:
        d = np.diff(s, axis=0)
        y = d / s[-1,:]
        y = np.insert(y, 0, 0.0, axis=0)
    plt.plot(x, y*100)
    plt.grid(True)
    if lgd is not None:
        plt.legend(lgd)
    plt.xlabel('percentile (%)')
    plt.ylabel('contribution (%)')
    plt.tight_layout()
    plt.show()


def drawContributionOfTop(data, topPoints, ncol=1):
    assert(data.ndim == 2)
    n, m = data.shape # n->point, m->iter
    assert(n>m)
    bins=np.array((1-np.array(topPoints))*n, dtype=int)
    data=np.sort(data, 0)
    s=data.sum(0)
    plt.figure()
    for b in bins:
        y=data[b:,:].sum(0)/s
        plt.plot(y*100)
    plt.grid(True)
    plt.legend(['top-'+str(t*100)+'%' for t in topPoints], ncol=ncol)
    plt.xlabel('iteration')
    plt.ylabel('contribution (%)')
    plt.tight_layout()
    plt.show()


def drawDeltaLoss(data, nbins, lgd=None):
    assert(data.ndim == 2)
    n, m = data.shape
    assert(n>m)
    x = np.arange(1, nbins+1)/nbins*100
    binId = np.linspace(0, n-1, nbins+1, dtype=int)[1:]
    s = np.flip(np.sort(data, 0), 0)
    sc = np.cumsum(s, 0)[binId, :]
    y = np.zeros_like(sc)
    for i in range(m):
       y[:,i] = sc[:,i]/(binId+1)
    plt.figure()
    plt.plot(x, y)
    plt.grid(True)
    if lgd is not None:
        plt.legend(lgd)
    plt.xlabel('top percentile (%)')
    plt.ylabel('delta loss')
    plt.tight_layout()
    plt.show()

def drawPriorityDiff(data, step=1, stride=1, ratio=True, avg=False, lgd=None):
    assert(data.ndim == 2)
    n, m = data.shape
    assert(n>m)
    plt.figure()
    plt.xlabel(('epoch (x%d)'%stride) if stride != 1 else 'epoch')
    if ratio:
        pratio = data[range(0,n-step,stride),:]/data[range(step,n,stride),:]
        pd = pratio
        plt.ylabel('ratio')
    else:
        pdiff = data[range(0,n-step,stride),:]-data[range(step,n,stride),:]
        pd = pdiff
        plt.ylabel('difference')
    if avg:
        pd /= step
    plt.plot(pd)
    if lgd is not None:
        plt.legend(lgd)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


#p10=myio.loadPriority('E:/Code/FSB/grad/lr-1000-10k-10000-0.01.priority',10000)
#sp10=myio.loadPriority('E:/Code/FSB/grad/lr-1000-10k-10000-0.01.square.priority',10000)
#p101=myio.loadPriority('E:/Code/FSB/grad/lr-1000-10k-10000-0.01-1.priority',10000)
#p10p=myio.loadPriority('E:/Code/FSB/grad/lr-1000-10k-p0.05-r0.01-ld-0.01-1.priority',10000)
#pm=myio.loadPriority('E:/Code/FSB/grad/mlp-mnist300-bsp-4-b600.priority',60000)
#r=range(0,250,50);drawDistribution(p10[r,:].T,100,True,False,['epoch-%d'%v for v in r])
#plt.ylabel('probability');plt.xlabel('priority');plt.grid(True);plt.tight_layout()
#plt.ylabel('density');plt.xlabel('priority');plt.grid(True);plt.tight_layout()
#plt.xlabel('gradient length');
#plt.xlabel('gradient projection');
#plt.tight_layout()
#r=range(0,250,50);drawContribution(p10[r,:].T,100,False,['epoch-%d'%v for v in r])

#drawContributionOfTop(pm.T, np.linspace(0.1,0.9,9))
#plt.legend(['top-'+str(v)+'%' for v in range(1,10)])
#drawContributionOfTop(ps.T, [0.001, 0.01, 0.05, 0.1])

#r=np.random.randint(0,10000,10)
#r=[9759, 2663, 7733, 1341, 5248,  865, 2810, 3152, 6930,  131]
#drawPriorityDiff(p10[:,r])
#drawPriorityDiff(p10[:,r],relative=True,avg=True)


def drawGradDistribution(x, n, cumulative=-1, histtype='bar', logscale=False):
    assert(cumulative in [True, False, 0, 1, -1])
    assert(histtype in ['bar', 'barstacked', 'step', 'stepfilled'])
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


def drawDecayTrend(data, diff=False):
    assert(data.ndim==2)
    n,m=data.shape # n->iter, m->points
    assert(n>m)
    y=np.log(data)
    plt.figure()
    ylbl='log-priority'
    if diff:
        y=y[1:,:]-y[:-1,:]
        ylbl+='-difference'
    plt.plot(y)
    plt.grid(True)
    plt.xlabel('iteration')
    plt.ylabel(ylbl)
    plt.tight_layout()
    
    
#plr=myio.loadPriority('lr-1000-10k-10000-0.01-1.priority',10000)
#plr2=myio.loadPriority('lr-1000-10k-p0.05-r0.01-ld-0.01-1.priority',10000)
#pmlp=myio.loadPriority('mlp-mnist300-600-0.001.priority',60000)
#pmlp2=myio.loadPriority('mlp-mnist300-bsp-4-p0.01-r0-ld.priority',60000)
#pmlp3=myio.loadPriority('mlp-mnist300-tap-4-p0.01-r0-ld.priority',60000)
#pcnn=myio.loadPriority('cnn-mnist1020-600.priority',60000)
#pcnn2=myio.loadPriority('cnn-mnist1020-tap-4-p0.001-r0-ld.priority',60000)

#r=np.random.randint(0,60000,10)
#r=np.array([ 9372,  3584, 40396, 34916,  3171, 13232, 31886,  3578, 25493, 29979])
#drawDecayTrend(pmlp2[:,r],False)
#drawDecayTrend(pmlp2[:,r],True)
