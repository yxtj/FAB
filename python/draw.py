# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import pandas
import numpy as np
import matplotlib.pyplot as plt
import re

os.chdir('E:/Code/FSB/score/')
#os.chdir('E:/Code/FSB/score/lr/10-100k/1000-0.1')
#os.chdir('E:/Code/FSB/score/mlp/10,15,1-100k/1000-0.1')

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

__HEADER4__=['time', 'loss', 'difference', 'delta']
__HEADER5__=['time', 'loss', 'accuracy', 'difference', 'delta']
__HEADER6__=['iteration', 'time', 'loss', 'accuracy', 'difference', 'delta']
def getxyLabel(idx1, idx2, ncol):
    if ncol == 4 and idx1 < 4 and idx2 < 4:
        return __HEADER4__[idx1], __HEADER4__[idx2]
    elif ncol == 5 and idx1 < 5 and idx2 < 5:
        return __HEADER5__[idx1], __HEADER5__[idx2]
    elif ncol == 6 and idx1 < 6 and idx2 < 6:
        return __HEADER6__[idx1], __HEADER6__[idx2]
    return None, None
        
def renameLegend(lgd):
    for i in range(len(lgd)):
        s=lgd[i]
        s=s.replace('async','tap').replace('sync','bsp')
        s=s.replace('fsb','fsp').replace('fab','aap')
        lgd[i]=s
    return lgd


def drawOne(fn, n=None, ver=0, xlbl=None, ylbl=None):
    if fn.endswith('.txt'):
        d = pandas.read_csv(fn,header=None);
        lgd = fn.replace('.txt','')
    else:
        d = pandas.read_csv(fn+'.txt',header=None);
        lgd = fn
    idx1,idx2=getIdxByVer(ver)
    xr,yr = getxyLabel(idx1, idx2, d.shape[1])
    plt.plot(d[:n][idx1], d[:n][idx2])
    xlbl=xlbl if xlbl is not None else xr
    ylbl=ylbl if ylbl is not None else yr  
    plt.xlabel(xlbl)
    plt.ylabel(ylbl)
    plt.legend(lgd)
    plt.show()

#drawOne('sync', 1)
#drawOne('fab', 1)

def plotUnit(legendList, fn, name, lineMarker, color, n, idx1, idx2):
    d=pandas.read_csv(fn, skiprows=0, header=None)
    line = plt.plot(d[:n][idx1], d[:n][idx2], lineMarker, color=color)
    if(legendList is not None):
        legendList.append(name)
    return line[0].get_color(), d.shape[1]

def plotScoreUnit(legendList, d, name, lineMarker, color, n, idx1, idx2):
    line = plt.plot(d[:n][idx1], d[:n][idx2], lineMarker, color=color)
    if(legendList is not None):
        legendList.append(name)
    return line[0].get_color(), d.shape[1]
    
def drawList(prefix, mList, n=None, ver=0, xlbl=None, ylbl=None):
    plt.figure();
    idx1,idx2=getIdxByVer(ver)
    for m in mList:
        _, nc = plotUnit(None, prefix+m+'.txt', m, '-', None, n, idx1, idx2)
    #plt.hold(True)
    plt.legend(renameLegend(mList))
    xr,yr = getxyLabel(idx1, idx2, nc)
    xlbl=xlbl if xlbl is not None else xr
    ylbl=ylbl if ylbl is not None else yr  
    plt.xlabel(xlbl)
    plt.ylabel(ylbl)
    plt.tight_layout()
    plt.show()

def drawScoreList(dataList, nameList, n=None, ver=0, xlbl=None, ylbl=None):
    plt.figure();
    idx1,idx2=getIdxByVer(ver)
    for i in range(len(dataList)):
        d = dataList[i]
        m = nameList[i]
        _, nc = plotScoreUnit(None, d, m, '-', None, n, idx1, idx2)
    #plt.hold(True)
    plt.legend(renameLegend(mList))
    xr,yr = getxyLabel(idx1, idx2, nc)
    xlbl=xlbl if xlbl is not None else xr
    ylbl=ylbl if ylbl is not None else yr  
    plt.xlabel(xlbl)
    plt.ylabel(ylbl)
    plt.tight_layout()
    plt.show()

#drawList('../10000-0.01/',['fab-1','fab-2','fab-4','fab-8'],10)
#drawList('10000-0.1/',['sync-4','fsb-4','async-4','fab-4'])
#ln=[1,2,4,8,12,16,20,24]
#lmode=['bsp','tap','aap']
#lmode_=['bsp-','tap-','aap-']
#l=['%d-0.01/aap-%i' % (10*i*i,i) for i in ln]

def genFLpre(pre, l):
    return [str(pre)+str(i) for i in l]

def genFLpost(l, post):
    return [str(i)+str(post) for i in l]

def genFL(pre, l, post=''):
    return [str(pre)+str(i)+post for i in l]

def drawListCmp(prefix, mList1, mList2, mList3=None, n=None, ncol=1, ver=0, xlbl=None, ylbl=None, save=False):
    llen=len(mList1)
    if mList2 is None or len(mList2) == 0:
        mList2=None
    else:
        llen=max(llen, len(mList2))
    if mList3 is None or len(mList3) == 0:
        mList3=None
    else:
        llen=max(llen, len(mList3))
    plt.figure()
    lgd=[]
    idx1,idx2=getIdxByVer(ver)
    for i in range(llen):
        c=None
        if i < len(mList1) and mList1[i]:
            c,nc=plotUnit(lgd, prefix+mList1[i]+'.txt', mList1[i], '-', c, n, idx1, idx2)
        if mList2 and i < len(mList2) and mList2[i]:
            c,nc=plotUnit(lgd, prefix+mList2[i]+'.txt', mList2[i], '--', c, n, idx1, idx2)
        if mList3 and i < len(mList3) and mList3[i]:
            c,nc=plotUnit(lgd, prefix+mList3[i]+'.txt', mList3[i], '-.', c, n, idx1, idx2)
    #plt.hold(True)
    plt.legend(renameLegend(lgd), ncol=ncol)
    xr,yr = getxyLabel(idx1, idx2, nc)
    xlbl=xlbl if xlbl is not None else xr
    ylbl=ylbl if ylbl is not None else yr
    plt.xlabel(xlbl)
    plt.ylabel(ylbl)
    if save:
        gfn=re.sub('^../','',prefix)
        gfn=re.sub('/$','',gfn)
        gfn=gfn.replace('-100k/','/').replace('/','-')
        plt.savefig(gfn+'.png')
    plt.tight_layout()
    plt.show()


#drawListCmp('../10000-0.1/',['async-1','async-2','async-4'],['fab-1','fab-2','fab-4'])
#drawListCmp('10,15,1-100k/1000-0.1/',['async-1','async-2','async-4', 'async-8'],['fab-1','fab-2','fab-4','fab-8'])
#bsl=[100,250,500,750,1000]
#drawListCmp('',genFLpre(bsl_all,'-0.1/tap-4'), genFLpre(bsl_all,'-0.1/aap-4'),None,4)

# returnn time and score, each file holds a ROW
def getRecord(prefix, mList, asMatrix=False):
    time=[]
    score=[]
    for m in mList:
        t=pandas.read_csv(prefix+m+'.txt',skiprows=0, header=None)
        time.append(t[:][0].values)
        score.append(t[:][1].values)
    if asMatrix == True:
        time=np.array(time)
        score=np.array(score)
    return time,score


def score2progress(score, p0=None, pinf=None):
    if isinstance(score, np.ndarray):
        # 1-D and 2-D np.array
        if p0 is None:
            p0=score.max()
        if pinf is None:
            pinf=score.min()
        return (p0-score)/(p0-pinf)
    else:
        # list of np.array
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

def findTime4score(time, score, spoint):
    if isinstance(score, np.ndarray) and score.ndim == 2:
        # matrix
        t=np.abs(score-spoint)
        idx=t.argmin(1)
    elif isinstance(score, list) and isinstance(score[0], np.ndarray):
        # list of np.array
        idx=[]
        for sl in score:
            t=np.abs(sl-spoint)
            idx.append(t.argmin())
    else: # isinstance(score, numpy.ndarray) && score.ndim == 1:
        # 1-D array
        t=np.abs(score-spoint)
        idx=t.argmin()
        return time[idx]
    res=[]
    for i in range(len(idx)):
        res.append(time[i][idx[i]])
    return np.array(res)


def drawScale(time, score, x, ppoints, refX=None):
    assert len(time) == len(x) and len(score) == len(x)
    if refX==None:
        refIdx=0
    else:
        refIdx=x.index(refX)
    factor = np.array(x).reshape([len(x),1]) / x[refIdx]
    progress=score2progress(score)
    lgd=[]
    plt.figure()
    for pp in ppoints:
        t = findTime4score(time, progress, pp)
        plt.plot(x, t, '-')
        plt.plot(x, t[refIdx]/factor, '--')
        lgd.append(str(pp)+'-act')
        lgd.append(str(pp)+'-opt')
    plt.xlabel('# of workers')
    plt.ylabel('time (s)')
    plt.legend(lgd)
    plt.show()
    
    
#time,score=getRecord('../10000-0.1/',['fab-1','fab-2','fab-4','fab-8'])
#drawScale(time, score, [1,2,4,8], [0.9])


def drawScaleCmpAll(prefix, modes, x, ppoint, showRef=False):
    def getNameList(head, rng):
        return [head+str(i) for i in rng]
    time=[]; score=[];
    for m in modes:
        t,s=getRecord(prefix, getNameList(m+'-', x))
        time.append(t)
        score.append(s)
    
    def getMinMax(scoreList):
        mi = min([sl.min() for sl in scoreList])
        ma = max([sl.max() for sl in scoreList])
        return mi,ma
    s0=0; sinf=np.inf
    for s in score:
        mi,ma=getMinMax(s)
        s0=max(s0,ma)
        sinf=min(sinf,mi)
    
    progress=[]
    for s in score:
        p=score2progress(s, s0, sinf)
        progress.append(p)
    
    plt.figure()
    factor = np.array(x) / x[0]
    lgd=[]
    for i in range(len(modes)):
        t = findTime4score(time[i], progress[i], ppoint)
        plt.plot(x, t)
        lgd.append(modes[i])
        if showRef:
            plt.plot(x, t[0]/factor, 'k--')
            lgd.append(modes[i]+'-ref')
    plt.legend(lgd)
    plt.show()

#drawScaleCmpAll('../1000-0.1/', ['sync','fsb','async','fab'], [1,2,4,8], 0.85)

    