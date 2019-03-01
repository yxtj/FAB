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

os.chdir('E:/Code/FSB/score/rnn')
#os.chdir('E:/Code/FSB/score/lr/10-100k/1000-0.1')
#os.chdir('E:/Code/FSB/score/mlp/10,15,1-100k/1000-0.1')

USE_HEADER=True;

def drawOne(mode, nw, n=None):
    d = pandas.read_csv(mode+'-'+str(nw)+'.txt',header=None);
    plt.plot(d[:n][0], d[:n][1])
    plt.legend(mode+'-'+str(nw))
    plt.show()

#drawOne('sync', 1)
#drawOne('fab', 1)

def drawGroup(mode, rng, n):
    #plt.hold(True)
    for i in rng:
        d = pandas.read_csv(mode+'-'+str(i)+'.txt',skiprows=0, header=None);
        plt.plot(d[:][0], d[:][1])
    plt.legend([str(i) for i in rng])
    plt.title(mode)
    plt.show()
    #plt.hold(False)

#rng=list(range(1,7))
rng=[1, 2, 4, 8]

#drawGroup('sync', rng)
#drawGroup('fsb', rng)
#drawGroup('async', rng)
#drawGroup('fab', rng)

def drawCmp(mode1, mode2, nw, n=200):
    plt.figure();
    d1=pandas.read_csv(mode1+'-'+str(nw)+'.txt',skiprows=0, header=None);
    d2=pandas.read_csv(mode2+'-'+str(nw)+'.txt',skiprows=0, header=None);
    #plt.hold(True)
    plt.plot(d1[:n][0], d1[:n][1])
    plt.plot(d2[:n][0], d2[:n][1])
    plt.legend([mode1, mode2])
    plt.show()

#drawCmp('async','fsb',8)
#drawCmp('sync','fab',8,300)


def plotUnit(legendList, fn, name, lineMarker, color, n):
    d=pandas.read_csv(fn, skiprows=0, header=None)
    line = plt.plot(d[:n][0], d[:n][1], lineMarker, color=color)
    if(legendList is not None):
        legendList.append(name)
    return line[0].get_color()

def renameLegend(lgd):
    for i in range(len(lgd)):
        s=lgd[i]
        s=s.replace('async','tap').replace('sync','bsp')
        s=s.replace('fsb','fsp').replace('fab','fap')
        lgd[i]=s
    return lgd

def drawList(prefix, mList, n=200):
    plt.figure();
    for m in mList:
        plotUnit(None, prefix+m+'.txt', m, '-', None, n)
    #plt.hold(True)
    plt.legend(renameLegend(mList))
    plt.show()

#drawList('../10000-0.01/',['fab-1','fab-2','fab-4','fab-8'],10)
#drawList('10000-0.1/',['sync-4','fsb-4','async-4','fab-4'])

def genFLpre(pre, l):
    return [pre+str(i) for i in l]

def genFLpost(l, post):
    return [str(i)+post for i in l]

def drawListCmp(prefix, mList1, mList2, mList3=None, n=200, save=False):
    assert(len(mList1) == len(mList2))
    assert(mList3 is None or len(mList3) == 0 or len(mList3) == len(mList1))
    if mList3 is None or len(mList3) == 0:
        mList3=None
    plt.figure()
    l=len(mList1)
    lgd=[]
    for i in range(l):
        c=plotUnit(lgd, prefix+mList1[i]+'.txt', mList1[i], '-', None, n)
        c=plotUnit(lgd, prefix+mList2[i]+'.txt', mList2[i], '--', c, n)
        if(mList3):
            plotUnit(lgd, prefix+mList3[i]+'.txt', mList3[i], '-.', c, n)
    #plt.hold(True)
    plt.legend(renameLegend(lgd))
    plt.xlabel('time (s)')
    plt.ylabel('loss')
    if save:
        gfn=re.sub('^../','',prefix)
        gfn=re.sub('/$','',gfn)
        gfn=gfn.replace('-100k/','/').replace('/','-')
        plt.savefig(gfn+'.png')
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

    