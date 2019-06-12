# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#import os
import numpy as np
import matplotlib.pyplot as plt
import myio
import util

#os.chdir('E:/Code/FSB/score/')
#os.chdir('E:/Code/FSB/score/lr/10-100k/1000-0.1')
#os.chdir('E:/Code/FSB/score/mlp/10,15,1-100k/1000-0.1')

def drawOne(fn, n=None, ver=0, xlbl=None, ylbl=None):
    x, y, xr, yr = myio.loadScore(fn, n, ver)
    lgd = fn.replace('.txt','')
    plt.plot(x, y)
    xlbl=xlbl if xlbl is not None else xr
    ylbl=ylbl if ylbl is not None else yr  
    plt.xlabel(xlbl)
    plt.ylabel(ylbl)
    plt.legend(lgd)
    plt.show()

#drawOne('sync', 1)
#drawOne('fab', 1)

def plotUnit(legendList, fn, name, lineMarker, color, n, idx1, idx2):
    x, y, xr, yr = myio.loadScore(fn, n, idx1=idx1, idx2=idx2)
    line = plt.plot(x, y, lineMarker, color=color)
    if(legendList is not None):
        legendList.append(name)
    return line[0].get_color(), (xr, yr)

def drawList(prefix, mList, n=None, ver=1, xlbl=None, ylbl=None):
    plt.figure();
    idx1,idx2=myio.getIdxByVer(ver)
    for m in mList:
        _, xyr = plotUnit(None, prefix+m+'.txt', m, '-', None, n, idx1, idx2)
    #plt.hold(True)
    plt.legend(myio.renameLegend(mList))
    xr,yr = xyr
    xlbl=xlbl if xlbl is not None else xr
    ylbl=ylbl if ylbl is not None else yr  
    plt.xlabel(xlbl)
    plt.ylabel(ylbl)
    plt.tight_layout()
    plt.show()


def plotScoreUnit(legendList, d, name, lineMarker, color, n, idx1, idx2):
    line = plt.plot(d[:n][idx1], d[:n][idx2], lineMarker, color=color)
    if(legendList is not None):
        legendList.append(name)
    return line[0].get_color(), d.shape[1]

def drawScoreList(dataList, nameList, n=None, ver=1, xlbl=None, ylbl=None):
    plt.figure();
    idx1,idx2=myio.getIdxByVer(ver)
    for i in range(len(dataList)):
        d = dataList[i]
        m = nameList[i]
        _, nc = plotScoreUnit(None, d, m, '-', None, n, idx1, idx2)
    #plt.hold(True)
    plt.legend(myio.renameLegend(nameList))
    xr,yr = myio.getxyLabel(idx1, idx2, nc)
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

def drawListCmp(prefix, mList1, mList2, mList3=None, n=None, ncol=1, ver=1, xlbl=None, ylbl=None, save=False):
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
    idx1,idx2=myio.getIdxByVer(ver)
    for i in range(llen):
        c=None
        if i < len(mList1) and mList1[i]:
            c,nc=plotUnit(lgd, prefix+mList1[i]+'.txt', mList1[i], '-', c, n, idx1, idx2)
        if mList2 and i < len(mList2) and mList2[i]:
            c,nc=plotUnit(lgd, prefix+mList2[i]+'.txt', mList2[i], '--', c, n, idx1, idx2)
        if mList3 and i < len(mList3) and mList3[i]:
            c,nc=plotUnit(lgd, prefix+mList3[i]+'.txt', mList3[i], '-.', c, n, idx1, idx2)
    #plt.hold(True)
    plt.legend(myio.renameLegend(lgd), ncol=ncol)
    xr,yr = myio.getxyLabel(idx1, idx2, nc)
    xlbl=xlbl if xlbl is not None else xr
    ylbl=ylbl if ylbl is not None else yr
    plt.xlabel(xlbl)
    plt.ylabel(ylbl)
    if save and isinstance(save, str):
        plt.savefig(save)
    plt.tight_layout()
    plt.show()


def drawScale(prefix, l_nw, nameList, value, speedup=False, ref=False, fit=False, est=False, refIdx=0, ver=1):
    points=np.array([util.whenReachValue(prefix+fn, value, est, ver) for fn in nameList])
    x=np.array(l_nw)
    p=np.argmax(points!=np.nan) # first points[i] != np.nan
    if p!=0 and points[0] == np.nan:
        points=points[p:]
        x=x[p:]
        refIdx-=p
    plt.figure()
    plt.xlabel('number of workers')
    if speedup:
        su=points[refIdx]/points*x[refIdx]
        plt.plot(x, su, '*-', label='actual')
        if ref:
            plt.plot(x, x, '-', label='optimal')
        if fit:
            z=np.polyfit(x, su, 1)
            fun=np.poly1d(z)
            plt.plot(x, fun(x), '--', label='fit')
        plt.ylabel('speed-up')
    else:
        plt.plot(x, points, '*-', label='actual')
        if ref:
            plt.plot(x, points[refIdx]*x[refIdx]/x, '-', label='optimal')
        if fit:
            z=np.polyfit(x, 1/points, 1)
            fun=np.poly1d(z)
            plt.plot(x, 1/fun(l_nw), '--', label='fit')
        plt.ylabel('time (s)')
    plt.ylim([0,None])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    return points

#drawListCmp('../10000-0.1/',['async-1','async-2','async-4'],['fab-1','fab-2','fab-4'])
#drawListCmp('10,15,1-100k/1000-0.1/',['async-1','async-2','async-4', 'async-8'],['fab-1','fab-2','fab-4','fab-8'])
#bsl=[100,250,500,750,1000]
#drawListCmp('',genFLpre(bsl_all,'-0.1/tap-4'), genFLpre(bsl_all,'-0.1/aap-4'),None,4)

