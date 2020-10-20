# -*- coding: utf-8 -*-
"""
Spyder Editor
"""

#import os
import numpy as np
import matplotlib.pyplot as plt
import myio
import util

from itertools import cycle
from util import genFL

#os.chdir('E:/Code/FSB/score/')
#os.chdir('E:/Code/FSB/score/lr/10-100k/1000-0.1')
#os.chdir('E:/Code/FSB/score/mlp/10,15,1-100k/1000-0.1')

def set_small_figure(fontsize=12):
    plt.rcParams["figure.figsize"] = [4,3]
    plt.rcParams["font.size"] = fontsize


def set_large_figure(fontsize=16):
    plt.rcParams["figure.figsize"] = [6,4.5]
    plt.rcParams["font.size"] = fontsize


__lineStyles__ = ["-","--","-.",":"]

def drawOne(fn, n=None, ver=0, xlbl=None, ylbl=None, ls='-', smooth=False, smNum=100):
    x, y, xr, yr = myio.loadScore(fn, n, ver)
    lgd = fn.replace('.txt','')
    if smooth:
        x, y = util.smooth(x, y, smNum)
    plt.plot(x, y, ls)
    xlbl=xlbl if xlbl is not None else xr
    ylbl=ylbl if ylbl is not None else yr
    plt.xlabel(xlbl)
    plt.ylabel(ylbl)
    plt.legend(lgd)
    plt.show()

#drawOne('sync', 1)
#drawOne('fab', 1)

def plotUnit(legendList, fn, name, lineMarker, color, n, idx1, idx2,
             smooth=False, smNum=100):
    x, y, xr, yr = myio.loadScore(fn, n, idx1=idx1, idx2=idx2)
    if smooth:
        x, y = util.smooth(x, y, smNum)
    line = plt.plot(x, y, lineMarker, color=color)
    if(legendList is not None):
        legendList.append(name)
    return line[0].get_color(), (xr, yr)

def drawList(prefix, mList, n=None, ver=1, xlbl=None, ylbl=None,
             useLS=False, smooth=False, smNum=100):
    plt.figure();
    idx1,idx2=myio.getIdxByVer(ver)
    if useLS:
        lsc = cycle(__lineStyles__)
    next_ls = lambda : (next(lsc) if useLS else '-' )
    for m in mList:
        _, xyr = plotUnit(None, prefix+m+'.txt', m, next_ls(), None, n, idx1, idx2, smooth, smNum)
    #plt.hold(True)
    plt.legend(myio.renameLegend(mList))
    xr,yr = xyr
    xlbl=xlbl if xlbl is not None else xr
    ylbl=ylbl if ylbl is not None else yr  
    plt.xlabel(xlbl)
    plt.ylabel(ylbl)
    plt.tight_layout()
    plt.show()


def plotScoreUnit(legendList, d, name, lineMarker, color, n, idx1, idx2,
                  smooth, smNum):
    x, y = d[:n][idx1], d[:n][idx2]
    if smooth:
        x, y = util.smooth(x, y, smNum)
    line = plt.plot(x, y, lineMarker, color=color)
    if(legendList is not None):
        legendList.append(name)
    return line[0].get_color(), d.shape[1]

def drawScoreList(dataList, nameList, n=None, ver=1, xlbl=None, ylbl=None,
                  useLS=False, smooth=False, smNum=100):
    plt.figure();
    idx1,idx2=myio.getIdxByVer(ver)
    if useLS:
        lsc = cycle(__lineStyles__)
    next_ls = lambda : (next(lsc) if useLS else '-' )
    for i in range(len(dataList)):
        d = dataList[i]
        m = nameList[i]
        _, nc = plotScoreUnit(None, d, m, next_ls(), None, n, idx1, idx2, smooth, smNum)
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

def drawListCmp(prefix, mList1, mList2, mList3=None, n=None, ncol=1, ver=1,
                xlbl=None, ylbl=None, save=False, smooth=False, smNum=100):
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
            c,xyr=plotUnit(lgd, prefix+mList1[i]+'.txt', mList1[i], '-', c, n,
                           idx1, idx2, smooth, smNum)
        if mList2 and i < len(mList2) and mList2[i]:
            c,xyr=plotUnit(lgd, prefix+mList2[i]+'.txt', mList2[i], '--', c, n,
                           idx1, idx2, smooth, smNum)
        if mList3 and i < len(mList3) and mList3[i]:
            c,xyr=plotUnit(lgd, prefix+mList3[i]+'.txt', mList3[i], '-.', c, n,
                           idx1, idx2, smooth, smNum)
    #plt.hold(True)
    plt.legend(myio.renameLegend(lgd), ncol=ncol)
    xr,yr = xyr
    xlbl=xlbl if xlbl is not None else xr
    ylbl=ylbl if ylbl is not None else yr
    plt.xlabel(xlbl)
    plt.ylabel(ylbl)
    if save and isinstance(save, str):
        plt.savefig(save)
    plt.tight_layout()
    plt.show()


# bar chart
def drawConvergeTime(prefix, fLists, value, refFiles=None, nameList=None, groupList=None,
                     est=False, width=0.8, ncol=1, ver=1, xlbl=None):
    assert isinstance(fLists, list) or isinstance(fLists, np.array)
    assert refFiles == None or isinstance(refFiles, list) or isinstance(refFiles, str)
    if not isinstance(fLists[0], list):
        fLists=[fLists]
    # fLists is [ [a1, b1, c1], [a2, b2, c2] ]
    data=[]
    for fl in fLists:
        points=[util.whenReachValue(prefix+fn, value, est, ver) for fn in fl]
        data.append(points)
    data=np.array(data).transpose()
    # group into [ [a1, a2], [b1, b2], [c1, c2] ]
    # num-group, num-bar
    nb, ng = data.shape
    if refFiles:
        if isinstance(refFiles, str):
            refV=util.whenReachValue(prefix+refFiles, value, est, ver)
            data/=refV
        elif isinstance(refFiles, list):
            refVs=[util.whenReachValue(prefix+rf, value, est, ver) for rf in refFiles]
            refVs=np.array(refVs).reshape([nb, 1])
            data=data/refVs
    barWidth = width/nb
    x = np.arange(ng)
    off = -width/2 + barWidth/2
    #print(ng,nb,x,off,barWidth)
    #print(data)
    plt.figure()
    for i in range(nb):
        y=data[i,:]
        #print(y)
        plt.bar(x + off + barWidth*i, y, barWidth)
    if nameList:
        plt.legend(nameList, ncol=ncol)
    if groupList:
        plt.xticks(x, groupList)
    if xlbl:
        plt.xlabel(xlbl)
    ylbl = 'speed-up' if refFiles else 'time (s)'
    plt.ylabel(ylbl)
    plt.tight_layout()
    return data


def drawScale(prefix, l_nw, nameList, value, speedup=False, ref=False,
              fit=False, est=False, refIdx=0, ver=1):
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
        plt.plot(x, su, '*-', label='experiment')
        if ref:
            plt.plot(x, x, '-', label='optimal')
        if fit:
            z=np.polyfit(x, su, 1)
            fun=np.poly1d(z)
            plt.plot(x, fun(x), '--', label='exp.-ref.')
        plt.ylabel('speed-up')
    else:
        plt.plot(x, points, '*-', label='experiment')
        if ref:
            plt.plot(x, points[refIdx]*x[refIdx]/x, '-', label='optimal')
        if fit:
            z=np.polyfit(x, 1/points, 1)
            fun=np.poly1d(z)
            plt.plot(x, 1/fun(l_nw), '--', label='exp.-ref.')
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

