# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 11:26:29 2019

@author: yanxi
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from draw import *

plt.rcParams["figure.figsize"] = [4,3]

os.chdir(r'E:\Code\FSB\score\mlp\784,300,10-60k')
os.chdir(r'E:\Code\FSB\score\mlp\100,40,20,1-100k')

namepre='mnist-m300'
namepre='mlp100'

plt.rcParams["figure.figsize"] = [6,4.5]
ln=[2,4,8,12,16,20,24]

lr=0.01
lr=0.001

for m in lmode:
    drawList('1000-%s/' % lr, genFL(m+'-',ln,''))
    saveimg(namepre+'-scale-%s' % m)
    plt.close()

l=[1,2,4,8,12]
drawListCmp('1000-0.01/', genFL('bsp-',l),genFL('tap-',l), genFL('aap-',l))
saveimg(namepre+'-cmp-%s' % ','.join([str(v) for v in l]))

# priority for MLP
bs0=100000
bs0=60000
pre=str(bs0)+'-0.001/bsp-4-'
l=[0.01, 0.05, 0.1, 0.15, 0.2]

drawListCmp('',genFL('',[prio2bs(bs0,v) for v in l],'-0.001/bsp-4'),genFL(pre+'pso',l),genFL(pre+'pgo',l),n=None,ver=1)

k=0.01
lr=[0.001, 0.01, 0.05]
lr=[0, 0.001, 0.01, 0.05]
# square priority vs. global priority
drawListCmp('',[str(prio2bs(bs0,k))+'-0.001/bsp-4'],[None]+genFL(pre+'ps'+str(k)+'-r',lr),[None]+genFL(pre+'pg'+str(k)+'-r',lr),n=None,ver=1)

# decay priority
k=0.01;
d=0.7
drawListCmp('',[str(prio2bs(bs0,k))+'-0.001/bsp-4'],[None]+genFL(pre+'p'+str(k)+'-r',lr,'-d'+str(d)),n=None,ver=1)
plt.legend(['sgd']+genFL('psgd-r',lr))

for i in lvername:
    drawListCmp('',[str(prio2bs(bs0,k))+'-0.001/bsp-4'],[None]+genFL(pre+'p'+str(k)+'-r',lr,'-d'+str(d)),n=None,ver=i)
    plt.legend(['sgd']+genFL('psgd-r',lr))
    plt.savefig('cmp-p'+str(k)+'-d'+str(d)+'-'+lvername[i]+'.pdf')

drawList('',genFL(pre+'p'+str(k)+'-r',lr,'-d'+str(d)),n=None,ver=1)
drawList('',genFL(pre+'p'+str(k)+'-r',lr,'-d'+str(1-k)),n=None,ver=1)
drawListCmp('',[str(prio2bs(bs0,k))+'-0.001/bsp-4'],[None]+genFL(pre+'p'+str(k)+'-r',lr,'-d'+str(1-k)),[None]+genFL(pre+'p'+str(k)+'-r',lr,'-d'+str(d)),n=None,ver=1)
d1=0.7; d2=0.9;
drawListCmp('',[str(prio2bs(bs0,k))+'-0.001/bsp-4'],[None]+genFL(pre+'p'+str(k)+'-r',lr,'-d'+str(d1)),[None]+genFL(pre+'p'+str(k)+'-r',lr,'-d'+str(d2)),n=None,ver=1)

k=0.01;r=0.01
ld=[1-k, 0.9, 0.8, 0.7]
drawListCmp('',[str(prio2bs(bs0,k))+'-0.001/bsp-4'],[None]+genFL(pre+'p'+str(k)+'-r'+str(r)+'-d',ld),n=None,ver=1)
plt.legend(['sgd']+genFL('decay-',ld))

for i in lvername:
    drawListCmp('',[str(prio2bs(bs0,k))+'-0.001/bsp-4'],[None]+genFL(pre+'p'+str(k)+'-r'+str(r)+'-d',ld),n=None,ver=i)
    plt.legend(['sgd']+genFL('decay-',ld))
    plt.savefig('decay-p'+str(k)+'-r'+str(r)+'-'+lvername[i]+'.pdf')

os.chdir(r'E:\Code\FSB\score\mlp\100,40,20,1-100k\krd')
ld=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
ld2=[0.1, 0.3, 0.5, 0.7, 0.9]
ld3=[0.01,0.02,0.03,0.04,0.05,0.1]
lr=[0, 0.05, 0.1, 0.15, 0.2]
lr2=[0, 0.01, 0.05, 0.1, 0.15, 0.2]

drawList('',genFL(pre+'p',ld2,'-r0.1-d0.8'),ver=1)
plt.legend(['top-'+str(v*100)+'%' for v in ld2]);plt.grid(True)
drawList('',genFL(pre+'p',ld3,'-r0.1-d0.8'),ver=1)
plt.legend(['top-'+str(v*100)+'%' for v in ld3]);plt.ylim([0,2]);plt.grid(True)

drawList('',genFL(pre+'p0.01-r',lr,'-d0.8'),ver=1)
plt.legend(['r='+str(v*100)+'%' for v in lr]);plt.grid(True)

drawListCmp(pre,['p0.02-r0.01-d0.8','p0.03-r0.01-d0.8','p0.04-r0.01-d0.8','p0.05-r0.01-d0.8'],['p0.01-r0.02-d0.8','p0.01-r0.03-d0.8','p0.01-r0.04-d0.8','p0.01-r0.05-d0.8'],ver=1)
plt.ylim([0,3])
plt.legend(['k:2%,r:1%','k:1%,r:2%','k:3%,r:1%','k:1%,r:3%','k:4%,r:1%','k:1%,r:4%','k:5%,r:1%','k:1%,r:5%'])

# convergence speed
# speed 6%
drawList('60000-0.001/',['../3600-0.001/bsp-4','bsp-4-p0.05-r0.01-lp','bsp-4-p0.05-r0.01-ld','bsp-4-p0.05-r0.01-ld-vj'],ver=1)
plt.xlim([-10,310])
plt.legend(['SGD','PSGD','PSGD+D','PSGD+D+A'])
plt.grid(True)

drawList('60000-0.001/',['../600-0.001/bsp-4','../1200-0.001/bsp-4']+genFL('bsp-4-p0.01-r0.01-',['lp','ld','ld-vj']),ver=1)
plt.legend(['SGD-1%','SGD-2%','PSGD','PSGD+D','PSGD+D+A'])
plt.grid(True)

drawList('60000-0.001/',['../3000-0.001/bsp-4','../3600-0.001/bsp-4']+genFL('bsp-4-p0.05-r0.01-',['lp','ld','ld-vj']),ver=1)
plt.legend(['SGD-5%','SGD-6%','PSGD','PSGD+D','PSGD+D+A'])
plt.grid(True)



# scale
l_nw=[1,2,3,4,6,8,12,12,16]
drawList('60000-0.001/',genFL('bsp-',l_nw,'-p0.05-r0.01-ld'),ver=1)
plt.legend(l_nw,ncol=2)

# scale - speedup
drawScale('60000-0.001/',l_nw,genFL('bsp-',l_nw,'-p0.05-r0.01-ld'),0.87,True)
plt.xticks(range(0,17,4))
drawScale('60000-0.001/',[2,4,6,8,12,16],genFL('tap-',[2,4,6,8,12,16],'-p0.05-r0.01-ld'),0.43,True,ref=True,fit=True,est=True)
plt.xticks(range(0,17,4))

# scale - time
drawScale('60000-0.001/',l_nw,genFL('bsp-',l_nw,'-p0.05-r0.01-ld'),0.87,False)
plt.xticks(range(0,17,4))
