# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 20:02:49 2019

@author: yanxi
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from draw import *

plt.rcParams["figure.figsize"] = [4,3]
plt.rcParams["font.size"]=12
plt.rcParams["font.size"]=14

ln=[1,2,4,8,12,16,20,24]
ln2=[4,8,12,16,20,24]
lmode=['bsp','tap','aap']
lmode_=['bsp-','tap-','aap-']
lvername={1:'time', 2:"iter", 3:"time-acc", 4:"iter-acc"}

def prio2bs(bs0, p0, n=None):
    if n:
        return int(bs0 * p0 * n);
    return int(bs0*p0);

def saveimg(name):
    plt.savefig(name+'.png')
    plt.savefig(name+'.pdf')


# ---- LR ----
    
os.chdir(r'E:\Code\FSB\score\lr\1000-10k')

def LRNameBBS(bbs, i, m, lr='0.01'):
    return '%d-%s/%s-%i' % (bbs*i*i, lr, m, i)

def LRDrawBBS(bbs, i, fn):
    drawList(str(bbs*i**2)+'-0.01/',genFLpost(lmode,'-'+str(i)))
    plt.ylim(0)
    plt.tight_layout()
    saveimg(fn)


for i in [1,2,4,8,12,16,20,24]:
    bbs=10
    LRDrawBBS(bbs, i, 'lr-cmp-%d-%d' % (i,bbs))
    plt.close()
    bbs=20
    LRDrawBBS(bbs, i, 'lr-cmp-%d-%d' % (i,bbs))
    plt.close()


l=[ [LRNameBBS(bbs, i, m) for i in [16,24]] for m in lmode]
drawListCmp('', l[0], l[1], l[2])
plt.ylim(0)
plt.tight_layout()
saveimg('lr-cmp-scale-bbs-16,24')

l=[4,8,16]
drawListCmp('2000-0.01/', genFL('bsp-',l), genFL('tap-',l), genFL('aap-',l))
plt.ylim(0)
plt.tight_layout()
saveimg('lr-cmp-scale-' + ','.join([str(v) for v in l]))


l=['%d-0.01/aap-%i' % (10*i*i,i) for i in ln]
drawList('', l)
plt.ylim(0)
plt.tight_layout()
saveimg('lr-scale-aap-bbs-10')

# priority of LR
bs0=10000
l=[0.01, 0.05, 0.1, 0.15, 0.2]
fnl0=genFL('',[prio2bs(bs0,v) for v in l],'-0.01/bsp-4')
drawList('',genFL('',fnl0),n=None,ver=1)

pre='10000-0.01/bsp-4-'
drawListCmp('',fnl0,genFL(pre+'pso',l),n=None,ver=1)
drawListCmp('',fnl0,genFL(pre+'ps',l),genFL(pre+'pg',l),n=None,ver=2)
drawListCmp('',fnl0,genFL(pre+'psp',l),genFL(pre+'pgp',l),n=None,ver=1)
drawListCmp('',fnl0,genFL(pre+'pso',l),genFL(pre+'pgo',l),n=None,ver=1)

k=0.01
lr=[0.01, 0.05, 0.1]
lgd=['sgd']+genFL('psgd-r',lr)

drawList('',[str(prio2bs(bs0,k))+'-0.01/bsp-4',pre+'pg'+str(k)+'-r0']+genFL(pre+'p'+str(k)+'-',lr,'-r0'),n=None,ver=1)
drawList('',[str(prio2bs(bs0,k))+'-0.01/bsp-4',pre+'ps'+str(k)+'-r0']+genFL(pre+'ps'+str(k)+'-',lr),n=None,ver=1)

drawListCmp('',[str(prio2bs(bs0,k))+'-0.01/bsp-4',pre+'pso'+str(k),pre+'pgo'+str(k)],genFL(pre+'psr'+str(k)+'-',lr),genFL(pre+'pgr'+str(k)+'-',lr),n=None,ver=1)

d=0.9;r=0.01
drawListCmp('',[str(prio2bs(bs0,k))+'-0.01/bsp-4'],[None]+genFL(pre+'p'+str(k)+'-r',lr,'-d'+str(d)),n=None,ver=1)

k=0.1;d=0.9
lr=[0, 0.01, 0.05, 0.1]
lgd=['sgd','psgd-no','psgd-r0.01','psgd-r0.05','psgd-r0.1']
drawListCmp('',[str(prio2bs(bs0,k))+'-0.01/bsp-4'],[None]+genFL(pre+'p'+str(k)+'-r',lr,'-d'+str(d)),n=None,ver=1)
plt.legend(['sgd']+genFL('psgd-r',lr))

for i in lvername:
    drawListCmp('',[str(prio2bs(bs0,k))+'-0.01/bsp-4'],[None]+genFL(pre+'p'+str(k)+'-r',lr,'-d'+str(d)),n=None,ver=i)
    plt.legend(['sgd']+genFL('psgd-r',lr))
    plt.savefig('cmp-p'+str(k)+'-d'+str(d)+'-'+lvername[i]+'.pdf')

k=0.1;r=0.01
ld=[0.9, 0.8, 0.7]
drawList('',genFL(pre+'p'+str(k)+'-r'+str(r)+'-d',ld),n=None,ver=1)
plt.legend(genFL('decay-',ld))

# k-r combination
drawListCmp('10000-0.01/krd/bsp-4-',['p0.02-r0.01-d0.8','p0.03-r0.01-d0.8','p0.05-r0.01-d0.8'],['p0.01-r0.02-d0.8','p0.01-r0.03-d0.8','p0.01-r0.05-d0.8'],ver=1)
plt.ylim([0,2.5])
plt.grid(True)
plt.legend(['k:2%,r1%','k:1%,r2%','k:3%,r1%','k:1%,r3%','k:5%,r1%','k:1%,r5%'])

drawListCmp('10000-0.01/bsp-4-',['p0.02-r0.01-ld','p0.03-r0.01-ld','p0.05-r0.01-ld'],['p0.01-r0.02-ld','p0.01-r0.03-ld','p0.01-r0.05-ld'],ver=1)

# speed 5%
drawList('10000-0.01/',['../500-0.01/bsp-4','bsp-4-p0.03-r0.02-lp','bsp-4-p0.03-r0.02-ld','bsp-4-p0.03-r0.02-ld-vj'],ver=1)
plt.xlim([-1,41])
# speed 6%
drawList('10000-0.01/',['../600-0.01/bsp-4','bsp-4-p0.05-r0.01-lp','bsp-4-p0.05-r0.01-ld','bsp-4-p0.05-r0.01-ld-vj'],ver=1)
plt.legend(['SGD','PSGD','PSGD+D','PSGD+D+A'])
plt.grid(True)

# ---- KM ----
os.chdir(r'E:\Code\FSB\score\km\100,500-100k')

namepre='km15'

