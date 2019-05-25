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


# ---- MLP ----

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

# ---- CNN ----

os.chdir(r'E:\Code\FSB\score\cnn\28x28,4c5x5p2x2,3c5x5p2x2,10f-60k')
os.chdir(r'E:\Code\FSB\score\cnn\28x28,12c5x5p4x4,10f-60k')
os.chdir(r'E:\Code\FSB\score\cnn\28x28,10c5x5rp2x2,2c5x5rp2x2,10f-60k')
os.chdir(r'E:\Code\FSB\score\cnn\28x28,10c5x5rp3x3,10f-60k')

namepre='mnist-c12'
namepre='mnist-c4c3'

for m in lmode:
    drawList('1000-0.001/', genFL(m+'-',ln,''))
    saveimg(namepre+'-scale-%s' % m)
    plt.close()

plt.rcParams["figure.figsize"] = [6,4.5]

l=[4,12,24]

drawListCmp('1000-0.001/', genFL('bsp-',l,''),genFL('tap-',l,''), genFL('aap-',l,''))
saveimg(namepre+'-cmp-%s' % ','.join([str(v) for v in l]))

drawListCmp('1000-0.001/', genFL('bsp-',[24],''), genFL('tap-',[24],''), genFL('aap-',[24],''),None,120)
saveimg(namepre+'-cmp-24')

drawListCmp('1000-0.001/', genFL('tap-',[24],''), genFL('aap-',[24],''),None,20)
saveimg(namepre+'-cmp-24-zoomin')

drawListCmp('1000-0.001/', genFL('bsp-',[12],''), genFL('tap-',[12],''), genFL('aap-',[12],''),None,120)
saveimg(namepre+'-cmp-12')

drawListCmp('1000-0.001/', genFL('tap-',[12],''), genFL('aap-',[12],''),None,30)
saveimg(namepre+'-cmp-12-zoomin')

# priority for CNN
bs0=60000
pre=str(bs0)+'-0.001/bsp-4-'
l=[0.01, 0.05, 0.1, 0.15, 0.2]
drawListCmp('',genFL('',[prio2bs(bs0,v) for v in l],'-0.001/bsp-4'),genFL(pre+'pso',l),genFL(pre+'pgo',l),n=None,ver=1)

k=0.01
lr=[0.001, 0.01, 0.05]
drawList('',[str(prio2bs(bs0,k))+'-0.001/bsp-4',pre+'pgo'+str(k)]+genFL(pre+'pgr'+str(k)+'-',lr),n=None,ver=1)
drawList('',[str(prio2bs(bs0,k))+'-0.001/bsp-4',pre+'pso'+str(k)]+genFL(pre+'psr'+str(k)+'-',lr),n=None,ver=1)
drawListCmp('',[str(prio2bs(bs0,k))+'-0.001/bsp-4',pre+'pso'+str(k),pre+'pgo'+str(k)],genFL(pre+'psr'+str(k)+'-',lr),genFL(pre+'pgr'+str(k)+'-',lr),n=None,ver=1)
drawListCmp('',[str(prio2bs(bs0,k))+'-0.001/bsp-4'],[None,pre+'pso'+str(k)]+genFL(pre+'psr'+str(k)+'-',lr),[None,pre+'pgo'+str(k)]+genFL(pre+'pgr'+str(k)+'-',lr),n=None,ver=1)


pre=str(bs0)+'-0.001/tap-4-'

k=0.01;d=0.9
lr=[0.001, 0.01, 0.05]
drawListCmp('',[str(prio2bs(bs0,k))+'-0.001/tap-4'],[None]+genFL(pre+'p'+str(k)+'-r',lr,'-d'+str(d)),n=None,ver=1)
plt.legend(['sgd']+genFL('psgd-r',lr))

for i in lvername:
    drawListCmp('',[str(prio2bs(bs0,k))+'-0.001/tap-4'],[None]+genFL(pre+'p'+str(k)+'-r',lr,'-d'+str(d)),n=None,ver=i)
    plt.legend(['sgd']+genFL('psgd-r',lr))
    plt.savefig('cmp-p'+str(k)+'-d'+str(d)+'-'+lvername[i]+'.pdf')


k=0.01;r=0.01
ld=[1-k, 0.9, 0.8, 0.7]
drawListCmp('',[str(prio2bs(bs0,k))+'-0.001/tap-4'],[None]+genFL(pre+'p'+str(k)+'-r'+str(r)+'-d',ld),n=None,ver=1)
plt.legend(['sgd']+genFL('decay-',ld))


# ---- KM ----
os.chdir(r'E:\Code\FSB\score\km\100,500-100k')

namepre='km15'

