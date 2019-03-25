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

def saveimg(name):
    plt.savefig(name+'.png')
    plt.savefig(name+'.pdf')


# LR for
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


#MLP
os.chdir(r'E:\Code\FSB\score\mlp\784,300,10-60k')

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

# CNN
os.chdir(r'E:\Code\FSB\score\cnn\28x28,4c5x5p2x2,3c5x5p2x2,10f-60k')
os.chdir(r'E:\Code\FSB\score\cnn\28x28,12c5x5p4x4,10f-60k')

namepre='mnist-c12'
namepre='mnist-c4c3'

for m in lmode:
    drawList('1000-0.001/', genFL(m+'-',ln,''))
    saveimg(namepre+'-scale-%s' % m)
    plt.close()

plt.rcParams["figure.figsize"] = [6,4.5]
ln3=[4,12,24]

drawListCmp('1000-0.001/', genFL('bsp-',ln3,''),genFL('tap-',ln3,''), genFL('aap-',ln3,''))
saveimg(namepre+'-cmp-%s' % ','.join([str(v) for v in ln3]))

drawListCmp('1000-0.001/', genFL('bsp-',[24],''), genFL('tap-',[24],''), genFL('aap-',[24],''),None,120)
saveimg(namepre+'-cmp-24')

drawListCmp('1000-0.001/', genFL('tap-',[24],''), genFL('aap-',[24],''),None,20)
saveimg(namepre+'-cmp-24-zoomin')

drawListCmp('1000-0.001/', genFL('bsp-',[12],''), genFL('tap-',[12],''), genFL('aap-',[12],''),None,120)
saveimg(namepre+'-cmp-12')

drawListCmp('1000-0.001/', genFL('tap-',[12],''), genFL('aap-',[12],''),None,30)
saveimg(namepre+'-cmp-12-zoomin')



