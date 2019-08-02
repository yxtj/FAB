# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 11:39:05 2019

@author: yanxi
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from draw import *

plt.rcParams["figure.figsize"] = [4,3]

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

k=0.01;d=0.9
lr=[0.001, 0.01, 0.05]
drawListCmp('',[str(prio2bs(bs0,k))+'-0.001/tap-4'],[None]+genFL(pre+'p'+str(k)+'-r',lr,'-d'+str(d)),n=None,ver=1)
plt.legend(['sgd']+genFL('psgd-r',lr))


k=0.01;r=0.01
ld=[1-k, 0.9, 0.8, 0.7]
drawListCmp('',[str(prio2bs(bs0,k))+'-0.001/tap-4'],[None]+genFL(pre+'p'+str(k)+'-r'+str(r)+'-d',ld),n=None,ver=1)
plt.legend(['sgd']+genFL('decay-',ld))


lp_cnn=[0.01,0.02,0.03,0.04]
drawListCmp('60000-0.001/',genFL('tap-4-p',lp_cnn,'-r0.01-ld'),genFL('tap-4-p',lp_cnn,'-r0.01-ld-vj'),ver=1)
plt.ylim([0.5, 3])
plt.legend(np.array([['k:'+str(v*100)+'%','k:'+str(v*100)+'%+A'] for v in lp_cnn]).flatten(),ncol=2)
#plt.legend(np.array([['top:'+str(v*100)+'%','top:'+str(v*100)+'%+A'] for v in lp_cnn]).flatten(),ncol=2)

lr_cnn=[0.01,0.02,0.03,0.04]
drawListCmp('60000-0.001/',genFL('tap-4-p0.02-r',lr_cnn,'-ld'),genFL('tap-4-p0.02-r',lr_cnn,'-ld-vj'),ver=1)
#plt.xlim([0,100])
plt.ylim([0.5, 3])
plt.legend(np.array([['r:'+str(v*100)+'%','r:'+str(v*100)+'%+A'] for v in lp_cnn]).flatten(),ncol=2)

# convergence speed
drawList('60000-0.001/',['../600-0.001/bsp-4']+genFL('tap-4-p0.04-r0.01-',['lp','ld','ld-vj']),ver=1)
plt.legend(['SGD','PSGD','PSGD+D','PSGD+D+A'])
plt.grid(True)

drawList('60000-0.001/',genFL('../',[900],'-0.001/tap-4')+genFL('tap-4-p0.01-r0.01-lp-l',['','-rla']),ver=1)
drawList('60000-0.001/',genFL('../',[3000],'-0.001/tap-4')+genFL('tap-4-p0.04-r0.01-lp-l',['','-rla']),ver=1)
plt.xlim([-10,310])
plt.legend(['SGD','PSGD','A-PSGD'])
plt.grid(True)

# fix r=1%, range k
lk=[0.01,0.02,0.03]
drawListCmp('60000-0.001/',genFL('../',[str(prio2bs(bs0,p)) for p in lk],'-0.001/tap-4'),genFL('tap-4-p',lk,'-r0.01-lp-l'),ver=1)
plt.xlim([-10,310])
plt.legend(['SGD-1%','PSGD-k:1%,r:1%','SGD-2%','PSGD-k:2%,r:1%','SGD-3%','PSGD-k:3%,r:1%'])
plt.grid(True)


# fix k+r = 5%
drawListCmp('60000-0.001/',['../3000-0.001/tap-4'],[None]+['tap-4-p0.01-r0.04-ld','tap-4-p0.02-r0.03-ld','tap-4-p0.03-r0.02-ld','tap-4-p0.04-r0.01-ld'],ver=1)
plt.xlim([-10,310])
plt.ylim([0.5,3])
plt.grid(True)
plt.legend(['SGD-5%','PSGD-k:1%,r:4%','PSGD-k:2%,r:3%','PSGD-k:3%,r:2%','PSGD-k:4%,r:1%'])

drawList('60000-0.001/',['tap-4-p0.01-r0.04-ld','tap-4-p0.02-r0.03-ld','tap-4-p0.03-r0.02-ld','tap-4-p0.04-r0.01-ld'],ver=1)
plt.legend(['k:1%,r:4%','k:2%,r:3%','k:3%,r:2%','k:4%,r:1%'])


drawListCmp('60000-0.001/',['../3000-0.001/tap-4'],[None]+genFL('tap-4-',['p0.01-r0.04','p0.02-r0.03','p0.03-r0.02','p0.04-r0.01'],'-lp-l-rla'),ver=1)
plt.legend(['SGD-5%','PSGD-k:1%,r:4%','PSGD-k:2%,r:3%','PSGD-k:3%,r:2%','PSGD-k:4%,r:1%'])
plt.grid(True)

# compare k-r combination
drawListCmp('60000-0.001/',['tap-4-p0.01-r0.03-ld','tap-4-p0.01-r0.04-ld','tap-4-p0.02-r0.03-ld'],['tap-4-p0.03-r0.01-ld','tap-4-p0.04-r0.01-ld','tap-4-p0.03-r0.02-ld'],ver=1)
plt.ylim([0.5,3])
plt.grid(True)
plt.legend(np.array([['PSGD-k:%d,r:%d'%(k,r),'PSGD-k:%d,r:%d'%(r,k)] for k,r in [(1,3),(1,4),(2,3)]]).flatten())
plt.legend(np.array([['k:%d%%,r:%d%%'%(k,r),'k:%d%%,r:%d%%'%(r,k)] for k,r in [(1,3),(1,4),(2,3)]]).flatten())

# speed: fix k = 1%, 2%, 3%
drawListCmp('60000-0.001/',genFL('../',[6,12,18,24],'00-0.001/tap-4'),genFL('tap-4-p',[0.01,0.02,0.03,0.04],'-r0.01-ld'),ver=1)
plt.xlim([-10,310])
plt.legend(np.array([genFL('SGD-',[1,2,3,4],'%'),genFL('PSGD-k:',[1,2,3,4],'%,r:1%')]).T.flatten())
plt.grid(True)

drawListCmp('60000-0.001/',genFL('../',[6,12,18],'00-0.001/tap-4'),genFL('tap-4-p',[0.01,0.02,0.03],'-r0.01-ld'),ver=1)
plt.xlim([-10,310])
plt.legend(np.array([genFL('SGD-',[1,2,3],'%'),genFL('PSGD-k:',[1,2,3],'%,r:1%')]).T.flatten())
plt.grid(True)

# find tune of k and r
lp_cnn2=[0.002,0.003,0.005]
lr_cnn2=[0.002,0.003,0.005]

# scale
l_nw=[1,2,4,6,8,10,12,16]
drawList('60000-0.001/',genFL('tap-',l_nw,'-p0.05-r0.01-ld'),ver=1)
drawList('60000-0.001/',genFL('tap-',l_nw,'-p0.02-r0.01-ld'),ver=1)
plt.legend(l_nw,ncol=2)

# scale - speedup
drawScale('60000-0.001/',l_nw,genFL('tap-',l_nw,'-p0.05-r0.01-ld'),0.87,True,ref=True,fit=True,est=True)
drawScale('60000-0.001/',l_nw,genFL('tap-',l_nw,'-p0.02-r0.01-ld'),0.899687,True,ref=True,fit=True,est=True)
plt.xticks(range(0,17,4))
# scale - time
drawScale('60000-0.001/',l_nw,genFL('tap-',l_nw,'-p0.05-r0.01-ld'),0.87,False)
plt.xticks(range(0,17,4))


# compare priority estimation method
refcnn='score/cnn/28x28,10c5x5rp2x2,2c5x5rp2x2,10f-60k/60-0.001/tap-4'
precnn='cnn/28x28,10c5x5rp2x2,2c5x5rp2x2,10f-60k/60000-0.001/'
drawListCmp('',[refcnn],[None]+genFL('score',['','-e2','-e3','-ee'],'/'+precnn+'tap-4-p0.01-r0-ld'))
plt.legend(['SGD']+genFL('PSGD',['e1','-e2','-e3','-ee']))

drawListCmp('',['60-0.001/tap-4','120-0.001/tap-4','600-0.001/tap-4'],genFL('60000-0.001/tap-4-p',[0.001,0.002,0.01],'-r0-lp-l'),genFL('60000-0.001/tap-4-p',[0.001,0.002,0.01],'-r0-lp-q'),ver=1)
