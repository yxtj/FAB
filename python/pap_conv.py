# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 18:29:45 2019

@author: yanxi
"""

import numpy as np

class Kmeans():        
    def __init__(self, dataset):
        self.data = np.array(dataset)
        self.len = self.data.shape[0]
        
    def ready(self, nc):
        self.nc=nc
        self.dim = self.data.shape[1]
        self.center = self.data[np.random.randint(0, self.len, self.nc), :]
        dis = ((self.data - self.center[:,np.newaxis,:])**2).sum(axis=2)
        self.z = np.argmin(dis, axis=0)
    
    def estep(self):
        dis = ((self.data - self.center[:,np.newaxis,:])**2).sum(axis=2)
        self.z = np.argmin(dis, axis=0)
    
    def mstep(self):
        temp = [ self.data[self.z==k].mean(0) for k in range(self.nc)]
        self.center = np.array(temp)
        
    def loss(self):
        dis = np.sqrt(((self.data - self.center[self.z,:])**2).sum(axis=1))
        return dis.sum()


class KmeansIncr(Kmeans):    
    def __init__(self, dataset):
        super().__init__(dataset)
    
    def ready(self, nc, bs=None):
        self.nc=nc
        self.dim = self.data.shape[1]
        self.bs = min(bs, self.len) if bs else self.len
        self.p = 0
        self.z = np.random.randint(0, self.nc, self.len)
        self.center = np.array([self.data[self.z==k].mean(0) for k in range(self.nc)])
        self.c = np.array([sum(self.z==k) for k in range(self.nc)])
        self.s = np.array([self.data[self.z==k].sum(0) for k in range(self.nc)])
    
    def estep(self, bs = None, center = None):
        bs=min(bs, self.len) if bs is not None else self.bs
        batch = np.arange(self.p, min(self.p + bs, self.len))
        if self.p + bs > self.len:
            left = self.p + bs - self.len
            np.hstack((batch, np.arange(0, left)))
        self.p = (self.p + bs) % self.len
        
        if center is None:
            center = self.center
        
        for i in batch:
            d=self.data[i,:] # shape is (self.dim,)
            dis = center - d
            dis = (dis**2).sum(1)
            newz = dis.argmin()
            oldz = self.z[i]
            self.z[i] = newz
            self.s[oldz]-=d
            self.c[oldz]-=1
            self.s[newz]+=d
            self.c[newz]+=1
            
    
    def mstep(self):
        self.center = self.s / self.c[:,np.newaxis]
    
    def eta(self):
        delta = (self.s / self.c[:,np.newaxis]) - self.center
        temp = [self.data[self.z==k].sum(0) - self.center[k,:] for k in range(self.nc)]
        grad = np.array(temp)
        # result is 1 / self.c
        return delta/grad
    
    def lossOnline(self):
        dis = np.sqrt(((self.data - self.center[self.z,:])**2).sum(axis=1))
        return dis.sum()
    
    def loss(self, center=None):
        if center is None:
            center = self.center
        dis = ((self.data - center[:,np.newaxis,:])**2).sum(axis=2)
        z = np.argmin(dis, axis=0)
        dis = np.sqrt(((self.data - self.center[z,:])**2).sum(axis=1))
        return dis.sum()
    
def __test__():
    points = np.vstack(((np.random.randn(100, 2) * 0.75 + np.array([1, 0])),
                  (np.random.randn(25, 2) * 0.25 + np.array([-0.5, 0.5])),
                  (np.random.randn(25, 2) * 0.5 + np.array([-0.5, -0.5]))))
    km=KmeansIncr(points)
    # normal
    km.ready(3,30)
    l=[km.loss()]
    lo=[km.lossOnline()]
    for i in range(20):
        km.estep()
        km.mstep()
        l.append(km.loss())
        lo.append(km.lossOnline())
    #plt.plot(np.arange(len(l)),l,lo)
    
    # pap:
    km.ready(3,15)
    last=km.center
    for i in range(10):
        km.estep(15,last)
        km.estep(15)
        last=km.center
        km.mstep()
        l.append(km.loss())
        lo.append(km.lossOnline())
    #plt.plot(np.arange(len(l)),l,lo)
    

