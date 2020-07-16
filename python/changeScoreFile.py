# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 00:23:43 2019

@author: yanxi
"""

def addDummyNP(fname):
    res=[]
    with open(fname) as f:
        for line in f:
            l = line.split(',')
            l.insert(2,'0')
            res.append(','.join(l))
    if len(res) != 0:
        with open(fname, 'w') as f:
            for line in res:
                f.write(line)


