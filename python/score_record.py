# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 10:42:28 2019

@author: yanxi
"""

import pandas
import numpy as np


def loadScore(filename, n=None, columns=[0,1,2]):
    d = pandas.read_csv(filename,skiprows=0,header=None)
    data = d[:n][columns]
    return np.array(data)


def loadScoreProcessTime(filename, portionRemove, fixedRemove=0, n=None, columns=[0,1,2]):
    data = loadScore(filename, n, columns)
    niter = data[-1,0]
    data[:,1] -= (data[:,0] * portionRemove / niter) + fixedRemove
    return data


def dumpScore2File(filename, data):
    with open(filename, 'w') as f:
        for line in data:
            f.write(','.join([str(v) for v in line]))
            f.write('\n')

# the portionRemove should be <all-priority-time> + <k>*<gradient-calc-time>
def processScoreFile(fnInput, fnOutput, portionRemove, fixedRemove=0):
    d = loadScoreProcessTime(fnInput, portionRemove, fixedRemove)
    dumpScore2File(fnOutput, d)


#os.chdir(r'E:\Code\FSB\score\lr\1000-10k')
#pre='10000-0.01/bsp-4-'
#k=0.01;processScoreFile(pre+'ps'+str(k)+'.txt',pre+'psp'+str(k)+'.txt',264+(1-k)*295)
#k=0.05;processScoreFile(pre+'ps'+str(k)+'.txt',pre+'psp'+str(k)+'.txt',261+(1-k)*293)
#k=0.1;processScoreFile(pre+'ps'+str(k)+'.txt',pre+'psp'+str(k)+'.txt',259+(1-k)*288.5)
#k=0.15;processScoreFile(pre+'ps'+str(k)+'.txt',pre+'psp'+str(k)+'.txt',255+(1-k)*285)
#k=0.2;processScoreFile(pre+'ps'+str(k)+'.txt',pre+'psp'+str(k)+'.txt',252+(1-k)*281.3)
#k=0.3;processScoreFile(pre+'ps'+str(k)+'.txt',pre+'psp'+str(k)+'.txt',245,(1-k)*274)

#k=0.05;processScoreFile(pre+'pg'+str(k)+'.txt',pre+'pg'+str(k)+'-ben.txt',190+(1-k)*373)
#k=0.01;processScoreFile(pre+'pg'+str(k)+'.txt',pre+'pg'+str(k)+'-ben.txt',194.5+(1-k)*377)
#k=0.1;processScoreFile(pre+'pg'+str(k)+'.txt',pre+'pg'+str(k)+'-ben.txt',191+(1-k)*370)
#k=0.15;processScoreFile(pre+'pg'+str(k)+'.txt',pre+'pg'+str(k)+'-ben.txt',218+(1-k)*332)
#k=0.2;processScoreFile(pre+'pg'+str(k)+'.txt',pre+'pg'+str(k)+'-ben.txt',215+(1-k)*327.8)
#k=0.3;processScoreFile(pre+'pg'+str(k)+'.txt',pre+'pg'+str(k)+'-ben.txt',210+(1-k)*320.7)


