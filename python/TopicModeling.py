# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 14:01:28 2019

@author: tzhou
"""

import string
import re
import numpy as np

doc1 = "Sugar is bad to consume. My sister likes to have sugar, but not my father."
doc2 = "My father spends a lot of time driving my sister around to dance practice."
doc3 = "Doctors suggest that driving may cause increased stress and blood pressure."
doc4 = "Sometimes I feel pressure to perform well at school, but my father never seems to drive my sister to do better."
doc5 = "Health experts say that Sugar is not good for your lifestyle."

doc_complete = [doc1, doc2, doc3, doc4, doc5]
topic_complete = [0, 1, 1, 1, 0]

def doc2id(document, dictionary):
    l=[]
    pat=re.compile('['+string.punctuation+']')
    s = r.sub('',document.lower()).split()
    for word in s:
        if word not in dictionary:
            dictionary[word] = len(dictionary)
        l.append(dictionary[word])
    return l;

def iddoc2matrix(idds, nw):
    res=np.zeros([len(idds), nw])
    for i in range(len(idds)):
        for w in idds[i]:
            res[i,w]+=1
    return res

def preprocess(docs):
    d={}
    idds=[doc2id(doc,d) for doc in docs]
    m=iddoc2matrix(idds, len(d))
    return m,d


def main():
    m, d = preprocess(docs)
    nd, nw = m.shape # number of doc., number of word
    nt = len(set(topic_complete)) # number of topic
    
