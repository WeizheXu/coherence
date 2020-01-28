import sys
sys.path.append('./semvecpy/semvecpy/vectors')
sys.path.append('./semvecpy/semvecpy')
sys.path.append('./spg')
import os
import semvec_utils as sv
import vectors.real_vectors as rv
from platform import python_version
print("python version: " + str(python_version()))
from os import listdir
from os.path import isfile,join
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from textblob import TextBlob as tb
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import spearmanr
from IPython.display import Markdown, display
from sklearn.metrics import roc_curve, auc, average_precision_score, precision_recall_curve, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD
from scipy.stats import linregress
from scipy.stats import pearsonr
import spacy
import fileinput
import re
import gensim
from gensim.models import Word2Vec
import gensim.models.keyedvectors as kv
import spg.speechgraph.speechgraph as sg
import networkx as nx
import altair as alt

#option is "naive", "stem" or "partial"
def generategraph(textfile, option):
    if option == "naive":
        graph = sg.naiveGraph() # create a Naive graph
    elif option == "stem":
        graph = sg.stemGraph() # create a Stem Graph
    elif option == "partial":
        graph = sg.posGraph() # create a Part of Speech Graph
    else:
        graph = sg.naiveGraph()
    
    sample = open(textfile, 'rb')
    s = sample.read().decode('utf8', 'ignore')
    result = graph._text2graph(s)
    sample.close()
    return result

def model2dict(model):
    term_dict = {}

    for word in model.wv.vocab:
        term_dict[word] = model.wv.__getitem__(word)

    #normalize term_dict vectors
    for key in term_dict:
        term_dict[key] = term_dict[key]/np.linalg.norm(term_dict[key])

    return term_dict

def maken2vmodel(textfile, namedotmodel, dimension, walklength, numberwalks):
    testgraph = generategraph(textfile, "naive")
    print('creating model from graph')
    n2v = Node2Vec(testgraph, dimensions=dimension, walk_length=walklength, num_walks=numberwalks)
    model = n2v.fit(window=10, min_count=1)
    print('model created')
    model.save(namedotmodel)
    print('model saved')

def normalize(series):
    nseries = []
    maximum = series.max()
    minimum = series.min()
    for x in series:
        nx = (x-minimum)/(maximum-minimum)
        nseries.append(nx)
    nseries = pd.Series(nseries)
    return nseries

def get_idf(word, idfdict):
    if word in idfdict:
        return idfdict[word]
    else:
        return 1

def generatecoherence(vectors):
    seqcohs=[]
    fromean=[]
    fromfirst=[]
    fromrunningmean=[]
    
    #seq vectors
    if len(vectors) > 1:
        lastvec=vectors[0]
        for vec in vectors[1:]:
            seqcohs.append(np.dot(vec,lastvec))
            lastvec=vec
    
    #mean vectors
    meanvec=np.sum(vectors,axis=0)
    meanvec=meanvec/np.linalg.norm(meanvec)
    for vec in vectors:
        fromean.append(np.dot(vec,meanvec))
    
    #running mean vectors
    if len(vectors) > 0:
        runningmean = np.zeros(len(vectors[0]))
        for index, vec in enumerate(vectors):
            runningmean = np.add(runningmean, vec)
            runningmean = runningmean/np.linalg.norm(runningmean)
            fromrunningmean.append(np.dot(vec, runningmean))
    
    #from first vectors
    for vec in vectors:
        fromfirst.append(np.dot(vec,vectors[0]))
    
    #from all vectors
    fromall=np.array(vectors)
    fromall=np.dot(fromall, fromall.T)
    
    return [seqcohs, fromean, fromrunningmean, fromfirst, fromall]

def wordcoherence(term_dict,incomingdata):
    stop_words = set(stopwords.words('english')) 
    terms = word_tokenize(incomingdata)
    terms = [w for w in terms if not w in stop_words] 

    localwordvectors=[]
    
    for term in terms:
        term=term.lower()
        if term in term_dict:
            v0=term_dict[term]
            localwordvectors.append(v0)
    
    #return localwordvectors
    return generatecoherence(localwordvectors)

def sentcoherenceweighted(term_dict,incomingdata):
    ps=sent_tokenize(incomingdata)
    stop_words = set(stopwords.words('english')) 
    sentvecs=[]
    
    for sent in ps:
        sentvec=np.zeros(len(term_dict[next(iter(term_dict))]))
        terms=word_tokenize(sent)
        terms= [w for w in terms if not w in stop_words] 
        for term in terms:
                    if term.lower() in term_dict:
                        weight = get_idf(term.lower(), idf_dict)
                        v1=term_dict[term.lower()]*weight
                        sentvec=np.add(v1,sentvec)
       
        if np.linalg.norm(sentvec) > 0:
            sentvec=sentvec/np.linalg.norm(sentvec)
            sentvecs.append(sentvec)
    
    return generatecoherence(sentvecs)