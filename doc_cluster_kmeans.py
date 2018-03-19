# -*- coding: utf-8 -*-
"""
Created on Sat Mar 03 19:42:16 2018

@author: anirudh.saraiya
"""
import os
import re
import bs4
from bs4 import BeautifulSoup as BS
from bs4 import NavigableString
from datetime import datetime
import time

import pandas as pd
import numpy as np

from scipy.spatial.distance import cdist
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

def alphanum(tex):
    
    import string
    st = string.punctuation
    count = 0
    for char in tex:
        if char in st:
            count+=1
    
    if count > 0:
        return 0
    else:
        if tex.isalpha() == False and tex.isdigit() == False:
            return 1
        else:
            return 0
        
def refine_text(text):
    ##remove unwanted punctuations
    text = re.sub('[*!?+();&/:]+', ' ' ,text)
    text = re.sub('[.,$€]','',text)
    
    ##Coverting digits and date to same string
    text = text.split()
    pattern = re.compile("\w+[-/.]\w+[-/.]\w+")
    for i in range(len(text)):
        text[i] = text[i].lower() ##lower the text
        text[i] = text[i].strip("-")
        if text[i].isdigit():
            text[i] = "DDDD"
        elif pattern.match(text[i]):
            text[i] = "DaTe"
    ##stop words for english
    stops = open("./stops.txt","r").read().split("\n")
    meaningful_words = [w for w in text if not w in stops] 
    
    for i in range(len(meaningful_words)):
        if alphanum(meaningful_words[i]) ==1:
            meaningful_words[i] = "AlphaNum"
    
    return " ".join(meaningful_words)
    
def data_collection(path):

    data_collect = []
    filenames = []
    count =0
    print "Creating bag of words...\n"
    for filename in os.listdir(path):
        
        req_html = open(path + "/" +filename,"r").read()
        req_html= req_html.replace("&nbsp;"," ")
        req_html= req_html.replace("12:00:00 AM"," ")
        req_html= req_html.replace("0:00:00"," ")
    
        soup = BS(req_html, 'html.parser')
        #    s.write(filename+"\n")
        try:
            doctext = soup.body.get_text(separator=u' ')
        except Exception as e:
            doctext = " "
            print("error in " + str(filename))

        
        refined_doc_words = refine_text(doctext)
        data_collect.append(refined_doc_words)
        filenames.append(filename)
        count+=1
        
        if count%100 == 0:
            print(("%s documents completed....") %(count))
    
    return data_collect,filenames


def k_means_elbow(data_collect,n_clust):
    # vectorizer = TfidfVectorizer(sublinear_tf=True, analyzer='word', 
    #                  stop_words='english',use_idf=True,ngram_range=(1,2),max_features = 10000)

    vectorizer = TfidfVectorizer(sublinear_tf=True, analyzer='word', 
                     stop_words='english',use_idf=True,max_features = 10000)
    
    train_data_features = vectorizer.fit_transform(data_collect)
    
    train_data_features = train_data_features.toarray()
        
    print train_data_features.shape
    
    #  words in the vocabulary
#    vocab = vectorizer.get_feature_names()
    #print vocab
    

    # k means determine k
    distortions = []
    K = range(1,n_clust)
    count= 0
    for k in K:
        count+=1
        if count%10 == 0:
            print("%s clusters finished out of %s clusters" %(k,len(K)))
        kmeanModel = KMeans(n_clusters=k).fit(train_data_features)
        kmeanModel.fit(train_data_features)
        distortions.append(sum(np.min(cdist(train_data_features, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / train_data_features.shape[0])
    #    distortions.append(kmeanmodel.inertia_)
    clusters = kmeanModel.labels_.tolist()
    
    plt.plot(K,distortions,"bx-")
    plt.xlabel("k")
    plt.ylabel("Distortion")
    plt.title("Elbow Curve")
    # plt.show()
    plt.savefig("./doc_clusters_elbow.png")
    
    return K,distortions,clusters,plt

def k_means(data_collect,n_clust):

    vectorizer = TfidfVectorizer(sublinear_tf=True, analyzer='word', 
    stop_words='english',use_idf=True,ngram_range=(1,4),max_features = 10000)
    
    train_data_features = vectorizer.fit_transform(data_collect)
    
    train_data_features = train_data_features.toarray()
        
    print train_data_features.shape
    num_clusters = n_clust
    
    km = KMeans(n_clusters=num_clusters,random_state = 42)
    
    km.fit(train_data_features)
    
    clusters = km.labels_.tolist()

    cluster_centers = km.cluster_centers_

    plt.hist(clusters,edgecolor='black', linewidth=1.2)
    plt.title("Cluster Histogram")
    plt.xlabel("Cluster Range")
    plt.ylabel("Frequency")
    
    return clusters,cluster_centers,plt

if __name__ == "__main__":
    path = ""
    req_data,filenames = data_collection(path)
    n_clust = 50
    start_t = time.time()
    clusters,cluster_centers,plt = k_means(req_data,n_clust)
    
    # K,distortions,clusters,plt = k_means_elbow(req_data,n_clust)

    total_time = time.time() - start_t
    print("Clustering cpmpleted in %s seconds " %round(total_time,2))
    df = pd.DataFrame()
    df["Filenames"] = filenames
    df["Cluster"] = clusters
    df.to_csv("./doc_clusters.csv",encoding="utf-8",index=False)
    # print(len(cluster_centers))
    # plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c=clusters,s=50, cmap='viridis');
    # plt.show()
    plt.savefig("./documents_division.png")
    plt.show()

