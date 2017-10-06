# coding: utf-8

from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import re 
import fileinput 
from bs4 import BeautifulSoup
import os
import glob
from os import path
import nltk
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from collections import Counter
import pprint 
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import itertools
from collections import defaultdict
import numpy as np
from numpy import array
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn import svm
import scipy
from sklearn.linear_model import SGDClassifier
import sklearn.linear_model.tests.test_randomized_l1
from sklearn import metrics
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import csv
from sklearn import tree
from sklearn.feature_selection import RFE
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier


url_string = r"(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:[a-z]{2,13})(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:'\".,<>?������])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:[a-z]{2,13})\b/?(?!@))))"

def main():
    
    classificationTasks()



#preprocess: clean urls, decide about code mix, stem?, remove stopwords?
def preprocess():
    fp = open('train_clean.txt', 'w')
    with open("train.txt") as fn:
        for line in fn:
            text = re.sub(url_string,"URL",line)
            fp.write(text)
    fp.close()

def transformTarget():
    
    transformDict= {'Fake Seller':'0','Reseller':'1','No Seller':'2'}

    changed = open("target_transformed.txt","w")
    original = open("target.txt", 'r')
    data = original.read()
    original.close()
    for key, value in transformDict.items():
        data = data.replace(key, value)
    changed.write(data)
    changed.close()


            
def extractFeats():
    lines_list = open('train_clean.txt').read().splitlines()
    #print(len(lines_list))
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(lines_list)
    tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
    X_train_tf = tf_transformer.transform(X_train_counts)
    #print(X_train_tf)
    return(X_train_tf)
     
def classificationTasks():
    
    preprocess()
    transformTarget()
    features = extractFeats()
    target = np.array(newReadFile(open("target_transformed.txt"))).astype(np.int)
    trainfeat_np, testfeat_np, traintarget_np, testtarget_np = train_test_split(features,target, test_size = 0.2, random_state = 0)
    
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(15,), random_state=1)
    clf.fit(trainfeat_np,traintarget_np)
    y_pred = clf.predict(testfeat_np)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(testtarget_np, predictions)
    print("Accuracy for multilayer perceptron (neural Network): %.2f%%" % (accuracy * 100.0))
    
    
    #baseline: most frequent class
    clf = DummyClassifier(strategy='most_frequent',random_state=0)
    clf.fit(trainfeat_np,traintarget_np)
    y_pred = clf.predict(testfeat_np)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(testtarget_np, predictions)
    print("Accuracy for dummy majority classifier: %.2f%%" % (accuracy * 100.0))
    
    
    svm = SVC(kernel='linear', C=1)
    svm.fit(trainfeat_np,traintarget_np)
    y_pred = svm.predict(testfeat_np)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(testtarget_np, predictions)
    print("Accuracy for SVM: %.2f%%" % (accuracy * 100.0))
    
    #random forests
    clf = RandomForestClassifier(n_estimators = 50)
    clf.fit(trainfeat_np,traintarget_np)
    y_pred = clf.predict(testfeat_np)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(testtarget_np, predictions)
    print("Accuracy for Random Forest: %.2f%%" % (accuracy * 100.0))
    
    #gradient boosted trees
    xgb = XGBClassifier()
    xgb.fit(trainfeat_np,traintarget_np)
    y_pred = xgb.predict(testfeat_np)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(testtarget_np, predictions)
    print("Accuracy for gradient boosted trees: %.2f%%" % (accuracy * 100.0))
    

    
def newReadFile(f):
    #print(f)
    lst=[]
    for i in f.readlines():
        lst.append(int(i.rstrip()))
    #print lst
    return(lst)    
        
    
    

    
if __name__ == "__main__":
    main() 




































