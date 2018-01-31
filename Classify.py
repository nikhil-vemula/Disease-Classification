# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 21:22:13 2018

@author: vsnick
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
#Import training data
train = pd.read_csv('train.csv')
train.symptoms
train.disease
#Vectorizing the symptoms
tfid_vectorizer = TfidfVectorizer(min_df=1)
X = tfid_vectorizer.fit_transform(train.symptoms)
tfid_vectorizer.get_feature_names()

#Vectorizing the diseases
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
Y = le.fit_transform(train.disease)
le.classes_
#Fitting the model
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X,Y)

test = pd.read_csv('test.csv')
test.symptoms

predicted = clf.predict_proba(tfid_vectorizer.transform(test.symptoms))
prob = predicted.tolist()[0]
prob = [x*100 for x in prob]
#Probabilities of diseases
print dict(zip(le.classes_,prob))
#Most probable
predicted = clf.predict(tfid_vectorizer.transform(test.symptoms))
print le.classes_[predicted]



