# -*- coding: utf-8 -*-
"""
Created on Thu May 10 21:39:54 2018

@author: jishn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## importing the tab seperated list quoting is for the quotes in the text
dataset=pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting=3)
## now data cleaning removing punctuations and also steming which is basically
## comprehending from diff tense of the word eg: loved and love 
## removing captials

import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords 
#nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
corpus=[] ## In NLP this is used to reprsent a collection of text html and everything
for i in range(0,1000):
    review=re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
    review=review.lower()
    review=review.split() # to make a list
    review=[ps.stem(word) for word in review if not word in stopwords.words('english')]
    ## BY doing this all the non required words is removed using nltk library stopwords module
    
    ## next is stemming which basically is changing all the words which impart same meaning
    ## to the source word eg love loves and loved all to love because from all this we get the same knowledge 
    ## about the intention or review is it positive or negative
    
    review=" ".join(review)
    corpus.append(review)
    
    
 ## Next step is to create a Bag of word models using tokenisation
## It creates a Sparse matrix with its attributes being all the words in the corpse without repetition
## Each review being the row and each cell will indicate the occurrence of that particular word in that
## review

## we are doing this because once we create a matrix of features and corresponding data in the cell
 ## Therafter simple machine learning algorithms must be applied and it will be a classification problem
 
 ## Tokenising
 
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500) ## max features to remove some none relebvant words. this would select the top 1500 frequently used words
X=cv.fit_transform(corpus).toarray()
y=dataset.iloc[:,1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


