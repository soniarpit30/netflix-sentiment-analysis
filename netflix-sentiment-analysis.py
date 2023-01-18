# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 23:09:37 2023

@author: Arpit Soni
"""

# importing the libraries
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import string

punct = string.punctuation
stopwords = stopwords.words('english')

# on positive reviews
pos_review = pd.read_csv('C:\\Users\\sonia\\Downloads\\01-Data-Science\\github-repositories\\NLP\\netflix-sentiment-analysis\\pos.txt', encoding = 'latin-1', header = None, sep='\r\n')
pos_review['mood'] = 1
pos_review.rename(columns = {0:'review'}, inplace = True)

# on negative reviews
neg_review = pd.read_csv('C:\\Users\\sonia\\Downloads\\01-Data-Science\\github-repositories\\NLP\\netflix-sentiment-analysis\\negative.txt', encoding = 'latin-1', header = None, sep='\r\n')
neg_review['mood'] = 0
neg_review.rename(columns = {0:'review'}, inplace = True)

# cleaning the positive reviews
# lower --> remove punctuations --> remove stopwords --> lemmatization --> join back
pos_review['review'] = pos_review.review.apply(lambda x: x.lower())
pos_review['review'] = pos_review.review.apply(lambda x: " ".join([word for word in nltk.word_tokenize(x) if word not in punct]) )
pos_review['review'] = pos_review.review.apply(lambda x: " ".join([word for word in nltk.word_tokenize(x) if word not in stopwords]) )

# cleaning the negative reviews
neg_review['review'] = neg_review.review.apply(lambda x: x.lower())
neg_review['review'] = neg_review.review.apply(lambda x: " ".join([word for word in nltk.word_tokenize(x) if word not in punct]) )
neg_review['review'] = neg_review.review.apply(lambda x: " ".join([word for word in nltk.word_tokenize(x) if word not in stopwords]) )

# concat two dfs
com_reviews = pd.concat([pos_review, neg_review], axis = 0).reset_index(drop = True)

# train test split
X_train, X_test, y_train, y_test = train_test_split(com_reviews['review'].values, com_reviews['mood'].values, test_size = 0.2, random_state = 42)

train_data = pd.DataFrame({'review':X_train, 'mood':y_train})
test_data = pd.DataFrame({'review':X_test, 'mood':y_test})

vectorizer = TfidfVectorizer()
train_vectors = vectorizer.fit_transform(train_data['review'])
test_vectors = vectorizer.transform(test_data['review'])

from sklearn import svm
from sklearn.metrics import classification_report

classifier = svm.SVC()
classifier.fit(train_vectors, train_data['mood'])

# if data is balanced - we use accuracy
# if data is imbalanced - we use F1_score, confusion matrix

pred = classifier.predict(test_vectors)

report = classification_report(test_data['mood'], pred, output_dict=True)

print(f"Positive : {report['1']['recall']}")
print(f"Negative : {report['0']['recall']}")

import joblib
joblib.dump(vectorizer, 'C:\\Users\\sonia\\Downloads\\01-Data-Science\\github-repositories\\NLP\\netflix_sentiment_analysis\\tfidf_vector_model.pkl')
joblib.dump(classifier, 'C:\\Users\\sonia\\Downloads\\01-Data-Science\\github-repositories\\NLP\\netflix_sentiment_analysis\\netflix.pkl')

# load the models
tfidf = joblib.load('C:\\Users\\sonia\\Downloads\\01-Data-Science\\github-repositories\\NLP\\netflix_sentiment_analysis\\tfidf_vector_model.pkl')
model = joblib.load('C:\\Users\\sonia\\Downloads\\01-Data-Science\\github-repositories\\NLP\\netflix_sentiment_analysis\\netflix.pkl')

# predictions
data = ['good movie']
vector = tfidf.transform(data).toarray()
my_pred = model.predict(vector)
if my_pred[0] == 1:
    print('Positive review')
else:
    print('Negative review')
        




