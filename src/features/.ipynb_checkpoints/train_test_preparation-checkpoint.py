from .spacy_helpers import setup_spacy
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

import numpy as np
nlp=setup_spacy()
class average_tweet_vectorizer(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass
    
    def transform(self, X):
        word_vectorize = []
        for i,token in enumerate(X):
            word_vector_list = [nlp(word).vector for word in token]
            average_word_vector = np.mean(word_vector_list, axis=0)
            word_vectorize.append(average_word_vector.sum())
        return np.nan_to_num(np.array(word_vectorize).reshape(-1,1))
    
    def fit(self, X, y=None):
        return self

def y_X_preparation(df):
    "return numeric y,X"
    le = LabelEncoder()
    y = le.fit_transform(df['sentiment_target'])
    X = df['tokens']
    return X,y

def train_test_preparation_for_model(X,y):
    """prepare numeric features using Features Union and two classes"""
    X = X.apply(lambda x: ' '.join(x))
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)
    
    pipeline = Pipeline([
    ('feats', FeatureUnion([
       ('average_tweet_vectorizer', average_tweet_vectorizer()),
      ('tfidf',TfidfVectorizer())  
    ]))])
    X_train = pipeline.fit_transform(X_train)
    X_test = pipeline.transform(X_test)
    return X_train, X_test, y_train, y_test

