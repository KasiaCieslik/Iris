import sys
import numpy as np
import pandas as pd

sys.path.append('/home/kasia/sentiment-analysis-of-tweets-using-emoticons')
from data_preparation.read_data import read_csv,save_pickle
from data_preprocessing.clean_tweet import preprocess_data
from features.spacy_helpers import setup_spacy, add_spacy_features
from features.prepare_target import add_polarity_scores, sum_polarity, prepare_target
from features.train_test_preparation import y_X_preparation, train_test_preparation_for_model
from data_preprocessing.clean_dataframe import additional_processing, downsample_target
from data_preparation.reference_emoji_list import prepare_reference_emoji_list

nlp = setup_spacy()
df = read_csv("../data/raw/raw_tweets.csv")
df = preprocess_data(df)
df = add_spacy_features(df=df, nlp=nlp)
df = additional_processing(df)

emoji_dict = prepare_reference_emoji_list(nlp=nlp)
df = add_polarity_scores(df, emoji_dict, emoji_column='unique_emoji')
df = df[df.polarity_for_unique_emoji.map(bool)]
df = sum_polarity(df)
df = prepare_target(df)
df = downsample_target(df)

X, y = y_X_preparation(df)
X_train, X_test, y_train, y_test = train_test_preparation_for_model(X, y)

filenames = ['X_train.pickle', 'X_test.pickle', 'y_train.pickle', 'y_test.pickle']
data = [X_train, X_test, y_train, y_test]
path = '../data/processed/'
for filename,values in dict(zip(filenames,data)).items():
    save_pickle(values,path, filename)