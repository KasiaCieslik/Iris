from spacy.lang.en.stop_words import STOP_WORDS as stop_words
from data_preprocessing.clean_dataframe import unique_tokens
from spacymoji import Emoji
import string 
import spacy
import en_core_web_sm
import numpy as np

def setup_spacy():
    nlp = en_core_web_sm.load()
    emoji = Emoji(nlp)
    nlp.add_pipe(emoji, first=True)
    return nlp

def add_spacy_features(df=None, column='tidy_text', nlp=None):
    """add columns with features with spacy"""
    
    collect_tweets = []
    collect_pos = []
    collect_pos_adj = []
    collect_emoji = []

    is_corrupt = []
    #prepare tweet
    for tweet in df[column].values:
        try:
            tweet = nlp(tweet)
            is_corrupt.append(False)
        except:
            is_corrupt.append(True)
            continue

        my_emoji = [word for word in tweet if word._.is_emoji]
        my_tokens = [word for word in tweet if not word._.is_emoji]
        my_tokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON_" 
                    else word.lower_ for word in my_tokens]
        my_tokens = [word for word in my_tokens if
                    word not in stop_words and word not in string.punctuation]

        my_pos = [word.pos_ for word in tweet]
        my_pos_adj = [word for word in tweet if word.pos_ == 'ADJ']
        collect_tweets.append(my_tokens)
        collect_pos.append(my_pos)
        collect_pos_adj.append(my_pos_adj)
        collect_emoji.append(my_emoji)

    df = df[~np.array(is_corrupt)]
    df['tokens'] = collect_tweets
    df['pos'] = collect_pos
    df['adj'] = collect_pos_adj
    df['emoji'] = collect_emoji
    df['unique_emoji'] = unique_tokens(df.emoji) 
    return df

