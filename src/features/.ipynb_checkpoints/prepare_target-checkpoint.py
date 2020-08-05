#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def add_polarity_scores(df,emoji_dict, emoji_column='unique_emoji'):
    """prepare list with polarity of emoji from 'unique_emoji' column which are in emoji_dict"""
    all_emoji_scores = []
    for emoji_group in  df[emoji_column] :
        tweet_emoji_score_list = []
        for emoji in emoji_group:
            if str(emoji) in emoji_dict:
                tweet_emoji_score_list.append(emoji_dict.get(str(emoji)))
            else:
                pass
        all_emoji_scores.append(tweet_emoji_score_list)
    df['polarity_for_unique_emoji'] = all_emoji_scores    
    return df

def sum_polarity(df):
    """return sum of polarity for unique emoji in every row"""
    df['sum_polarity_unique_emoji'] = df['polarity_for_unique_emoji'].map(sum)
    return df


def prepare_target(df):
    """take sum of polarity and use conditions to prepare the target"""
    sentiment_target = []
    for polarity in df['sum_polarity_unique_emoji']:
        if polarity <= -2:
            sentiment_target.append('negative')
        elif polarity >2:
            sentiment_target.append('positive')
        else:
            sentiment_target.append(0)
    df['sentiment_target'] = sentiment_target
    df = df[df['sentiment_target']!=0]
    return df

