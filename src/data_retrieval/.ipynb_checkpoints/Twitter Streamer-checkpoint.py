#!/usr/bin/env python
# coding: utf-8

# In[27]:


from twython import Twython
import json
import advertools as adv
import pandas as pd
import re
from open_json import open_json_file
from twython import TwythonStreamer
import csv
import os
import time
from http.client import IncompleteRead
from requests.exceptions import ChunkedEncodingError
from requests.packages.urllib3.exceptions import ProtocolError
from twython.exceptions import TwythonError


# In[28]:


# function, which chooses important information from saved tweets
def process_tweet(tweet):
    d = {}
    d['user_loc'] = tweet['user']['location']
    d['hashtags'] = [hashtag['text'] for hashtag in tweet['entities']['hashtags']]
    d['text'] = tweet['text']
    d['user'] = tweet['user']['screen_name']
    d['user_loc'] = tweet['user']['location']
    return d
    

#class where we use TwythonStreamer to save whole tweets, data will be sav
class MyStreamer(TwythonStreamer): 
    DATA = []
    def save_to_object(self, tweet):
        self.DATA.append(tweet)
        
    def on_success(self, data):
        
        if len(self.DATA)%100==0:
            print(len(self.DATA))
            time.sleep(10)#           
        if len(self.DATA)>2000:# how many tweeter we want to saved
            self.disconnect()
            raise ValueError
            print('ValueError- Length DATA bigger than 1000')

        # Only collect tweets in English
        if data['lang'] == 'en':
            #tweet_data = process_tweet(data)
            self.save_to_object(data)
            #print(len(self.DATA))

    # Problem with the API
    def on_error(self, *args):
        print(args)
        self.disconnect()


# In[29]:


creds = open_json_file('twitter_credentials.json')
stream = MyStreamer(creds['CONSUMER_KEY'], creds['CONSUMER_SECRET'], 
                    creds['ACCESS_TOKEN'], creds['ACCESS_SECRET'])
while True:
    try:
        stream.statuses.filter(track='COVID19')
    except (IncompleteRead, ChunkedEncodingError):
        continue
    except ProtocolError:
        continue
    except ValueError:
        break


# In[ ]:


#choose only important info from tweet
d = []
df = stream.DATA
for x in df:
    tweet_data = process_tweet(x)
    d.append(tweet_data)
    


# In[ ]:


#choose 2 country
d = pd.DataFrame.from_records(d)
d[(d['user_loc']=='England') | (d['user_loc']=='USA')]

