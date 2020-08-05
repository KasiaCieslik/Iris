import sys
sys.path.append('/home/kasia/sentiment-analysis-of-tweets-using-emoticons')
import pandas as pd
from data_preparation.process_tweet_before_save import process_tweet
from data_retrieval.twitter_streamer_real_time import scrap_tweet_from_twitter
from data_retrieval.save_tweets_to_csv import save_tweets
path = '/home/kasia/sentiment-analysis-of-tweets-using-emoticons/access_key/access.json'

tweets = scrap_tweet_from_twitter(query = '*',language = 'en',creds_path = path, range_number = 1,tweets_number_for_every_range = 10)

d = []
df = tweets
for x in df:
    tweet_data = process_tweet(x)
    d.append(tweet_data)
d = pd.DataFrame.from_records(d)

path = ('../data/raw')
save_tweets(d,path,'raw_tweets ')