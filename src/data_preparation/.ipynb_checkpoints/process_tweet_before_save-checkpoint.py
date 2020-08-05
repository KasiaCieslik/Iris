def process_tweet(tweet):
    """
    Take only user defined informations from tweets
    Parameters
        ----------
        tweet- dict
        
    Returns
         ----------
        dict
        salected info from tweet
    """
   
    d = {}
    d['user_loc'] = tweet['user']['location']
    d['hashtags'] = [hashtag['text'] for hashtag in tweet['entities']['hashtags']]
    d['text'] = tweet['text']
    d['retweet_count'] = tweet['retweet_count']
    #d['user'] = tweet['user']['screen_name']
    d['user_loc'] = tweet['user']['location']
    return d

