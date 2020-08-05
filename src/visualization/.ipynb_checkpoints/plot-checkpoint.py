import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud
import ast


def polarity_distribution(df, file_name,title):
    """
    Plot distribution of polarity
    Parameters
    ----------
    df: pd.DataFrame
    file_name: str
    title: str

    Returns
    -------
    None
    """
    save_path = '../reports/figures/'
    if 'sentiment_target' in df.columns: 
        plt.figure(figsize=(20,15))
        df_pos = df['sum_polarity_unique_emoji'][df['sentiment_target']=='positive']
        df_neg = df['sum_polarity_unique_emoji'][df['sentiment_target']=='negative']
        plt.hist(df_pos,bins = 20,label='positive')
        plt.hist(df_neg,bins = 20,label='negative')
        plt.title(title,fontsize=25)
        plt.xlabel('Sum of polarity',fontsize=15)
        plt.ylabel('Counts',fontsize=15)
        plt.legend()
    else:
        plt.figure(figsize=(20,15))
        plt.hist(df['sum_polarity_unique_emoji'],bins=40)
        plt.title(title,fontsize=25)
        plt.xlabel('Sum of polarity',fontsize=15)
        plt.ylabel('Counts',fontsize=15)
    plt.savefig(save_path+file_name)
    
def target_distribution(df,file_name,title):
    """
    Plot target distribution for classes
    Parameters
    ----------
    df: pd.DataFrame
    file_name: str
    title: str

    Returns
    -------
    None
    """
    save_path = '../reports/figures/'
    plt.figure(figsize=(15,10))
    counts = df['sentiment_target'].value_counts()
    plt.bar(counts.index, counts.values)
    plt.title(title,fontsize=25)
    plt.xlabel('Target',fontsize=15)
    plt.ylabel('Counts',fontsize=15)
    plt.savefig(save_path+file_name)

def word_cloud(df,emotion):
    """
    plot word cloud for the most important 100 words
    Parameters
    ----------
    df: pd.DataFrame
    emotion: str

    Returns
    -------

    """
    save_path = '../reports/figures/'
    connect_word_in_tweet = []
    df_neg = df[df['sentiment_target']==emotion]
    for case in df_neg['tokens']:
        x = ast.literal_eval(case)
        x = ' '.join(x)
        connect_word_in_tweet.append(x)
        text = ' '.join(connect_word_in_tweet)
        text = text.replace('-pron-','')
    wordcloud = WordCloud(max_words=100,width=1920, height=1080).generate(text)
    plt.figure( figsize=(20,10) )
    plt.title(emotion)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.savefig(save_path + 'wordcloud_' + emotion + '.png', facecolor='k', bbox_inches='tight')