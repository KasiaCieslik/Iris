from datetime import datetime

def save_tweets(df,path,file_name):
    """function to save data frame with filtered tweets
    File name contains currect date"""
    date = str(datetime.now().strftime("%d.%m.%Y %H:%M"))
    return df.to_csv(path + "/" + file_name + date +".csv")

