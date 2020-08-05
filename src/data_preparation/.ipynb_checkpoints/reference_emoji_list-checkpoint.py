import pandas as pd
from src.features.spacy_helpers import setup_spacy

def prepare_reference_emoji_list(fp='data/raw/emoji_list.csv', nlp=None):
    """preapre df with emoji in token form"""
    emoji_list = pd.read_csv(fp, index_col=0)
    emoji_token = [] 
    for x in emoji_list['symbol']:
        sym = nlp(x)
        emoji_token.append([word for word in sym if word._.is_emoji][0])
    emoji_list['symbol'] = emoji_token

    emoji_sym = list(emoji_list['symbol'])
    emoji_sym = [str(e) for e in emoji_sym]
    emoji_polarity= list(emoji_list['polarity'])
    emoji_dict = dict(zip(emoji_sym,emoji_polarity))
    
    return emoji_dict

