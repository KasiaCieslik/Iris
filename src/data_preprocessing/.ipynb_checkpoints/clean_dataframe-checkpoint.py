def downsample_target(df):
    """return balanced dataframe"""
    min_target_number = df['sentiment_target'].value_counts().min()
    df = df.groupby('sentiment_target').apply(lambda x: x.sample(min_target_number, replace=False))
    df = df.reset_index(drop=True)#.drop(['level_0','index'],axis=1)
    return df

def remove_dup_tokens(tokens):
    """create two list first with all token and second only with unique token"""
    collect = []
    seen = []
    for token in tokens:
        if str(token) not in seen:
            collect.append(token)
        seen.append(str(token))
    return collect


def unique_tokens(emojis_column):    
    """remove duplication from columns with emoji"""
    unique_emoji = []
    for emoji in emojis_column:
        collect = remove_dup_tokens(emoji)
        unique_emoji.append(collect)
    return unique_emoji  

def additional_processing(df):
    """remove rows without emoji"""
    df = df[df['emoji'].map(len)!=0]
    return df

def downsample_target(df):
    """return balanced dataframe"""
    min_target_number = df['sentiment_target'].value_counts().min()
    df = df.groupby('sentiment_target').apply(lambda x: x.sample(min_target_number, replace=False))
    df = df.reset_index(drop=True)#.drop(['level_0','index'],axis=1)
    return df
