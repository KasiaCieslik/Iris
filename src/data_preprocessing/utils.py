from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import pandas as pd

# categorical 
### binary string to binary number
class string_to_binary(BaseEstimator,TransformerMixin):
    def __init__(self):
        pass 
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        for columns in X.columns:
            if X[columns].dtype=='object' and len(X[columns].unique())==2:
                X[columns] = pd.factorize(X[columns])[0]     
        return X
### object for a date to datetime and extra feature
obj_date_to_datetime = ['ScheduledDay','AppointmentDay']
class object_to_datatime(BaseEstimator,TransformerMixin):
    def __init__(self,obj_date_to_time):
        self.obj_date_to_time = obj_date_to_time
        pass
    def fit(self,X,y=None):
        return self
    def transform(self,X,Y=None):
        if self.obj_date_to_time:
            for columns in self.obj_date_to_time:
                X[columns] = X[columns].apply(pd.to_datetime).dt.normalize()
            X['DaysBetween'] = (X['AppointmentDay'] - X['ScheduledDay']).apply(lambda x:x.days) 
            return X
        else:
            return X

class dataframe_selector(BaseEstimator,TransformerMixin):
    def __init__(self,attributes_name):
        self.attributes_name = attributes_name
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        for columns in self.attributes_name:
            return X[columns].values
        
class get_dummies(BaseEstimator,TransformerMixin):
    def __init__(self):
        pass
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        return pd.get_dummies(X)
    
#class for numerical pipeline
class smaller_age_range(BaseEstimator,TransformerMixin):
    #age_range = ['Age']
    def __init__(self,age_range):
        self.age_range = age_range
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        for col in self.age_range:
            X[col][X[col]<0]=0
            X[col][X[col]>95]=95
        return X

class drop_NaN(BaseEstimator,TransformerMixin):
    def __init__(self):
        pass
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        return pd.DataFrame(X).dropna()
    
class drop_datetime(BaseEstimator,TransformerMixin):
    def __init__(self,drop_date_time):
        self.drop_date_time = drop_date_time
        pass
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        for columns in self.drop_date_time:
            X = X.drop(columns,axis=1)
        return X

def preprocess_categorical_to_binary(df):
    #categorical pipeline
    drop_date_time =  ['ScheduledDay','AppointmentDay']  
    obj_date_to_dt=['ScheduledDay','AppointmentDay']
    cat_pipeline = Pipeline([
    ('StringToBinary',string_to_binary()),
    ('ObjectToDataTime',object_to_datatime(obj_date_to_dt)),
    ('DropDateTime',drop_datetime(drop_date_time)),
    #('GetDummies',get_dummies())
])
    df = cat_pipeline.transform(df)
    return df

def change_age_range(df):
    #numerical pipeline
    age_range = ['Age']
    pipeline_num = Pipeline([
        ('smaller_age_range',smaller_age_range(age_range))
    ])
    df = pipeline_num.transform(df)
    return df

