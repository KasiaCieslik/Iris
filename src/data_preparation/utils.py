from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class StringToBinary(BaseEstimator,TransformerMixin):
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
class ObjectToDataTime(BaseEstimator,TransformerMixin):
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

class DataFrameSelector(BaseEstimator,TransformerMixin):
    def __init__(self,attributes_name):
        self.attributes_name = attributes_name
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        for columns in self.attributes_name:
            return X[columns].values
        
class Get_Dummies(BaseEstimator,TransformerMixin):
    def __init__(self):
        pass
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        return pd.get_dummies(X)
    
#class for numerical pipeline
class SmallerAgeRange(BaseEstimator,TransformerMixin):
    def __init__(self,age_range):
        self.age_range = age_range
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        for col in self.age_range:
            X[col][X[col]<0]=0
            X[col][X[col]>95]=95
        return X