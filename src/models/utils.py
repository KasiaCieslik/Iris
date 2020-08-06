import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

def split_train_test_with_unique_patient(df):
    random.seed(42)
    patient_list = df.PatientId.unique()
    np.random.shuffle(patient_list) 
    
    size_of_test = round(len(patient_list)*0.25)
    size_of_train = round(len(patient_list)*0.75)
    patient_train = patient_list[size_of_test:]
    patient_test =  patient_list[:size_of_test]

    train = df[df['PatientId'].isin(patient_train)]
    test = df[df['PatientId'].isin(patient_test)]
    return train, test

def split_train_test_with_sklearn(df):  
    np.random.seed(40)
    train, test  = train_test_split(df,test_size=0.33,random_state=42)
    return train, test

def prepare_for_modeling(df):
    y = df['No-show']
    y = pd.factorize(y)[0]
    X = df.drop(['No-show','PatientId','AppointmentID','Neighbourhood'],axis=1)
    return X,y

def downsampling(df,downsampling_number):
# Separate majority and minority classes
# I will take whole dataset to do down-sampling. After that I will do train and test set
    majority = df[df['No-show']==0]
    minority = df[df['No-show']==1]
 
    # Downsample minority class
    majority_D = resample(majority, 
                                 replace=False,     # sample with replacement
                                 n_samples=downsampling_number,    # to match majority class
                                 random_state=42) # reproducible results
 
    # Combine majority class with upsampled minority class
    df = pd.concat([minority, majority_D])
    return df