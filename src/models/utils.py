import random
import numpy as np

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
 