import pandas as pd
import pickle

def read_csv(path):
    """
    Load data from csv to df
    Parameters
        ----------
        path: str
        
    Returns
         ----------
        pd.DataFrame
    """
    df = pd.read_csv(path)
    df = pd.DataFrame(df['text'])
    return df

def save_pickle(matrix, path,filename):
    """
    Save data to pickle
    Parameters
        ----------
        matrix: scipy.sparse.csr.csr_matrix ('float64') or numpy.ndarray (int64)
        path: str
        filename: str
        
    Returns
         ----------
        pd.DataFrame
    """
    filename = path+filename
    with open(filename, 'wb') as outfile:
        pickle.dump(matrix, outfile, pickle.HIGHEST_PROTOCOL)
        
def load_pickle(filename,path): 
        """
    Open pickle file
    Parameters
        ----------
        path: str
        filename: str
        
    Returns
         ----------
        scipy.sparse.csr.csr_matrix ('float64') or numpy.ndarray ('int64')
    """
    filename = path+filename
    with open(filename, 'rb') as infile:
        matrix = pickle.load(infile)    
    return matrix  


