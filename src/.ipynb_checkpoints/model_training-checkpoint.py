import sys
import numpy as np
import pandas as pd

sys.path.append('/home/kasia/sentiment-analysis-of-tweets-using-emoticons')

from models.model import random_search_best_estimator, final_model, save_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from data_preparation.read_data import load_pickle


path = '../data/processed/'
X_train = load_pickle('X_train.pickle',path)
X_test = load_pickle('X_test.pickle',path)
y_train = load_pickle('y_train.pickle',path)
y_test = load_pickle('y_test.pickle',path)

logistic = LogisticRegression()
knn = KNeighborsClassifier()
rfc = RandomForestClassifier()
sgd = SGDClassifier()

scorer = make_scorer(accuracy_score)

param_grid = [  # {'alpha': np.linspace(0.00001, 1, 40)},
    {'penalty': ['l2'], 'C': [0.1, 1, 5, 10], 'solver': ['lbfgs', 'liblinear']},
    {'n_neighbors': [1, 3, 5, 10]},
    {'n_estimators': list(range(10, 101, 10)), 'max_features': list(range(6, 32, 5))},
    {'average': [True, False], 'alpha': np.linspace(0.001, 1, 40)}]
model_list = [logistic, knn, rfc, sgd]

grid_search, grid_results, results = random_search_best_estimator(scorer, param_grid, model_list, X_train, X_test,
                                                                  y_train, y_test)
results = pd.DataFrame(results)
best_estimator = results['best_estimator'][results['best_score'] == results['best_score'].max()].iloc[0]
final_accuracy_test,final_accuracy_train,pred_test,pred_train = final_model(X_train, X_test, y_train, y_test, best_estimator)
save_model('../models', 'best_estimator.sav', best_estimator)