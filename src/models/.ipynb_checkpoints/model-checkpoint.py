from datetime import datetime
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import pickle 

def random_search_best_estimator(scorer,param_grid,model_list,X_train, X_test, y_train, y_test):
    """return best estimator"""
    results = []
    for c,model in enumerate(model_list):
        grid_search = RandomizedSearchCV(model,param_grid[c],cv=5,n_iter=40,scoring=scorer)
        grid_search.fit(X_train,y_train)
        grid_results = grid_search.cv_results_
        results.append({'model_name':str(model_list[c]).split('X_train(')[0],'best_param':grid_search.best_params_,'best_score':grid_search.best_score_,'best_estimator':grid_search.best_estimator_})
    return grid_search, grid_results,results

def final_model(X_train, X_test, y_train, y_test,best_estimator):
    "return model for best estimator"

    final_model = Pipeline([
    ('best_estimator', best_estimator,)
    ])

    # Fit on train
    final_model.fit(X_train,y_train)
    # Predict on test
    pred_test = final_model.predict(X_test)
    pred_train = final_model.predict(X_train)
    final_accuracy_train = accuracy_score(y_train,pred_train)
    final_accuracy_test = accuracy_score(y_test,pred_test)
    # Return
    return final_accuracy_test,final_accuracy_train,pred_test,pred_train

def save_model(path,file_name,best_estimator):
    date = str(datetime.now().strftime("%Y%m%d_%H:%M"))
    filename = path+'/'+ date+'_'+ file_name
    pickle.dump(best_estimator, open(filename, 'wb'))
