import pandas as pd
from numpy import mean
from numpy import std
from numpy import absolute
from pandas import read_csv
import numpy as np
from numpy import arange
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.datasets import load_iris
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import ElasticNetCV
import time
import sqlite3
from sqlalchemy import create_engine
# "AU01_r","AU02_r","AU04_r","AU05_r","AU06_r","AU07_r","AU09_r","AU10_r","AU12_r","AU14_r","AU15_r","AU17_r","AU20_r","AU23_r","AU25_r","AU26_r","AU45_r"

file = 'C:/Users/Leon/Downloads/datazipper/DataMutated.xlsx'
output = 'output.xlsx'

engine = create_engine('sqlite://', echo=False)
df = pd.read_excel(file)
df.to_sql('AUs', engine, if_exists='replace', index=False)

results = engine.execute("Select * from AUs")

final=pd.DataFrame(results, columns=df.columns)
final.to_excel(output, index=False)


def LogWith5():
    print('Log predict score with AUs data')
    x = engine.execute("SELECT AU17_r, fear, AU25_r, AU14_r, AU10_r FROM {AUs} LIMIT 1000")
    y = engine.execute("SELECT peak FROM AUs")
    
    x_df = pd.DataFrame(x, columns=x._metadata.keys)
    y_df = pd.DataFrame(y, columns=y._metadata.keys)
    
    # Convert to numpy array
    X = np.array(x_df).astype('float32')
    y = np.array(y_df).astype(int)

    y = y.ravel()

    scaler = StandardScaler() 
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {'l1_ratio':[.1, .5, .7, .9, .95, .99]}

    clf = GridSearchCV(LogisticRegression(penalty = 'elasticnet', random_state = 2, solver = 'saga', max_iter = 10000), param_grid)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.best_params_)
    print(accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
LogWith5()

def LogWith10():
    print('Log predict score with AUs data')
    x = engine.execute("SELECT AU17_r, fear, AU25_r, AU14_r, AU10_r, AU26_r, digust, AU23_r, AU20_r, anger FROM AUs")
    y = engine.execute("SELECT peak FROM AUs")
    
    x_df = pd.DataFrame(x, columns=x._metadata.keys)
    y_df = pd.DataFrame(y, columns=y._metadata.keys)
    
    # Convert to numpy array
    X = np.array(x_df).astype('float32')
    y = np.array(y_df).astype(int)

    y = y.ravel()

    scaler = StandardScaler() 
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {'l1_ratio':[.1, .5, .7, .9, .95, .99]}

    clf = GridSearchCV(LogisticRegression(penalty = 'elasticnet', random_state = 2, solver = 'saga', max_iter = 10000), param_grid)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.best_params_)
    print(accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
LogWith10()

def LogWithAll():
    print('Log predict score with AUs data')
    x = engine.execute("SELECT AU17_r, fear, AU25_r, AU14_r,  AU10_r, AU26_r, digust, AU23_r, AU20_r, anger, AU15_r, AU12_r, sad, AU12_r, happy, AU09_r, AU01_r, AU02_r, AU06_r, condfidence, timestamp, AU04_r, AU45_r, AU07_r, AU05_r FROM AUs")
    y = engine.execute("SELECT peak FROM AUs")
    
    x_df = pd.DataFrame(x, columns=x._metadata.keys)
    y_df = pd.DataFrame(y, columns=y._metadata.keys)
    
    # Convert to numpy array
    X = np.array(x_df).astype('float32')
    y = np.array(y_df).astype(int)

    y = y.ravel()

    scaler = StandardScaler() 
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {'l1_ratio':[.1, .5, .7, .9, .95, .99]}

    clf = GridSearchCV(LogisticRegression(penalty = 'elasticnet', random_state = 2, solver = 'saga', max_iter = 10000), param_grid)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.best_params_)
    print(accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
LogWithAll()

################# KNN

import tensorflow as tf
import Classes
import Database
import numpy as np
from math import sqrt
from tensorflow.keras.models import Sequential
from sklearn.metrics import mean_squared_error
from tensorflow.keras.layers import Dense, Dropout, LSTM
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel,WhiteKernel
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn import metrics
from keras.layers import Layer
from keras import backend as K
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.losses import binary_crossentropy
from keras.initializers import RandomUniform, Initializer, Constant
from numpy import shape
from pyGRNN import GRNN
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import accuracy_score
from pyGRNN import GRNN
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import time
from sklearn.metrics import classification_report

###############################

def KNN5():
    print('KNN predict score with AUs data')
    x = engine.execute("SELECT AU17_r, fear, AU25_r, AU14_r, AU10_r FROM AUs")
    y = engine.execute("SELECT peak FROM AUs")
    
    x_df = pd.DataFrame(x, columns=x._metadata.keys)
    y_df = pd.DataFrame(y, columns=y._metadata.keys)
    
    # Convert to numpy array
    X = np.array(x_df).astype('float32')
    y = np.array(y_df).astype(int)
    
    y = y.ravel()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    parameters = {"n_neighbors": [1,3,5,10,15], "p": [1,2,3]}
    gridsearch = GridSearchCV(KNeighborsClassifier(), parameters)
    gridsearch.fit(X_train, y_train)
    y_pred = gridsearch.predict(X_test)
    print(gridsearch.best_params_)
    print(accuracy_score(y_test, y_pred))
        
    print(classification_report(y_test, y_pred))
    
KNN5()

def KNN10():
    print('KNN predict score with AUs data')
    x = engine.execute("SELECT AU17_r, fear, AU25_r, AU14_r, AU10_r, AU26_r, digust, AU23_r, AU20_r, anger FROM AUs")
    y = engine.execute("SELECT peak FROM AUs")
    
    x_df = pd.DataFrame(x, columns=x._metadata.keys)
    y_df = pd.DataFrame(y, columns=y._metadata.keys)
    
    # Convert to numpy array
    X = np.array(x_df).astype('float32')
    y = np.array(y_df).astype(int)
    
    y = y.ravel()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    parameters = {"n_neighbors": [1,3,5,10,15], "p": [1,2,3]}
    gridsearch = GridSearchCV(KNeighborsClassifier(), parameters)
    gridsearch.fit(X_train, y_train)
    y_pred = gridsearch.predict(X_test)
    print(gridsearch.best_params_)
    print(accuracy_score(y_test, y_pred))
        
    print(classification_report(y_test, y_pred))
    
KNN10()


def KNNAll():
    print('KNN predict score with AUs data')
    x = engine.execute("SELECT AU17_r, fear, AU25_r, AU14_r, AU10_r, AU26_r, digust, AU23_r, AU20_r, anger, AU15_r, AU12_r, sad, AU12_r, happy, AU09_r, AU01_r, AU02_r, AU06_r, condfidence, timestamp, AU04_r, AU45_r, AU07_r, AU05_r FROM AUs")
    y = engine.execute("SELECT peak FROM AUs")
    
    x_df = pd.DataFrame(x, columns=x._metadata.keys)
    y_df = pd.DataFrame(y, columns=y._metadata.keys)
    
    # Convert to numpy array
    X = np.array(x_df).astype('float32')
    y = np.array(y_df).astype(int)
    
    y = y.ravel()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    parameters = {"n_neighbors": [1,3,5,10,15], "p": [1,2,3]}
    gridsearch = GridSearchCV(KNeighborsClassifier(), parameters)
    gridsearch.fit(X_train, y_train)
    y_pred = gridsearch.predict(X_test)
    print(gridsearch.best_params_)
    print(accuracy_score(y_test, y_pred))
        
    print(classification_report(y_test, y_pred))
    
KNNAll()

from numpy import mean
from numpy import std
from numpy import absolute
from pandas import read_csv
import Database
import numpy as np
from numpy import arange
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.datasets import load_iris
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import ElasticNetCV
import time
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from matplotlib import pyplot as plt

def RFAll():
    time_start = time.time()
    
    print('RF all features')
    x = engine.execute("SELECT AU17_r, fear, AU25_r, AU14_r, AU10_r, AU26_r, digust, AU23_r, AU20_r, anger, AU15_r, AU12_r, sad, AU12_r, happy, AU09_r, AU01_r, AU02_r, AU06_r, condfidence, timestamp, AU04_r, AU45_r, AU07_r, AU05_r FROM AUs")
    y = engine.execute("SELECT peak FROM AUs")
    
    x_df = pd.DataFrame(x, columns=x._metadata.keys)
    y_df = pd.DataFrame(y, columns=y._metadata.keys)
    
    # Convert to numpy array
    X = np.array(x_df).astype('float32')
    y = np.array(y_df).astype(int)

    y = y.ravel()

    scaler = StandardScaler() 
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = { 
    'n_estimators': [10,20,30,40,50,60,70,80,90],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8,9,10,11,12],
    'criterion' :['gini', 'entropy']}
    rfc=RandomForestClassifier(random_state=42)

    model = GridSearchCV(estimator=rfc, param_grid=param_grid)
    # Fit on training data
    model.fit(X_train, y_train)
    print(model.best_params_)
    model.best_estimator_.coef_
    importance = model.coef_
    for i,v in enumerate(importance):
	    print('Feature: %0d, Score: %.5f' % (i,v))
    y_pred = model.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    end = time.time()
    print(end-time_start)
    
RFAll()

def RF10():
    time_start = time.time()
    
    print('RF all features')
    x = engine.execute("SELECT AU17_r, fear, AU25_r, AU14_r, AU10_r, AU26_r, digust, AU23_r, AU20_r, anger FROM AUs")
    y = engine.execute("SELECT peak FROM AUs")
    
    x_df = pd.DataFrame(x, columns=x._metadata.keys)
    y_df = pd.DataFrame(y, columns=y._metadata.keys)
    
    # Convert to numpy array
    X = np.array(x_df).astype('float32')
    y = np.array(y_df).astype(int)

    y = y.ravel()

    scaler = StandardScaler() 
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = { 
    'n_estimators': [10,20,30,40,50,60,70,80,90],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8,9,10,11,12],
    'criterion' :['gini', 'entropy']}
    rfc=RandomForestClassifier(random_state=42)

    model = GridSearchCV(estimator=rfc, param_grid=param_grid)
    # Fit on training data
    model.fit(X_train, y_train)
    print(model.best_params_)
    model.best_estimator_.coef_
    importance = model.coef_
    for i,v in enumerate(importance):
	    print('Feature: %0d, Score: %.5f' % (i,v))
    y_pred = model.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    end = time.time()
    print(end-time_start)
    
RF10()

def RF5():
    time_start = time.time()
    
    print('RF all features')
    x = engine.execute("SELECT AU17_r, fear, AU25_r, AU14_r, AU10_r FROM AUs")
    y = engine.execute("SELECT peak FROM AUs")
    
    x_df = pd.DataFrame(x, columns=x._metadata.keys)
    y_df = pd.DataFrame(y, columns=y._metadata.keys)
    
    # Convert to numpy array
    X = np.array(x_df).astype('float32')
    y = np.array(y_df).astype(int)

    y = y.ravel()

    scaler = StandardScaler() 
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = { 
    'n_estimators': [10,20,30,40,50,60,70,80,90],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8,9,10,11,12],
    'criterion' :['gini', 'entropy']}
    rfc=RandomForestClassifier(random_state=42)

    model = GridSearchCV(estimator=rfc, param_grid=param_grid)
    # Fit on training data
    model.fit(X_train, y_train)
    print(model.best_params_)
    model.best_estimator_.coef_
    importance = model.coef_
    for i,v in enumerate(importance):
	    print('Feature: %0d, Score: %.5f' % (i,v))
    y_pred = model.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    end = time.time()
    print(end-time_start)
    
RF5()



import pandas as pd
from sqlalchemy import create_engine
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

file = 'C:/Users/Leon/Downloads/datazipper/DataMutated.xlsx'
output = 'output.xlsx'

engine = create_engine('sqlite://', echo=False)
df = pd.read_excel(file)
df.to_sql('AUs', engine, if_exists='replace', index=False)

results = engine.execute("Select * from AUs")

final=pd.DataFrame(results, columns=df.columns)
final.to_excel(output, index=False)


def SVM5():
    print('SVM predict score with 5 data')
    x = engine.execute("SELECT AU17_r, fear, AU25_r, AU14_r, AU10_r FROM AUs")
    y = engine.execute("SELECT peak FROM AUs")

    x_df = pd.DataFrame(x, columns=x._metadata.keys)
    y_df = pd.DataFrame(y, columns=y._metadata.keys)

    # Convert to numpy array
    X = np.array(x_df).astype('float32')
    y = np.array(y_df).astype(int)

    y = y.ravel()

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                  'kernel': ['rbf', 'linear', 'poly', 'sigmoid']}
    clf = GridSearchCV(SVC(), param_grid, refit=True)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.best_params_)
    print(accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))


SVM5()