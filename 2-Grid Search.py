#!/usr/bin/env python
# coding: utf-8

# # Grid Search 1

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

# remove warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


# In[ ]:


import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix,accuracy_score, roc_auc_score,f1_score, recall_score, precision_score
from sklearn.utils import class_weight

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, BaggingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF
from sklearn.svm import LinearSVC

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import RFECV, VarianceThreshold, SelectKBest, chi2
from sklearn.feature_selection import SelectFromModel, SelectPercentile, f_classif

import feather


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# In[ ]:


# define output variables
outVar = 'Class'
seed = 42
np.random.seed(seed)


# ### ML with 1 split

# In[ ]:


# read tr and ts datasets
df_tr_std = feather.read_dataframe(r'datasets\ds.Class.std.tr.feather')
df_ts_std = feather.read_dataframe(r'datasets\ds.Class.std.ts.feather')


# In[ ]:


df_tr_std.shape


# In[ ]:


df_ts_std.shape


# In[ ]:


# get data for tr and ts
X_tr_std = df_tr_std.drop(outVar, axis=1).values
y_tr_std = df_tr_std[outVar].values
X_ts_std = df_ts_std.drop(outVar, axis=1).values
y_ts_std = df_ts_std[outVar].values


# In[ ]:


params = {
    'max_samples'  : [0.1, 0.5, 1.0],
    'n_estimators' : [5, 10, 20, 50]
}


# In[ ]:


cls = BaggingClassifier(base_estimator=DecisionTreeClassifier(), random_state=seed)
gs = GridSearchCV(estimator=cls,
                   param_grid=params, n_jobs=-1, verbose=10, scoring ='roc_auc', cv=3)

gs.fit(X_tr_std, y_tr_std)


# In[ ]:


best_params = gs.best_params_
print(best_params)


# In[ ]:


print("Best parameters set found on development set:")
print()
print(gs.best_params_)
print()
print("Grid scores on development set:")
print()
means = gs.cv_results_['mean_test_score']
stds = gs.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, gs.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))


# In[ ]:


print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
y_true, y_pred = y_ts_std, gs.predict(X_ts_std)
cls_rep = classification_report(y_true, y_pred,target_names=['0','1'],
                               output_dict=True, digits=3)
print(classification_report(y_true, y_pred))
print()

y_probs = gs.predict_proba(X_ts_std)[:, 1]
ACC       = accuracy_score(y_ts_std, y_pred)
AUROC     = roc_auc_score(y_ts_std, y_probs)
precision = cls_rep['weighted avg']['precision']
recall    = cls_rep['weighted avg']['recall']
f1score   = cls_rep['weighted avg']['f1-score']

print('ACC       = {0:0.3f}'.format(ACC))
print('AUROC     = {0:0.3f}'.format(AUROC))
print('precision = {0:0.3f}'.format(precision))
print('recall    = {0:0.3f}'.format(recall))
print('f1score   = {0:0.3f}'.format(f1score))


# In[ ]:


gsResults_df = pd.DataFrame(columns=['Best Grid Search', 'ACC','AUROC' ,'precision' ,'recall' ,'f1-score' ])
gsResults_df


# In[ ]:


gsResults_df = gsResults_df.append({'Best Grid Search': str(gs.best_params_),
                          'ACC': float(ACC),
                          'AUROC': float(AUROC),
                          'precision': float(precision),
                          'recall': float(recall),
                          'f1-score': float(f1score)}, ignore_index=True)
gsResults_df


# ### GS using function

# In[ ]:


def myGridSearch(gs, params, df_tr_std, df_ts_std):
    # get data for tr and ts
    X_tr_std = df_tr_std.drop(outVar, axis=1).values
    y_tr_std = df_tr_std[outVar].values
    X_ts_std = df_ts_std.drop(outVar, axis=1).values
    y_ts_std = df_ts_std[outVar].values
    
    gs.fit(X_tr_std, y_tr_std)
    
    print("Best parameters set found on development set:")
    print()
    print(gs.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = gs.cv_results_['mean_test_score']
    stds = gs.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, gs.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    
    print("Detailed classification report:\n")
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.\n")
    y_true, y_pred = y_ts_std, gs.predict(X_ts_std)
    cls_rep = classification_report(y_true, y_pred,target_names=['0','1'],
                                   output_dict=True, digits=3)
    print(classification_report(y_true, y_pred))

    y_probs = gs.predict_proba(X_ts_std)[:, 1]
    ACC       = accuracy_score(y_ts_std, y_pred)
    AUROC     = roc_auc_score(y_ts_std, y_probs)
    precision = cls_rep['weighted avg']['precision']
    recall    = cls_rep['weighted avg']['recall']
    f1score   = cls_rep['weighted avg']['f1-score']

    print('ACC       = {0:0.3f}'.format(ACC))
    print('AUROC     = {0:0.3f}'.format(AUROC))
    print('precision = {0:0.3f}'.format(precision))
    print('recall    = {0:0.3f}'.format(recall))
    print('f1score   = {0:0.3f}'.format(f1score))

    return gs, ACC, AUROC, precision, recall, f1score


# In[ ]:


# read tr and ts datasets
df_tr_std = feather.read_dataframe(r'datasets\ds.Class.std.tr.feather')
df_ts_std = feather.read_dataframe(r'datasets\ds.Class.std.ts.feather')


# In[ ]:


params = {
    'max_samples'  : [0.4, 0.5, 0.6, 1.0],
    'n_estimators' : [50, 100, 500]
    #'base_estimator__max_depth' : [2, 4, None],
    #'base_estimator__max_leaf_nodes' : [10,20,None]
}
cls = BaggingClassifier(base_estimator=DecisionTreeClassifier(), random_state=seed)
gs = GridSearchCV(estimator=cls,
                   param_grid=params, n_jobs=-1, verbose=10, scoring ='roc_auc', cv=3)


# In[ ]:


gs, ACC, AUROC, precision, recall, f1score = myGridSearch(gs, params, df_tr_std, df_ts_std)


# In[ ]:


gsResults_df = gsResults_df.append({'Best Grid Search': str(gs.best_params_),
                          'ACC': float(ACC),
                          'AUROC': float(AUROC),
                          'precision': float(precision),
                          'recall': float(recall),
                          'f1-score': float(f1score)}, ignore_index=True)
gsResults_df


# In[ ]:


gsResults_df.to_csv(r'results\gs_1ML.csv')


# ### Feature importance

# In[ ]:


# gs.best_estimator_
feature_importances = np.mean([
    tree.feature_importances_ for tree in gs.best_estimator_.estimators_
], axis=0)


# In[ ]:


len(feature_importances)


# ### Baseline

# In[ ]:


def ML_baseline(cls, X_tr, y_tr, X_ts, y_ts, seed=42, classes=['0','1']):
    ACC = 0
    AUROC = 0
    precision = 0 
    recall = 0
    f1score = 0
    
    cls_name = type(cls).__name__
    
    start_time = time.time()
    cls.fit(X_tr, y_tr)
    print('>', cls_name, "training: %0.2f mins " % ((time.time() - start_time)/60))
    
    # predictions
    y_pred  = cls.predict(X_ts)
    y_probs = cls.predict_proba(X_ts)[:, 1]
    cls_rep = classification_report(y_ts, y_pred, target_names=classes,
                                    output_dict=True, digits=3)
    print(cls_rep)
    
    ACC       = accuracy_score(y_ts, y_pred)
    AUROC     = roc_auc_score(y_ts, y_probs)
    precision = cls_rep['weighted avg']['precision']
    recall    = cls_rep['weighted avg']['recall']
    f1score   = cls_rep['weighted avg']['f1-score']  
    
    return ACC, AUROC, precision, recall, f1score


# In[ ]:


# read tr and ts datasets
df_tr_std = feather.read_dataframe(r'datasets\ds.Class.std.tr.feather')
df_ts_std = feather.read_dataframe(r'datasets\ds.Class.std.ts.feather')

# get data for tr and ts
X_tr_std = df_tr_std.drop(outVar, axis=1).values
y_tr_std = df_tr_std[outVar].values
X_ts_std = df_ts_std.drop(outVar, axis=1).values
y_ts_std = df_ts_std[outVar].values


# In[ ]:


# define the classifiers for baseline
classifiers = [
               BaggingClassifier(random_state=seed, n_estimators= 5, max_samples=0.5),
               BaggingClassifier(random_state=seed, n_estimators=10, max_samples=0.5),
               BaggingClassifier(random_state=seed, n_estimators=20, max_samples=0.5),
               BaggingClassifier(random_state=seed, n_estimators= 5, max_samples=0.6),
               BaggingClassifier(random_state=seed, n_estimators=10, max_samples=0.6),
               BaggingClassifier(random_state=seed, n_estimators=20, max_samples=0.6),
               BaggingClassifier(random_state=seed, n_estimators= 5, max_samples=1.0),
               BaggingClassifier(random_state=seed, n_estimators=10, max_samples=1.0),
               BaggingClassifier(random_state=seed, n_estimators=20, max_samples=1.0)
               ]


# In[ ]:


# define the classifiers for baseline
classifiers = [
               BaggingClassifier(random_state=seed, n_estimators=5),
               BaggingClassifier(random_state=seed, n_estimators=4),
               BaggingClassifier(random_state=seed, n_estimators=3),
               BaggingClassifier(random_state=seed, n_estimators=2)
               ]


# In[ ]:


# create a dataframe for ML baseline
df_ML = pd.DataFrame(columns=['Method', 'ACC','AUROC' ,'precision' ,'recall' ,'f1-score' ])
df_ML


# In[ ]:


# fit each classifier
for cls in classifiers:
    print("\n***", cls)
    ACC,AUROC,precision,recall,f1score=ML_baseline(cls, X_tr_std, y_tr_std, X_ts_std, y_ts_std)
    df_ML = df_ML.append({'Method': str(type(cls).__name__),
                          'ACC': float(ACC),
                          'AUROC': float(AUROC),
                          'precision': float(precision),
                          'recall': float(recall),
                          'f1-score': float(f1score)}, ignore_index=True)

df_ML


# In[ ]:


df_ML.to_csv(r'results\baseline_Bagging3.csv')


# In[ ]:


cls.n_estimators

