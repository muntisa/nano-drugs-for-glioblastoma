#!/usr/bin/env python
# coding: utf-8

# # Grid search 3

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
    
    return cls, ACC, AUROC, precision, recall, f1score


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


# ### Variation of estimators

# In[ ]:


# define the classifiers for baseline
classifiers = [
               BaggingClassifier(random_state=seed, n_estimators=20, max_samples=0.1),
               BaggingClassifier(random_state=seed, n_estimators=20, max_samples=0.2),
               BaggingClassifier(random_state=seed, n_estimators=20, max_samples=0.3),
               BaggingClassifier(random_state=seed, n_estimators=20, max_samples=0.4),
               BaggingClassifier(random_state=seed, n_estimators=20, max_samples=0.5),
               BaggingClassifier(random_state=seed, n_estimators=20, max_samples=0.6),
               BaggingClassifier(random_state=seed, n_estimators=20, max_samples=0.7),
               BaggingClassifier(random_state=seed, n_estimators=20, max_samples=0.8),
               BaggingClassifier(random_state=seed, n_estimators=20, max_samples=0.9),
               BaggingClassifier(random_state=seed, n_estimators=20, max_samples=1.0)
               ]


# In[ ]:


# create a dataframe for ML baseline
df_ML = pd.DataFrame(columns=['Method', 'n_estimators', 'max_samples', 'ACC','AUROC' ,'precision' ,'recall' ,'f1-score' ])
df_ML


# In[ ]:


# fit each classifier
for cls in classifiers:
    print("\n***", cls)
    cls_fit, ACC,AUROC,precision,recall,f1score=ML_baseline(cls, X_tr_std, y_tr_std, X_ts_std, y_ts_std)
    df_ML = df_ML.append({'Method': str(type(cls).__name__),
                          'n_estimators': cls_fit.n_estimators,
                          'max_samples': cls_fit.max_samples,
                          'ACC': float(ACC),
                          'AUROC': float(AUROC),
                          'precision': float(precision),
                          'recall': float(recall),
                          'f1-score': float(f1score)}, ignore_index=True)

df_ML


# In[ ]:


df_ML.to_csv(r'results\gs_Bagging_max_samples.csv')


# @muntisa

# In[ ]:




