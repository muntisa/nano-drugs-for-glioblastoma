#!/usr/bin/env python
# coding: utf-8

# # Create subsets and ML baseline for anti-glioblastoma drug-NP pairs

# We will use one split of data, standardized dataset, dataset preprocessing, etc.

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


# In[ ]:


sFile = './datasets/ds.'+str(outVar)+'.feather'


# Read the dataset:

# In[ ]:


df = feather.read_dataframe(sFile)


# In[ ]:


df.shape


# Data split:

# In[ ]:


X_data = df.drop(outVar, axis=1).values
Y_data = df[outVar].values
# 75% training - 25% test
X_tr, X_ts, y_tr, y_ts = train_test_split(X_data, Y_data, random_state=seed, stratify=Y_data)


# In[ ]:


print("Splits:", X_tr.shape, X_ts.shape, y_tr.shape, y_ts.shape)


# ### Remove features with low variance
# Using VarianceThreshold(threshold=0.0001)

# In[ ]:


featFilter = VarianceThreshold(threshold=0.0001)


# In[ ]:


X_high_variance = featFilter.fit_transform(X_tr)


# In[ ]:


X_high_variance.shape


# In[ ]:


featFilter.get_support()


# In[ ]:


len(featFilter.get_support(indices = True))


# In[ ]:


selected_features = set(list(df.columns[featFilter.get_support(indices=True)]))


# In[ ]:


pool_features = set(list(df.columns)[:-1]) # no Class


# In[ ]:


elimintated_feats = list(pool_features-selected_features)[:-1]


# In[ ]:


print("Elimintated features:", elimintated_feats)


# In[ ]:


len(elimintated_feats)


# We eliminated 14 features. Do the same with TEST subset:

# In[ ]:


X_high_variance_ts = featFilter.transform(X_ts)


# In[ ]:


X_high_variance_ts.shape


# In[ ]:


X_high_variance.shape


# In[ ]:


df_selVar = df[df.columns[featFilter.get_support(indices=True)]] 


# Add Y:

# In[ ]:


df_selVar['Class'] = df.Class


# In[ ]:


df_selVar.shape


# In[ ]:


df_selVar.columns


# Save dataframe without features with low variance:

# In[ ]:


df_selVar.to_csv(r'datasets\ds.Class.csv')


# In[ ]:


df_selVar.shape


# ### Standardize dataset

# In[ ]:


X_data = df_selVar.drop(outVar, axis=1).values
Y_data = df_selVar[outVar].values
X_tr, X_ts, y_tr, y_ts = train_test_split(X_data, Y_data, random_state=seed, stratify=Y_data)
print("Splits:", X_tr.shape, X_ts.shape, y_tr.shape, y_ts.shape)


# In[ ]:


scaler = StandardScaler()


# In[ ]:


X_tr_std = scaler.fit_transform(X_tr)


# In[ ]:


X_ts_std = scaler.transform(X_ts)


# Save the splits as files:

# In[ ]:


df_tr_std = pd.DataFrame(X_tr_std, columns = list(df_selVar.columns)[:-1])


# In[ ]:


df_tr_std['Class'] = y_tr


# In[ ]:


df_tr_std.shape


# In[ ]:


df_ts_std = pd.DataFrame(X_ts_std, columns = list(df_selVar.columns)[:-1])
df_ts_std['Class'] = y_ts
df_ts_std.shape


# ### Save feather format

# In[ ]:


feather.write_dataframe(df_tr_std, r'datasets\ds.Class.std.tr.feather')


# In[ ]:


feather.write_dataframe(df_ts_std, r'datasets\ds.Class.std.ts.feather')


# In[ ]:


# Stack tr and ts DataFrames on top of each other
vertical_stack = pd.concat([df_tr_std, df_ts_std], axis=0)


# In[ ]:


vertical_stack.shape


# In[ ]:


print(list(vertical_stack.columns))


# In[ ]:


feather.write_dataframe(vertical_stack, r'datasets\ds.Class.std.feather')


# ### ML with training and test subsets

# In[ ]:


outVar='Class'
seed = 42
np.random.seed(seed)


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


# define the classifiers for baseline
classifiers = [KNeighborsClassifier(n_jobs=-1, n_neighbors=3),
               GaussianNB(),
               LinearDiscriminantAnalysis(solver='svd'),
               LogisticRegression(solver='lbfgs',random_state=seed, max_iter=2000),
               DecisionTreeClassifier(random_state = seed),
               DecisionTreeClassifier(criterion="gini", max_depth=10, min_samples_leaf=17, min_samples_split=19), # tpot
               RandomForestClassifier(n_estimators=100,n_jobs=-1,random_state=seed),
               XGBClassifier(n_estimators=100, n_jobs=-1,seed=seed),
               GradientBoostingClassifier(random_state=seed),
               BaggingClassifier(random_state=seed),
               AdaBoostClassifier(random_state = seed)]


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


df_ML.to_csv(r'results\baseline_ML.csv')


# **BaggingClassifier** shows the best results. We shall try grid search for different methods.

# In the next scripts, a grid search for the best hyperparamters for the best model will be used.
# 
# Hf with ML! @muntisa

# In[ ]:




