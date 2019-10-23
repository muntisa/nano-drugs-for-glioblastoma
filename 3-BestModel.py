#!/usr/bin/env python
# coding: utf-8

# # Best Model - Feature importance

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


from sklearn.externals import joblib


# In[ ]:


# define output variables
outVar = 'Class'
seed = 42
np.random.seed(seed)


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


X_tr_std.shape


# In[ ]:


# define the classifiers for baseline
classifiers = [
               BaggingClassifier(random_state=seed, n_estimators=20, max_samples=0.5)
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


# gs.best_estimator_
feature_importances = np.mean([
    tree.feature_importances_ for tree in cls_fit.estimators_
], axis=0)
print(len(feature_importances))


# In[ ]:


feat_imp_df = pd.DataFrame(columns = ['Feature','Importance'])
feat_imp_df


# In[ ]:


feat_imp_df['Feature']=list(df_tr_std.columns)[:-1]
feat_imp_df


# In[ ]:


feat_imp_df['Importance']=feature_importances
feat_imp_df


# In[ ]:


feat_imp_df.to_csv(r'results\bestModel_featImps.csv')


# In[ ]:


# save best model
joblib.dump(cls_fit, r'results\best.model')


# In[ ]:


feat_imp_df_ord = feat_imp_df.sort_values('Importance', ascending=False)
feat_imp_df_ord.head(20)


# In[ ]:


feat_imp_df_ord.tail(20)


# In[ ]:


feat_imp_df_ord.to_csv(r'results\bestModel_featImps_ordered.csv')


# ### Best features

# In[ ]:


feat_imp_df.sort_values('Importance', ascending=False).head(52).sort_values('Importance', ascending=True).plot(kind='barh',x='Feature',y='Importance', figsize=(10,15))
plt.show() # half from 104


# ### Worst features

# In[ ]:


feat_imp_df_ord.tail(52).plot(kind='barh',x='Feature',y='Importance', figsize=(10,15))
plt.show() # half from 104


# ### Feature importance

# In[ ]:


feat_imp_df.sort_values('Importance', ascending=True).plot(kind='barh',x='Feature',y='Importance', figsize=(10,30))
plt.show()


# Try remove some features to improve the model?¿

# In[ ]:


feat_imp_df_ord.tail(20).sort_values('Importance', ascending=True)


# In[ ]:


# features to remove
feats2Rem = list(feat_imp_df_ord.tail(20).sort_values('Importance', ascending=True)['Feature'])
print(feats2Rem)


# In[ ]:


# define the classifier
BestClassifier = BaggingClassifier(random_state=seed, n_estimators=20, max_samples=0.5, n_jobs=-1)


# In[ ]:


# create a dataframe for ML baseline
df_FeatsImpRem = pd.DataFrame(columns=['Removed Feature', 'ACC','AUROC' ,'precision' ,'recall' ,'f1-score' ])
df_FeatsImpRem


# In[ ]:


print("*** Feature elimination by importance")
print("\n***", BestClassifier)

# read tr and ts datasets
df_tr_std_orig = feather.read_dataframe(r'datasets\ds.Class.std.tr.feather')
df_ts_std_orig = feather.read_dataframe(r'datasets\ds.Class.std.ts.feather')
    
# fit each classifier
for nFeats in range(1,len(feats2Rem)+1):
    # remove a number of features
    remFeatures = feats2Rem[:nFeats] # removed features
    
    print("-> Removed features:", remFeatures)
    df_tr_std = df_tr_std_orig.drop(remFeatures, axis=1)
    df_ts_std = df_ts_std_orig.drop(remFeatures, axis=1)

    # get data for tr and ts
    X_tr_std = df_tr_std.drop(outVar, axis=1).values
    y_tr_std = df_tr_std[outVar].values
    X_ts_std = df_ts_std.drop(outVar, axis=1).values
    y_ts_std = df_ts_std[outVar].values
    
    cls_fit, ACC,AUROC,precision,recall,f1score=ML_baseline(cls, X_tr_std, y_tr_std, X_ts_std, y_ts_std)
    df_FeatsImpRem = df_FeatsImpRem.append({'Removed Feature': remFeatures,
                                            'ACC': float(ACC),
                                            'AUROC': float(AUROC),
                                            'precision': float(precision),
                                            'recall': float(recall),
                                            'f1-score': float(f1score)}, ignore_index=True)

df_FeatsImpRem


# In[ ]:


df_FeatsImpRem


# In[ ]:


df_FeatsImpRem.to_csv(r'results\bestModel_20featsRem.csv')


# ### Extra features to remove

# Normalize Importance between 0 and 1:

# In[ ]:


scaler = MinMaxScaler()


# In[ ]:


feat_imp_df_ord['Importance'] = scaler.fit_transform(feat_imp_df_ord['Importance'].values.reshape(-1, 1))
feat_imp_df_ord.head()


# In[ ]:


feat_imp_df_ord.to_csv(r'results\bestModel_bestModel_featImps_ordered_norm.csv')


# In[ ]:


feats2Refeats2Rem_less10percm_less10perc = feat_imp_df_ord[feat_imp_df_ord['Importance']<0.1]
len(feats2Refeats2Rem_less10percm_less10perc)


# In[ ]:


feats2Refeats2Rem_less10percm_less10perc


# In[ ]:


feats2Refeats2Rem_less10percm_less10perc = feats2Refeats2Rem_less10percm_less10perc.sort_values('Importance', ascending=True)
feats2Refeats2Rem_less10percm_less10perc


# In[ ]:


# features to remove with importance less than 10%
feats2Rem2 = list(feats2Refeats2Rem_less10percm_less10perc['Feature'])
print(len(feats2Rem2))


# ### Eliminate more features:

# In[ ]:


print("*** Feature elimination by importance")
print("\n***", BestClassifier)

# read tr and ts datasets
df_tr_std_orig = feather.read_dataframe(r'datasets\ds.Class.std.tr.feather')
df_ts_std_orig = feather.read_dataframe(r'datasets\ds.Class.std.ts.feather')
    
# fit each classifier
for nFeats in range(35,len(feats2Rem2)+1):
    # remove a number of features
    remFeatures = feats2Rem2[:nFeats] # removed features
    
    print("-> Removed features:", remFeatures)
    df_tr_std = df_tr_std_orig.drop(remFeatures, axis=1)
    df_ts_std = df_ts_std_orig.drop(remFeatures, axis=1)

    # get data for tr and ts
    X_tr_std = df_tr_std.drop(outVar, axis=1).values
    y_tr_std = df_tr_std[outVar].values
    X_ts_std = df_ts_std.drop(outVar, axis=1).values
    y_ts_std = df_ts_std[outVar].values
    
    cls_fit, ACC,AUROC,precision,recall,f1score=ML_baseline(cls, X_tr_std, y_tr_std, X_ts_std, y_ts_std)
    df_FeatsImpRem = df_FeatsImpRem.append({'Removed Feature': remFeatures,
                                            'ACC': float(ACC),
                                            'AUROC': float(AUROC),
                                            'precision': float(precision),
                                            'recall': float(recall),
                                            'f1-score': float(f1score)}, ignore_index=True)

df_FeatsImpRem


# In[ ]:


df_FeatsImpRem.to_csv(r'results\bestModel_72featsRem_less10perc.csv')


# In[ ]:


df_BoxPlot = df_FeatsImpRem[['ACC', 'AUROC', 'precision', 'recall', 'f1-score']]
df_BoxPlot


# In[ ]:


plt.boxplot(df_BoxPlot['ACC'])
plt.show()


# In[ ]:


plt.boxplot(df_BoxPlot['AUROC'])
plt.show()


# In[ ]:


df_FeatsImpRem.columns


# In[ ]:


print(list(df_tr_std.columns))


# In[ ]:


len(list(df_tr_std.columns))


# ### Final model

# We chose 64 feature to eliminate (point 63): 'np_DNMUnp(c5)', 'np_DPDIcoat(c3)', 'np_DPDIcoat(c0)', 'np_DPDIcoat(c1)', 'np_DPDIcoat(c2)', 'np_DALOGPcoat(c0)', 'np_DALOGPcoat(c2)', 'np_DHycoat(c0)', 'np_DALOGPcoat(c1)', 'np_DHycoat(c3)', 'np_DUicoat(c0)', 'np_DHycoat(c2)', 'np_DAMRcoat(c3)', 'np_DHycoat(c1)', 'np_DALOGPcoat(c3)', 'np_DEnpu(c5)', 'np_DUicoat(c3)', 'np_DUccoat(c3)', 'np_DEnpu(c3)', 'np_DUccoat(c1)', 'np_DUicoat(c1)', 'np_DEnpu(c0)', 'np_DSAdoncoat(c3)', 'np_DVxcoat(c3)', 'np_DSAtotcoat(c3)', 'np_DALOGP2coat(c3)', 'np_DTPSA(Tot)coat(c3)', 'np_DSAacccoat(c3)', 'np_DTPSA(NO)coat(c3)', 'np_DVvdwMGcoat(c3)', 'np_DUccoat(c2)', 'np_DTPSA(NO)coat(c0)', 'np_DUicoat(c2)', 'np_DSAtotcoat(c1)', 'np_DVvdwZAZcoat(c3)', 'np_DTPSA(Tot)coat(c2)', 'np_DTPSA(Tot)coat(c0)', 'np_DNMUnp(c3)', 'np_DVvdwMGcoat(c0)', 'np_DSAdoncoat(c1)', 'np_DALOGP2coat(c0)', 'np_DUccoat(c0)', 'np_DAMRcoat(c2)', 'np_DVvdwZAZcoat(c1)', 'np_DALOGP2coat(c2)', 'np_DTPSA(NO)coat(c2)', 'np_DSAacccoat(c2)', 'np_DVxcoat(c2)', 'np_DVxcoat(c0)', 'np_DEnpu(c2)', 'np_DTPSA(NO)coat(c1)', 'np_DVxcoat(c1)', 'np_DSAacccoat(c1)', 'np_DALOGP2coat(c1)', 'np_DTPSA(Tot)coat(c1)', 'np_DVvdwZAZcoat(c2)', 'np_DSAdoncoat(c0)', 'np_DVvdwMGcoat(c1)', 'np_DEnpu(c1)', 'np_DVvdwMGcoat(c2)', 'np_DVvdwZAZcoat(c0)', 'np_DAMRcoat(c1)', 'np_DSAdoncoat(c2)'

# In[ ]:


df_tr_std = feather.read_dataframe(r'datasets\ds.Class.std.tr.feather')
df_ts_std = feather.read_dataframe(r'datasets\ds.Class.std.ts.feather')


# In[ ]:


len(df_tr_std.columns)


# In[ ]:


#print((df_FeatsImpRem[63:64])['Removed Feature'])
Removed = df_FeatsImpRem.iloc[63, 0]


# In[ ]:


print("-> Removed features:", Removed)
df_tr_std = df_tr_std.drop(Removed, axis=1)
df_ts_std = df_ts_std.drop(Removed, axis=1)


# In[ ]:


len(Removed)


# Final descriptors:

# In[ ]:


list(df_tr_std.columns)


# In[ ]:


len(df_tr_std.columns)-1


# Save final dataset splits with 41 features:

# In[ ]:


feather.write_dataframe(df_tr_std, r'datasets\ds.Final41feats.std.tr.feather')
feather.write_dataframe(df_ts_std, r'datasets\ds.Final41feats.std.ts.feather')


# In[ ]:


df_tr_std.columns


# In[ ]:


len(df_tr_std.columns)


# In[ ]:


df_tr_std.shape


# In[ ]:


df_ts_std.shape


# In[ ]:


df_tr_std.shape[0] + df_ts_std.shape[0]


# In[ ]:


print(list(df_tr_std.columns))


# @muntisa

# In[ ]:




