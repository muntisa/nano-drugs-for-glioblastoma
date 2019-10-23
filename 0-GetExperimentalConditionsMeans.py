#!/usr/bin/env python
# coding: utf-8

# # Means descriptors in experimental conditions for anti-glioblastoma drugs-NPs 

# In[1]:


import pandas as pd
import numpy as np
import feather


# In[2]:


df_test_drugs = pd.read_csv('./datasets/drug(neuro).csv')
print(list(df_test_drugs.columns))


# In[3]:


d_descriptors=['PSA', 'ALOGP']
exp_conds = ['c0=Activity', 'c1=CELL_NAME', 'c2=ORGANISM',
             'c3=TARGET_TYPE', 'c4=ASSAY_ORGANISM', 'c5=TARGETMAPPING',
             'c6=CONFIDENCE', 'c7=CURATEDBY', 'c8=ASSAYTYPE']


# In[5]:


# for each experimental condition get the mean values
for cond in exp_conds:
    temp = []
    temp.append(cond)
    cond_df = df_test_drugs[d_descriptors + temp].groupby([cond])
    print(cond_df.mean())
    cond_df.mean().to_csv('results/d_means_'+cond[:2]+'.csv')


# In[6]:


df_test_np = pd.read_csv('./datasets/nano(neuro).csv')
print(list(df_test_np.columns))


# In[8]:


np_descriptors=['NMUnp', 'Lnp', 'Vnpu', 'Enpu', 'Pnpu', 'Uccoat', 'Uicoat',
               'Hycoat', 'AMRcoat', 'TPSA(NO)coat', 'TPSA(Tot)coat', 'ALOGPcoat',
               'ALOGP2coat', 'SAtotcoat', 'SAacccoat', 'SAdoncoat', 'Vxcoat',
               'VvdwMGcoat', 'VvdwZAZcoat', 'PDIcoat']
np_exp_conds = ['c0(np)', 'c1(np)', 'c2(np)', 'c3(np)', 'c4(np)']


# In[9]:


# for each experimental condition get the mean values
for cond in np_exp_conds:
    temp = []
    temp.append(cond)
    np_cond_df = df_test_np[np_descriptors + temp].groupby([cond])
    print(np_cond_df.mean())
    np_cond_df.mean().to_csv('results/np_means_'+cond[:2]+'.csv')


# Hf with ML! @muntisa

# In[ ]:




