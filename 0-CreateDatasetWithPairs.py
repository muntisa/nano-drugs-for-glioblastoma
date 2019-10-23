#!/usr/bin/env python
# coding: utf-8

# # Create dataset with pairs of anti-glioblastoma drugs - nanoparticles

# In[4]:


import pandas as pd
import numpy as np
import feather


# ## Modify datasets
# 
# Read initial data for drugs:

# In[3]:


df_d = pd.read_csv('./datasets/drug(neuro).csv')
df_d.shape


# In[4]:


# remove duplicates in drugs data
print('Before:', df_d.shape)
df_d.drop_duplicates(keep=False, inplace=True)
print('After :', df_d.shape)


# In[6]:


df_d.head()


# Read initial data for nanoparticles:

# In[7]:


df_np = pd.read_csv('./datasets/nano(neuro).csv')
df_np.shape


# In[8]:


# remove duplicates in NPs data
print('Before:', df_np.shape)
df_np.drop_duplicates(keep=False, inplace=True)
print('After :', df_np.shape)


# In[9]:


df_np.head()


# ## Cutoffs
# 
# Define the cutoff values for biological activities for drugs and nanoparticles in order to create the output variable as a class (0/1 values).
# 
# ### Drug cutoff

# In[10]:


print(list(df_d.columns))


# In[11]:


# verify cutoff values
set(df_d['cutoff'])


# In[12]:


# verify c0
grouped = df_d[['c0=Activity','vij']].groupby('c0=Activity')
# how many examples by each c0
grouped.count()


# In[13]:


#remove some drug Activities
df_d = df_d[df_d['c0=Activity']!='EC50 ug.mL-1']
df_d = df_d[df_d['c0=Activity']!='IC50 ug.mL-1']
df_d.shape


# In[14]:


# verify c0
grouped = df_d[['c0=Activity','vij']].groupby('c0=Activity')
# how many examples by each c0
grouped.count()


# In[15]:


# verify c0 for drugs - median values
grouped = df_d[['c0=Activity','vij']].groupby('c0=Activity')
grouped.median()


# In[16]:


# create logaritm of activity
df_d['log_vij']=np.log(df_d['vij']+1E-15)


# In[17]:


# verify c0 for drugs - median values
grouped = df_d[['c0=Activity','log_vij']].groupby('c0=Activity')
grouped.median()


# In[18]:


grouped.describe()


# In[19]:


# set drug cutoffs
df_d.loc[df_d['c0=Activity'] == 'EC50 nM', 'cutoff'] = 10
df_d.loc[df_d['c0=Activity'] == 'IC50 nM', 'cutoff'] = 10
df_d.loc[df_d['c0=Activity'] == 'LC50 nM', 'cutoff'] = 10


# In[20]:


# make other value before modification
df_d['f(vij)obs'] = -1
df_d['f(vij)obs'] = np.where(df_d['log_vij'] < df_d['cutoff'], 1, 0)
print('Values=',list(set(df_d['f(vij)obs'])))
print('Count=',df_d['f(vij)obs'].count())
print('Sum=',df_d['f(vij)obs'].sum())


# In[21]:


# raw dataset ONLY DRUGS (with extra columns!)
df_d.to_csv('./drug_1_cutoff.csv', index=False)


# ### NP cutoff

# In[22]:


print(list(df_np.columns))
# verify cutoff values
set(df_np['cutoff'])


# In[23]:


# verify c0
grouped_np = df_np[['c0(np)','vij(np)']].groupby('c0(np)')
# how many examples by each c0
grouped_np.count()


# In[24]:


grouped_np = df_np[['c0(np)','vij(np)']].groupby('c0(np)')
# how many examples by each c0
grouped_np.median()


# In[25]:


# create logaritm of activity
df_np['log_vij(np)']=np.log(df_np['vij(np)']+1E-15)


# In[26]:


grouped_np = df_np[['c0(np)','log_vij(np)']].groupby('c0(np)')
# how many examples by each c0
grouped_np.median()


# In[27]:


grouped_np.describe()


# In[28]:


# set NP cutoffs
df_np['cutoff'] = 6


# In[29]:


# make other value before modification
df_np['f(vijnp)'] = -1
df_np['f(vijnp)'] = np.where(df_np['log_vij(np)'] < df_np['cutoff'], 1, 0)
print('Values=',list(set(df_np['f(vijnp)'])))
print('Count=',df_np['f(vijnp)'].count())
print('Sum=',df_np['f(vijnp)'].sum())


# In[30]:


# raw dataset ONLY NPs (with extra columns!)
df_np.to_csv('./NP_1_cutoff.csv', index=False)


# ## Desiderability
# 
# ### Desiderability for drugs

# In[31]:


# verify c0
grouped = df_d[['c0=Activity','log_vij']].groupby('c0=Activity')
# how many examples by each c0
grouped.count()


# In[32]:


# set drug desirabilities
df_d.loc[df_d['c0=Activity'] == 'EC50 nM', 'Desirability'] = -1
df_d.loc[df_d['c0=Activity'] == 'IC50 nM', 'Desirability'] = -1
df_d.loc[df_d['c0=Activity'] == 'LC50 nM', 'Desirability'] = 1


# In[33]:


set(df_d['Desirability'])


# ### Desiderability for NPs

# In[34]:


grouped_np = df_np[['c0(np)','log_vij(np)']].groupby('c0(np)')
# how many examples by each c0
grouped_np.count()


# In[35]:


# set NP desirabilities
df_np.loc[df_np['c0(np)'] == 'CC50 (uM)', 'Desirability']   = 1
df_np.loc[df_np['c0(np)'] == 'EC50 (uM)', 'Desirability']   = -1
df_np.loc[df_np['c0(np)'] == 'IC50 (uM)p', 'Desirability']  = -1
df_np.loc[df_np['c0(np)'] == 'LC50 (uM)', 'Desirability']   = 1
df_np.loc[df_np['c0(np)'] == 'TC50 (uM)', 'Desirability']   = 1


# In[36]:


set(df_np['Desirability'])


# ## Good - Bad
# 
# Two new columns (one for drugs, other for NPs) for future calculation of the final output variable:

# In[37]:


df_d.loc[(df_d['Desirability'] == 1) & (df_d['log_vij'] > df_d['cutoff']), 'd_Good_Bad'] = 'Good'
df_d.loc[(df_d['Desirability'] == 1) & (df_d['log_vij'] < df_d['cutoff']), 'd_Good_Bad'] = 'Bad'
df_d.loc[(df_d['Desirability'] == -1) & (df_d['log_vij'] < df_d['cutoff']), 'd_Good_Bad'] = 'Good'
df_d.loc[(df_d['Desirability'] == -1) & (df_d['log_vij'] > df_d['cutoff']), 'd_Good_Bad'] = 'Bad'


# In[38]:


set(df_d['d_Good_Bad'])


# In[39]:


df_d['d_Good_Bad'].value_counts()


# In[40]:


df_np.loc[(df_np['Desirability'] ==  1) & (df_np['log_vij(np)'] > df_np['cutoff']), 'np_Good_Bad'] = 'Good'
df_np.loc[(df_np['Desirability'] ==  1) & (df_np['log_vij(np)'] < df_np['cutoff']), 'np_Good_Bad'] = 'Bad'
df_np.loc[(df_np['Desirability'] == -1) & (df_np['log_vij(np)'] < df_np['cutoff']), 'np_Good_Bad'] = 'Good'
df_np.loc[(df_np['Desirability'] == -1) & (df_np['log_vij(np)'] > df_np['cutoff']), 'np_Good_Bad'] = 'Bad'


# In[41]:


set(df_np['np_Good_Bad'])


# In[42]:


df_np


# In[43]:


df_np['np_Good_Bad'].value_counts()


# In[44]:


df_d.to_csv('./drug_2_DesiderabilityGoodBad.csv', index=False)


# In[45]:


df_np.to_csv('./NP_2_DesiderabilityGoodBad.csv', index=False)


# ## Select descriptors for drugs and NPs
# 
# These data will be merged to create pairs.

# In[46]:


print(list(df_d.columns))


# In[47]:


drug_cols = ['c0=Activity','d_Good_Bad','d_DPSA(c0)', 'd_DALOGP(c0)', 'd_DPSA(c1)', 'd_DALOG(c1)', 
             'd_DPSA(c2)', 'd_DALOGP(c2)', 'd_DPSA(c3)', 'd_DALOGP(c3)', 'd_DPSA(c4)', 
             'd_DALOGP(c4)', 'd_DPSA(c5)', 'd_DALOGP(c5)', 'd_DPSA(c6)', 'd_DALOGP(c6)', 
             'd_DPSA(c7)', 'd_DALOGP(c7)', 'd_DPSA(c8)', 'd_DALOGP(c8)']


# In[48]:


print(list(df_np.columns))


# In[49]:


np_cols = ['c0(np)','np_Good_Bad','np_DNMUnp(c0)', 'np_DLnp(c0)', 'np_DVnpu(c0)', 'np_DEnpu(c0)', 
           'np_DPnpu(c0)', 'np_DUccoat(c0)',
           'np_DUicoat(c0)', 'np_DHycoat(c0)', 'np_DAMRcoat(c0)', 'np_DTPSA(NO)coat(c0)', 'np_DTPSA(Tot)coat(c0)', 'np_DALOGPcoat(c0)',
           'np_DALOGP2coat(c0)', 'np_DSAtotcoat(c0)', 'np_DSAacccoat(c0)', 'np_DSAdoncoat(c0)', 'np_DVxcoat(c0)', 'np_DVvdwMGcoat(c0)',
           'np_DVvdwZAZcoat(c0)', 'np_DPDIcoat(c0)', 'np_DNMUnp(c1)', 'np_DLnp(c1)', 'np_DVnpu(c1)', 'np_DEnpu(c1)', 'np_DPnpu(c1)',
           'np_DUccoat(c1)', 'np_DUicoat(c1)', 'np_DHycoat(c1)', 'np_DAMRcoat(c1)', 'np_DTPSA(NO)coat(c1)', 'np_DTPSA(Tot)coat(c1)',
           'np_DALOGPcoat(c1)', 'np_DALOGP2coat(c1)', 'np_DSAtotcoat(c1)', 'np_DSAacccoat(c1)', 'np_DSAdoncoat(c1)', 'np_DVxcoat(c1)',
           'np_DVvdwMGcoat(c1)', 'np_DVvdwZAZcoat(c1)', 'np_DPDIcoat(c1)', 'np_DNMUnp(c2)', 'np_DLnp(c2)', 'np_DVnpu(c2)', 'np_DEnpu(c2)',
           'np_DPnpu(c2)', 'np_DUccoat(c2)', 'np_DUicoat(c2)', 'np_DHycoat(c2)', 'np_DAMRcoat(c2)', 'np_DTPSA(NO)coat(c2)', 
           'np_DTPSA(Tot)coat(c2)', 'np_DALOGPcoat(c2)', 'np_DALOGP2coat(c2)', 'np_DSAtotcoat(c2)', 'np_DSAacccoat(c2)', 
           'np_DSAdoncoat(c2)', 'np_DVxcoat(c2)', 'np_DVvdwMGcoat(c2)', 'np_DVvdwZAZcoat(c2)', 'np_DPDIcoat(c2)', 'np_DNMUnp(c3)',
           'np_DLnp(c3)', 'np_DVnpu(c3)', 'np_DEnpu(c3)', 'np_DPnpu(c3)', 'np_DUccoat(c3)', 'np_DUicoat(c3)', 'np_DHycoat(c3)',
           'np_DAMRcoat(c3)', 'np_DTPSA(NO)coat(c3)', 'np_DTPSA(Tot)coat(c3)', 'np_DALOGPcoat(c3)', 'np_DALOGP2coat(c3)',
           'np_DSAtotcoat(c3)', 'np_DSAacccoat(c3)', 'np_DSAdoncoat(c3)', 'np_DVxcoat(c3)', 'np_DVvdwMGcoat(c3)', 'np_DVvdwZAZcoat(c3)',
           'np_DPDIcoat(c3)', 'np_DNMUnp(c5)', 'np_DLnp(c5)', 'np_DVnpu(c5)', 'np_DEnpu(c5)', 'np_DPnpu(c5)', 'np_DUccoat(c5)',
           'np_DUicoat(c5)', 'np_DHycoat(c5)', 'np_DAMRcoat(c5)', 'np_DTPSA(NO)coat(c5)', 'np_DTPSA(Tot)coat(c5)', 'np_DALOGPcoat(c5)',
           'np_DALOGP2coat(c5)', 'np_DSAtotcoat(c5)', 'np_DSAacccoat(c5)', 'np_DSAdoncoat(c5)', 'np_DVxcoat(c5)', 'np_DVvdwMGcoat(c5)',
           'np_DVvdwZAZcoat(c5)', 'np_DPDIcoat(c5)']


# In[50]:


# get only some columns (descriptors + Good_Bad)
df_d2 = df_d[drug_cols].copy()
df_np2= df_np[np_cols].copy()


# In[51]:


df_d2.shape


# In[52]:


df_np2.shape


# We are combining 14044 drugs data with 260 NP data using 18 drug descriptors and 100 NP descriptors.

# In[53]:


df_d.index


# In[54]:


df_np.index


# In[55]:


# simulate pairs 
pairs = 0
for d_index in df_d2.index:
    for np_index in df_np2.index:
        pairs +=1
print('Total pairs drug - np = ', pairs)


# In[56]:


# create temporal columns to combine both dataframes
df_d2['tmp']  = 1
df_np2['tmp'] = 1


# In[57]:


# merge dataframes
df_pairs = pd.merge(df_d2, df_np2, on=['tmp'])
# remove temporal column
df_pairs = df_pairs.drop('tmp', axis=1)


# In[58]:


df_pairs.shape


# In[59]:


df_pairs.head(2)


# ## Final class using Good/Bad of drugs and NPs

# In[60]:


df_pairs['Class'] = 0 # default is bad
df_pairs.loc[(df_pairs['d_Good_Bad'] ==  'Good') & (df_pairs['np_Good_Bad'] == 'Good'), 'Class'] = 1


# ## Add c0 pairs Probability

# In[82]:


groupedp = df_pairs[['Class','c0=Activity','c0(np)']].groupby(['c0=Activity','c0(np)'])
df_prob = groupedp.count().reset_index()
df_prob


# In[83]:


df_prob.rename(columns={'Class': 'Counts'}, inplace=True)
df_prob


# In[84]:


Total_c0s = df_prob['Counts'].sum()
Total_c0s


# In[85]:


#calculate probability of pairs of c0 drug - nano
df_prob['probability'] = df_prob['Counts']/Total_c0s
df_prob


# In[86]:


df_pairs2 = pd.merge(df_prob, df_pairs, on=['c0=Activity', 'c0(np)'])


# In[88]:


df_pairs2.head(2)


# In[89]:


print(list(df_pairs2.columns))


# In[90]:


# drop extra columns
df_pairs2 = df_pairs2.drop(['d_Good_Bad','np_Good_Bad','c0=Activity','c0(np)','Counts'], axis=1)


# In[91]:


print(list(df_pairs2.columns))


# In[92]:


#remove duplicates
print('Before:', df_pairs2.shape)
df_pairs2.drop_duplicates(keep=False, inplace=True)
print('After:', df_pairs2.shape)


# In[93]:


df_pairs2.info(memory_usage='deep')


# In[94]:


# df_last.to_feather('./datasets/ds.Class.feather')
feather.write_dataframe(df_pairs2, './datasets/ds.Class.feather')


# This dataset will be used with Machine Learning methods to find the best classifiers.
# 
# Hf with ML| @muntisa

# In[ ]:




