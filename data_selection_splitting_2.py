#!/usr/bin/env python
# coding: utf-8

# # Data selection and splitting

# The following file contains the code for selecting and splitting the data and then saving the results in pickle files for a particular target domain. Note: Make sure to select the correct domain when loading the data below.

# Make sure that the directories "data/sentence_embeddings/general/sorted/train/" and "data/sentence_embeddings/general/sorted/val_test/" exist such that the data can be saved there.

# ## Importing libraries

# In[308]:


# imports
import os
import numpy as np
import pandas as pd
import random as rn
import pickle as pkl
import tensorflow as tf


# ## Loading the data

# In[528]:


# import all the data from the general sentence embeddings
with open('data/sentence_embeddings/general/unsorted/sentemb/sentemb.p', 'rb') as f:
    data_general = pkl.load(f)

with open('data/sentence_embeddings/general/unsorted/label_domain/label_domain_train_sentemb.p', 'rb') as f:
    temp_train = pkl.load(f)

with open('data/sentence_embeddings/general/unsorted/label_domain/label_domain_test_sentemb.p', 'rb') as f:
    temp_test = pkl.load(f)

    
labels_general = np.hstack((temp_train, temp_test))


# In[577]:


# import all the data from the general sentence embeddings
with open('data/sentence_embeddings/general/unsorted/sentemb/sentemb_unlabeled.p', 'rb') as f:
    data_general = pkl.load(f)

with open('data/sentence_embeddings/general/unsorted/label_domain/label_domain_train_sentemb_unlabeled.p', 'rb') as f:
    temp_train = pkl.load(f)

with open('data/sentence_embeddings/general/unsorted/label_domain/label_domain_test_sentemb_unlabeled.p', 'rb') as f:
    temp_test = pkl.load(f)

labels_general = np.hstack((temp_train, temp_test))


# In[578]:


labels_general.shape


# In[579]:


# import all the specific sentence embedding data - here domain 0 was chosen
with open('data/sentence_embeddings/specific/sentemb/sentemb_unlabeled_1.p', 'rb') as f:
    data_spec = pkl.load(f)
    
with open('data/sentence_embeddings/specific/label_domain/label_domain_train_sentemb_unlabeled_1.p', 'rb') as f:
    temp_train = pkl.load(f)

with open('data/sentence_embeddings/specific/label_domain/label_domain_test_sentemb_unlabeled_1.p', 'rb') as f:
    temp_test = pkl.load(f)
    
labels_spec = np.hstack((temp_train, temp_test))


# In[551]:


labels_general.shape


# In[533]:


pd.set_option('display.max_rows', df.shape[0]+1)


# In[565]:



df = pd.DataFrame(labels_spec.transpose(), columns = ['label','domain','idx_domain'])


# In[566]:


df = df[df.label!=3]


# In[567]:


df


# In[568]:


array = df.to_numpy().astype("int")


# In[569]:


labels_spec = array.transpose()


# In[571]:


labels_general.shape


# Make sure to load the data of the desired target domain here:

# In[397]:


# import all the specific sentence embedding data - here domain 0 was chosen
with open('data/sentence_embeddings/specific/sentemb/sentemb_0.p', 'rb') as f:
    data_spec = pkl.load(f)
    
with open('data/sentence_embeddings/specific/label_domain/label_domain_train_sentemb_0.p', 'rb') as f:
    temp_train = pkl.load(f)

with open('data/sentence_embeddings/specific/label_domain/label_domain_test_sentemb_0.p', 'rb') as f:
    temp_test = pkl.load(f)
    
labels_spec = np.hstack((temp_train, temp_test))


# In[586]:


labels_spec.shape


# ## Necessary functions

# In[582]:


# function for sorting two arrays such that both arrays have the same labels
# returns indeces_sorted which consists of indices and is used for sorting array_to_sort
def sort_array(array_to_sort, array_ref):
    
    y, y_ref = array_to_sort[0].astype(int), array_ref[0].astype(int)
    indeces_zeros, indeces_ones = [], []

    # get indices when array_to_sort is 0 (indeces_zeros) and when it is 1 (indeces_ones)
    for i in np.arange(y.shape[0]):
        if y[i] == 0:
            indeces_zeros.append(i)
        else:
            indeces_ones.append(i)

    indeces_sorted = np.zeros(y_ref.shape[0])
    cnt_zeros, cnt_ones = 0,0
    
    # get sorted indeces
    # pair the first positive (/negative) instance of both arrays, etc. 
    for i in np.arange(y_ref.shape[0]):
        if y_ref[i] == 0:
            indeces_sorted[i] = indeces_zeros[cnt_zeros]
            cnt_zeros += 1
        else:
            indeces_sorted[i] = indeces_ones[cnt_ones]
            cnt_ones += 1
    
    return indeces_sorted.astype(int)


# ## Sort, split and save data

# In[583]:


# get indices for sorting the array
ind = sort_array(labels_general, labels_spec)

# save all the general instances that weren't chosen
data_left = np.delete(data_general, ind, axis = 0)        
labels_left = np.delete(labels_general, ind, axis = 1)

# sorted general sentence embeddings
data_general = data_general[ind]
labels_general = labels_general[:, ind]


# In[576]:


X_train


# In[574]:


# split the data 70-10-20 (train-validation-test) - data was already shuffled before
X_train = data_general[:1400]
X_val = data_general[1400:1600]
X_test = data_general[1600:]


# In[575]:


# save data
pkl.dump(np.vstack((X_val, X_test)), open("data/sentence_embeddings/general/sorted/val_test/vt_data_1.p", "wb"))
pkl.dump(np.hstack((labels_general[:,1400:1600],labels_general[:,1600:])), open("data/sentence_embeddings/general/sorted/val_test/vt_labels_1.p", "wb"))

pkl.dump(np.vstack((X_train,data_left)), open("data/sentence_embeddings/general/sorted/train/train_data_1.p", "wb"))
pkl.dump(np.hstack((labels_general[:,:1400],labels_left)), open("data/sentence_embeddings/general/sorted/train/train_labels_1.p", "wb"))


# In[ ]:




