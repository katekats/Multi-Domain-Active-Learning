#!/usr/bin/env python
# coding: utf-8

# # Data selection and splitting

# The following file contains the code for selecting and splitting the data and then saving the results in pickle files for a particular target domain. Note: Make sure to select the correct domain when loading the data below.

# Make sure that the directories "data/sentence_embeddings/general/sorted/train/" and "data/sentence_embeddings/general/sorted/val_test/" exist such that the data can be saved there.

# ## Importing libraries

# In[1893]:


# imports
import os
import numpy as np
import pandas as pd
import random as rn
import pickle as pkl
import tensorflow as tf


# ## Loading the data

# In[1907]:


# import all the data from the general sentence embeddings
with open('data/sentence_embeddings/general/unsorted/sentemb/sentemb_unlabeled4.p', 'rb') as f:
    data_general = pkl.load(f)

with open('data/sentence_embeddings/general/unsorted/label_domain/label_domain_train_sentemb_unlabeled4.p', 'rb') as f:
    temp_train = pkl.load(f)

with open('data/sentence_embeddings/general/unsorted/label_domain/label_domain_test_sentemb_unlabeled4.p', 'rb') as f:
    temp_test = pkl.load(f)
    
labels_general = np.hstack((temp_train, temp_test))

data_general = data_general.transpose()

#Make sure to load the data of the desired target domain here:
# import all the specific sentence embedding data - here domain 0 was chosen
with open('data/sentence_embeddings/specific/sentemb/sentemb_unlabeled4_5.p', 'rb') as f:
    data_spec = pkl.load(f)
    
with open('data/sentence_embeddings/specific/label_domain/label_domain_train_sentemb_unlabeled4_5.p', 'rb') as f:
    temp_train = pkl.load(f)

with open('data/sentence_embeddings/specific/label_domain/label_domain_test_sentemb_unlabeled4_5.p', 'rb') as f:
    temp_test = pkl.load(f)
    
labels_spec = np.hstack((temp_train, temp_test))

data_spec = data_spec.transpose()




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



labels_spec2=np.repeat(labels_spec,repeats=3, axis=1)

# get indices for sorting the array
ind = sort_array(labels_general, labels_spec2)

# save all the general instances that weren't chosen
data_left = np.delete(data_general, ind, axis = 0)        
labels_left = np.delete(labels_general, ind, axis = 1)

# sorted general sentence embeddings
data_general = data_general[ind]
labels_general = labels_general[:, ind]


# split the data 70-10-20 (train-validation-test) - data was already shuffled before
X_train = data_general[:4200]
X_val = data_general[4200:4800]
X_test = data_general[4800:]


# save data
pkl.dump(np.vstack((X_val, X_test)), open("data/sentence_embeddings/general/sorted/val_test/vt_data4_5.p", "wb"))
pkl.dump(np.hstack((labels_general[:,4200:4800],labels_general[:,4800:])), open("data/sentence_embeddings/general/sorted/val_test/vt_labels4_5.p", "wb"))

pkl.dump(np.vstack((X_train,data_left)), open("data/sentence_embeddings/general/sorted/train/train_data4_5.p", "wb"))
pkl.dump(np.hstack((labels_general[:,:4200],labels_left)), open("data/sentence_embeddings/general/sorted/train/train_labels4_5.p", "wb"))


# In[1844]:


data_spec.shape


# ## Necessary functions

# In[1901]:


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

# In[1902]:


labels_spec2=np.repeat(labels_spec,repeats=3, axis=1)


# In[1851]:


data_general.shape


# In[1876]:


labels_spec2=np.repeat(labels_spec,repeats=10, axis=1)


# In[1903]:


# get indices for sorting the array
ind = sort_array(labels_general, labels_spec2)

# save all the general instances that weren't chosen
data_left = np.delete(data_general, ind, axis = 0)        
labels_left = np.delete(labels_general, ind, axis = 1)

# sorted general sentence embeddings
data_general = data_general[ind]
labels_general = labels_general[:, ind]


# In[1842]:


# get indices for sorting the array
ind = sort_array(labels_general, labels_spec)

# save all the general instances that weren't chosen
data_left = np.delete(data_general, ind, axis = 0)        
labels_left = np.delete(labels_general, ind, axis = 1)

# sorted general sentence embeddings
data_general = data_general[ind]
labels_general = labels_general[:, ind]


# In[1843]:


labels_spec2.shape


# In[375]:


X_train.shape


# In[1904]:


# split the data 70-10-20 (train-validation-test) - data was already shuffled before
X_train = data_general[:4200]
X_val = data_general[4200:4800]
X_test = data_general[4800:]


# In[1527]:


# split the data 70-10-20 (train-validation-test) - data was already shuffled before
X_train = data_general[:1400]
X_val = data_general[1400:1600]
X_test = data_general[1600:]


# In[1826]:


# split the data 70-10-20 (train-validation-test) - data was already shuffled before
X_train = data_general[:14000]
X_val = data_general[14000:16000]
X_test = data_general[16000:]


# In[1905]:


# save data
pkl.dump(np.vstack((X_val, X_test)), open("data/sentence_embeddings/general/sorted/val_test/vt_data4_3.p", "wb"))
pkl.dump(np.hstack((labels_general[:,4200:4800],labels_general[:,4800:])), open("data/sentence_embeddings/general/sorted/val_test/vt_labels4_3.p", "wb"))

pkl.dump(np.vstack((X_train,data_left)), open("data/sentence_embeddings/general/sorted/train/train_data4_3.p", "wb"))
pkl.dump(np.hstack((labels_general[:,:4200],labels_left)), open("data/sentence_embeddings/general/sorted/train/train_labels4_3.p", "wb"))


# In[1670]:


# save data
pkl.dump(np.vstack((X_val, X_test)), open("data/sentence_embeddings/general/sorted/val_test/vt_data_0.p", "wb"))
pkl.dump(np.hstack((labels_general[:,1400:1600],labels_general[:,1600:])), open("data/sentence_embeddings/general/sorted/val_test/vt_labels_0.p", "wb"))

pkl.dump(np.vstack((X_train,data_left)), open("data/sentence_embeddings/general/sorted/train/train_data_0.p", "wb"))
pkl.dump(np.hstack((labels_general[:,:1400],labels_left)), open("data/sentence_embeddings/general/sorted/train/train_labels_0.p", "wb"))


# In[1829]:


# save data
pkl.dump(np.vstack((X_val, X_test)), open("data/sentence_embeddings/general/sorted/val_test/vt_data3_10_1.p", "wb"))
pkl.dump(np.hstack((labels_general[:,14000:16000],labels_general[:,16000:])), open("data/sentence_embeddings/general/sorted/val_test/vt_labels3_10_1.p", "wb"))

pkl.dump(np.vstack((X_train,data_left)), open("data/sentence_embeddings/general/sorted/train/train_data3_10_1.p", "wb"))
pkl.dump(np.hstack((labels_general[:,:14000],labels_left)), open("data/sentence_embeddings/general/sorted/train/train_labels3_10_1.p", "wb"))


# In[1078]:


X_train.shape


# In[379]:


labels_general[:,4800:].shape


# In[ ]:


# Get the optimal hyperparameters
best_hps=tuner2.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. The optimal batch_size
layer is {best_hps.get('batch_size')}, the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}, the optimal dropout rate is {best_hps.get('dropout')}, the optimal number of epochs is {best_hps.get('epochs')} the optimal number of units1 is {best_hps.get('units1')} and th.
""")

