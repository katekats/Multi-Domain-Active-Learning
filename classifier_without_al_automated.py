#!/usr/bin/env python
# coding: utf-8

# In[7]:


import re
import os
import glob
import numpy as np
import numpy as np1
import pandas as pd
import random as rn
import pickle as pkl
from collections import Counter
import matplotlib.pyplot as plt
from scipy.spatial import distance
from keras_self_attention import SeqSelfAttention
#from keras.layers.recurrent import LSTM
#from keras.layers.wrappers import Bidirectional
#from keras.layers.core import RepeatVector
#from keras.callbacks import EarlyStopping
#from keras.callbacks import ModelCheckpoint
#from keras.utils.vis_utils import plot_model


import os
import h5py
import numpy as np
import random as rn
import pickle as pkl
import tensorflow as tf
import pandas as pd


# In[286]:


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


# In[388]:


# importing the data for the general sentence embeddings, here corresponding data from domain 0 was chosen
def return_results_without_AL(k): 
    # set the target domain
    index_spec = k
    print(k)
    with open('data/sentence_embeddings/general/sorted/train/train_data6_'+str(k)+'.p', 'rb') as f:
        X_train_gen = pkl.load(f)

    with open('data/sentence_embeddings/general/sorted/train/train_labels6_'+str(k)+'.p', 'rb') as f:
        y_train = pkl.load(f)

    with open('data/sentence_embeddings/general/sorted/val_test/vt_data6_'+str(k)+'.p', 'rb') as f:
        X_val_test_spec = pkl.load(f)

    with open('data/sentence_embeddings/general/sorted/val_test/vt_labels6_'+str(k)+'.p', 'rb') as f:
        y_val_test = pkl.load(f)

    labels_total = np1.hstack((y_train[:,:4200], y_val_test))
    X_train_gen, X_val_gen, X_test_gen = X_train_gen[:4200], X_val_test_spec[:600], X_val_test_spec[600:]
    y_train, y_val, y_test = y_train[0,:4200], y_val_test[0,:600], y_val_test[0,600:]
   # X_test_spec = np1.delete(X_valid_gen1, ind, axis = 0)      
    #y_test_spec = np1.delete(y_valid_gen1, ind, axis = 0)  
    print(X_train_gen.shape, y_train.shape)
    print(X_val_gen.shape, y_val.shape)


# import the data from the specific sentence embeddings, here corresponding data from domain 0 was chosen
    with open('data/sentence_embeddings/specific/sentemb/sentemb_unlabeled6_'+str(k)+'.p', 'rb') as f:
        X_spec = pkl.load(f)
    
#X_train_spec, X_val_spec, X_test_spec = X_spec[:1400], X_spec[1400:1600], X_spec[1600:2000] 

    import numpy as np
    X_spec=np.repeat(X_spec,repeats=3, axis=1)


    X_train_spec, X_val_spec, X_test_spec = X_spec.transpose()[:4200], X_spec.transpose()[4200:4800], X_spec.transpose()[4800:]
    #X_test_spec, indices = np1.unique(X_test_spec, return_index=True, axis=0)
    #y_test = y_test[indices]
    
    #X_test_gen = X_test_gen[indices]
    #y_test_gen =  y_test_gen[indices]
 
    #print(X_test_spec.shape, X_test_gen.shape, y_test.shape)


    # load the original, unsorted data
    with open('data/sentence_embeddings/general/unsorted/sentemb/sentemb_unlabeled6.p', 'rb') as f:
        data_general = pkl.load(f)

    with open('data/sentence_embeddings/general/unsorted/label_domain/label_domain_train_sentemb_unlabeled6.p', 'rb') as f:
        labels_train = pkl.load(f)
    
    with open('data/sentence_embeddings/general/unsorted/label_domain/label_domain_test_sentemb_unlabeled6.p', 'rb') as f:
        labels_test = pkl.load(f)
    
    labels_general = np.hstack((labels_train, labels_test))

    data_general = data_general.transpose()

# load the cleaned data
    with open('data/cleaned_data/merged_cleaned.p', 'rb') as f:
        df_train = pkl.load(f)
    with open('data/cleaned_data/test_cleaned.p', 'rb') as f:
        df_test = pkl.load(f)
        
  

    # create a list of data frames dfs, each data frame represents one domain
#df = pd.concat([df_train, df_test],ignore_index=True)
#dfs = [x for _, x in df.groupby('domain')]2

    list_unlabel = df_train.index[df_train['label'] == 3].to_list()

    df_train = df_train[~df_train.index.isin(list_unlabel)].reset_index(drop=True)



# create a list of data frames dfs, each data frame represents one domain
    df = pd.concat([df_train, df_test],ignore_index=True)
    #df = df_train
    dfs = [x for _, x in df.groupby('domain')]

# get the word distribution of each domain
# the frequency of each existing word is computed in every domain
    import collections
    import regex as re
    word_counter = []
    for df in dfs:
        counts = collections.Counter()
        words = re.compile(r'\w+')
        reviews = np.array([s for s in df['text']])
        for review in reviews:
            counts.update(words.findall(review.lower()))
        word_counter.append(counts)

# the rows of df are the 16 domains, the columns are all existing words
# the number of the cells of df is the word frequency for the word in the domain
    df_dist = pd.DataFrame(word_counter)
    df_dist = df_dist.fillna(0)
 


# get list js_d of jensen_shannon distances to the target domain
    js_d = []
    for i in range(df_dist.shape[0]):
        d = distance.jensenshannon(np.array(df_dist.iloc[index_spec]), np.array(df_dist.iloc[i]))
        js_d.append(d)
    
# take 5 most similiar distributions
# most_sim_dist is a list of 5 elements with the 5 closest domains to the target domain
    most_sim_dist = sorted(range(len(js_d)), key=lambda i: js_d[i], reverse=True)[-5:]
    most_sim_dist.remove(index_spec)

# remove general embeddings that aren't from these 5 domains
    index_to_keep = [index for index, value in enumerate(labels_general[1]) if int(value) in most_sim_dist]
    labels_general, data_general = labels_general[:, index_to_keep], data_general[index_to_keep]


# get indices for sorting the array
   # ind = sort_array(labels_general, labels_total)

# sort general sentence embeddings
    #data_general, labels_general = data_general[ind], labels_general[:, ind]


    ind = sort_array(labels_general, labels_total)
    #print(ind.shape)
# sort general sentence embeddings
    data_general, labels_general = data_general[ind], labels_general[:, ind]

# data splitting
    #print(X_train_spec.shape)
    X_train_gen, X_val_gen, X_test_gen = data_general[:4200], data_general[4200:4800], data_general[4800:]
    #y_train_gen, y_val_gen, y_test_gen = labels_general[:4200], labels_general[4200:4800], labels_general[4800:]
    
   # X_test_spec, indices = np1.unique(X_test_spec, return_index=True, axis=0)
    #y_test = y_test[indices]
    
    #X_test_gen = X_test_gen[indices]
 
   # print(X_test_spec.shape, X_test_gen.shape, y_test.shape)

    #THE LSTM Classifier

    INPUT_SIZE = 300
    LATENT_SIZE = 300

# domain-general model parts
    inp_gen = tf.keras.Input(shape=(1,INPUT_SIZE))
    #out_gen = tf.keras.layers.Dense(300, activation='sigmoid')(inp_gen)
#
    out_gen = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(LATENT_SIZE, input_shape=(None,1,INPUT_SIZE)))(inp_gen)
    #out_gen, attn_weights_gen = SeqSelfAttention(return_attention = True)(out_gen1)
# domain-specific model parts
    inp_spec = tf.keras.Input(shape=(1,INPUT_SIZE))
   # out_spec = tf.keras.layers.Dense(300, activation='sigmoid')(inp_spec)
    inp_spec_att, attn_weights_spec = SeqSelfAttention(return_attention = True)(inp_spec)
    out_spec = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(LATENT_SIZE, input_shape=(None,1,INPUT_SIZE)))(inp_spec)
   # out_spec, attn_weights_spec = SeqSelfAttention(return_attention = True)(out_spec1)
# concatenate domain-general and domain-specific results
    merged = tf.keras.layers.Concatenate()([out_gen, out_spec])
   # merged = tf.keras.layers.Dense(500, activation='sigmoid')(merged)


    #merged = tf.keras.layers.Dense(100, activation='sigmoid')(merged)
    merged = tf.keras.layers.Dense(100, activation='sigmoid')(merged)
# drop out layer and dense layer
    merged = tf.keras.layers.Dropout(.5)(merged)
    merged = tf.keras.layers.Dense(1, activation='sigmoid')(merged)

    classifier = tf.keras.Model([inp_gen,inp_spec], merged)
    #classifier.summary()
    
    # training the model
    print(X_val_gen.shape, X_val_spec.shape, y_val.shape)
    classifier.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.01), metrics=['accuracy'])
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath="weights/classifier/classifier_without_al/certainty_sampling/classifier_domain_data_6"+str(k)+".h5")
    #print(X_train_gen.shape, X_train_spec.shape)
    history = classifier.fit([np.expand_dims(np.asarray(X_train_gen).astype(np.float32), 1), np.expand_dims(np.asarray(X_train_spec).astype(np.float32), 1)], np.asarray(y_train).astype(np.float32), epochs=20, validation_data = ([np.expand_dims(np.asarray(X_val_gen).astype(np.float32), 1), np.expand_dims(np.asarray(X_val_spec).astype(np.float32), 1)], np.asarray(y_val).astype(np.float32)), callbacks = [checkpoint, es], batch_size=32)

# evaluating the model
    score = classifier.evaluate([np.expand_dims(np.asarray(X_test_gen).astype(np.float32), 1), np.expand_dims(np.asarray(X_test_spec).astype(np.float32), 1)], np.asarray(y_test).astype(np.float32), verbose=0) 
    print(k)
    return(k, 'Final accuracy score: '+str(score[1]))


# In[389]:


class TestCase:
   def __init__(self, name, i_range):
       self.name = name
       self.i_range = i_range
      # self.pars = pars
       
test_cases = [
   TestCase("peirama 1", range(0,16))
#     TestCase("peirama 1", range(0,16), [(1400, 2200), (2100, 3000)]),
  # TestCase("peirama 2", range(0,16), [1400, 2200])
  # TestCase("peirama 3", range(0,100), [(1400, 2200), (2100, 3000)]),
]

for test_case in test_cases:
   for i in test_case.i_range:
      # for pars in test_case.pars:
       print(test_case.name, i)
       x = return_results_without_AL(i)
       print(x)


# In[399]:


# importing the data for the general sentence embeddings, here corresponding data from domain 0 was chosen
def return_results_without_AL_seperated_test(k): 
    # set the target domain
    index_spec = k
    print(k)
    with open('data/sentence_embeddings/general/sorted/train/train_data_all_6_'+str(k)+'.p', 'rb') as f:
        X_train_gen = pkl.load(f)

    with open('data/sentence_embeddings/general/sorted/train/train_labels6_'+str(k)+'.p', 'rb') as f:
        y_train = pkl.load(f)

    with open('data/sentence_embeddings/general/sorted/val_test/test_data6_'+str(k)+'.p', 'rb') as f:
        X_test_spec = pkl.load(f)
    with open('data/sentence_embeddings/general/sorted/val_test/val_data6_'+str(k)+'.p', 'rb') as f:
        X_val_spec = pkl.load(f)    

    with open('data/sentence_embeddings/general/sorted/val_test/test_labels6_'+str(k)+'.p', 'rb') as f:
        y_test = pkl.load(f)
        
    with open('data/sentence_embeddings/general/sorted/train/val_labels6_'+str(k)+'.p', 'rb') as f:
        y_val = pkl.load(f)    
   # print(y_train.shape)
    labels_total_train_val = y_train[:,:]
    labels_total_test = y_test
    X_train_gen, X_val_gen, X_test_gen = X_train_gen[:4200], X_val_spec, X_test_spec
    y_train= y_train[0,:4200]
    y_val= y_val[0,:]
    y_test= y_test[0,:]
   # X_test_spec = np1.delete(X_valid_gen1, ind, axis = 0)      
    #y_test_spec = np1.delete(y_valid_gen1, ind, axis = 0)  
   # print(X_val_gen.shape)
    #labels_total_train_val = y_train[:,:]
    #labels_total_test = y_test
    #X_train_gen, X_val_gen, X_test_gen = X_train_gen[:4200], X_val_spec, X_test_spec
    #y_train= y_train[0,:4200]
    #y_val= y_val[0,:]
    #y_test= y_test[0,:]


# import the data from the specific sentence embeddings, here corresponding data from domain 0 was chosen
    with open('data/sentence_embeddings/specific/sentemb/sentemb_unlabeled6_'+str(k)+'.p', 'rb') as f:
        X_spec = pkl.load(f)
       # print(X_spec.shape)
        X_test_spec = X_spec.transpose()[-400:]
        #print(X_test_spec.shape)
        X_train_val_spec = X_spec.transpose()[:(X_spec.transpose().shape[0]-400)]
        #print(X_train_val_spec.shape)
#X_train_spec, X_val_spec, X_test_spec = X_spec[:1400], X_spec[1400:1600], X_spec[1600:2000] 

    import numpy as np
    X_train_val_spec=np.repeat(X_train_val_spec,repeats=3, axis=0)

    
    X_train_spec, X_val_spec = X_train_val_spec[:4200], X_train_val_spec[4200:]
    #X_train_spec, X_val_spec = X_train_val_spec[:4200], X_train_val_spec[4200:]
    #print(X_train_gen.shape, X_train_spec.shape)
    #print(X_val_spec.shape)
    #X_test_spec, indices = np1.unique(X_test_spec, return_index=True, axis=0)
    #y_test = y_test[indices]
    
   # X_test_gen = X_test_gen[indices]
    #y_test_gen =  y_test_gen[indices]
 
    #print(X_test_spec.shape, X_test_gen.shape, y_test.shape)


    # load the original, unsorted data
    with open('data/sentence_embeddings/general/unsorted/sentemb/sentemb_unlabeled6.p', 'rb') as f:
        data_general = pkl.load(f)

    with open('data/sentence_embeddings/general/unsorted/label_domain/label_domain_train_sentemb_unlabeled6.p', 'rb') as f:
        labels_train = pkl.load(f)
    
    with open('data/sentence_embeddings/general/unsorted/label_domain/label_domain_test_sentemb_unlabeled6.p', 'rb') as f:
        labels_test = pkl.load(f)
    
    #labels_general = np.hstack((labels_train, labels_test))

    data_general_train = data_general[:,:25380]
    data_general_test = data_general[:,25380:]
    data_general_train = data_general_train.transpose()
    data_general_test = data_general_test.transpose()
# load the cleaned data
    with open('data/cleaned_data/merged_cleaned.p', 'rb') as f:
        df_train = pkl.load(f)
    with open('data/cleaned_data/test_cleaned.p', 'rb') as f:
        df_test = pkl.load(f)

# create a list of data frames dfs, each data frame represents one domain
#df = pd.concat([df_train, df_test],ignore_index=True)
#dfs = [x for _, x in df.groupby('domain')]

    list_unlabel = df_train.index[df_train['label'] == 3].to_list()

    df_train = df_train[~df_train.index.isin(list_unlabel)].reset_index(drop=True)



# create a list of data frames dfs, each data frame represents one domain
    df = pd.concat([df_train, df_test],ignore_index=True)
    #df = df_train
    dfs = [x for _, x in df.groupby('domain')]

# get the word distribution of each domain
# the frequency of each existing word is computed in every domain
    import collections
    import regex as re
    word_counter = []
    for df in dfs:
        counts = collections.Counter()
        words = re.compile(r'\w+')
        reviews = np.array([s for s in df['text']])
        for review in reviews:
            counts.update(words.findall(review.lower()))
        word_counter.append(counts)

# the rows of df are the 16 domains, the columns are all existing words
# the number of the cells of df is the word frequency for the word in the domain
    df_dist = pd.DataFrame(word_counter)
    df_dist = df_dist.fillna(0)
 


# get list js_d of jensen_shannon distances to the target domain
    js_d = []
    for i in range(df_dist.shape[0]):
        d = distance.jensenshannon(np.array(df_dist.iloc[index_spec]), np.array(df_dist.iloc[i]))
        js_d.append(d)
    
# take 5 most similiar distributions
# most_sim_dist is a list of 5 elements with the 5 closest domains to the target domain
    most_sim_dist = sorted(range(len(js_d)), key=lambda i: js_d[i], reverse=True)[-5:]
    most_sim_dist.remove(index_spec)
    #print(most_sim_dist)
# remove general embeddings from the training set that aren't from these 5 domains
    index_to_keep_train = [index for index, value in enumerate(labels_train[1]) if int(value) in most_sim_dist]
    labels_train, data_general_train = labels_train[:, index_to_keep_train], data_general_train[index_to_keep_train]
    #print(labels_train.shape, data_general_train.shape)

# remove general embeddings from test set that aren't from these 5 domains
    index_to_keep_test = [index for index, value in enumerate(labels_test[1]) if int(value) in most_sim_dist]
    labels_test, data_general_test = labels_test[:, index_to_keep_test], data_general_test[index_to_keep_test]
    #print(data_general, labels_general)    
    
# get indices for sorting the array
   # ind = sort_array(labels_general, labels_total)

# sort general sentence embeddings
    #data_general, labels_general = data_general[ind], labels_general[:, ind]

    print(labels_train.shape, labels_total_train_val.shape)
    ind_train = sort_array(labels_train, labels_total_train_val)
    #print(ind.shape)
# sort general sentence embeddings
    data_general_train, labels_train = data_general_train[ind_train], labels_train[:, ind_train]

    print(data_general_train.shape)
    
    ind_test = sort_array(labels_test, labels_total_test)
    #print(ind.shape)
# sort general sentence embeddings
    data_general_test, labels_test = data_general_test[ind_test], labels_test[:, ind_test]
# data splitting
    #print(X_train_spec.shape)
    X_train_gen, X_val_gen = data_general_train[:4200], data_general_train[4200:]
    print(X_val_gen.shape)
    
     #X_train_gen, X_val_gen = data_general_train[:4200], data_general_train[4200:]
    #print(data_general_train.shape, X_val_gen.shape)
   # y_train_gen, y_val_gen, y_test_gen = labels_general[:4200], labels_general[4200:4800], labels_general[4800:]
   # X_test = data_general_test[:400]
   # y_train_gen, y_val_gen, y_test_gen = labels_general[:5000], labels_general[5000:6000]
    #print(data_general_train.shape, X_val_gen.shape)
   # y_train_gen, y_val_gen, y_test_gen = labels_general[:4200], labels_general[4200:4800], labels_general[4800:]
    X_test = data_general_test[:400]
   # X_test_spec, indices = np1.unique(X_test_spec, return_index=True, axis=0)
   # y_test = y_test[indices]
    
   # X_test_gen = X_test_gen[indices]
 
   # print(X_test_spec.shape, X_test_gen.shape, y_test.shape)




    
     #THE LSTM Classifier

    INPUT_SIZE = 300
    LATENT_SIZE = 300

# domain-general model parts
    inp_gen = tf.keras.Input(shape=(1,INPUT_SIZE))
    #out_gen = tf.keras.layers.Dense(300, activation='sigmoid')(inp_gen)
#
    #out_gen = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(LATENT_SIZE, input_shape=(None,1,INPUT_SIZE)))(inp_gen)
    #out_gen, attn_weights_gen = SeqSelfAttention(return_attention = True)(out_gen1)
# domain-specific model parts
    inp_spec = tf.keras.Input(shape=(1,INPUT_SIZE))
   # out_spec = tf.keras.layers.Dense(300, activation='sigmoid')(inp_spec)
#inp_spec_att, attn_weights_spec = SeqSelfAttention(return_attention = True)(inp_spec)
   # out_spec = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(LATENT_SIZE, input_shape=(None,1,INPUT_SIZE)))(inp_spec)
   # out_spec, attn_weights_spec = SeqSelfAttention(return_attention = True)(out_spec1)
# concatenate domain-general and domain-specific results
    merged = tf.keras.layers.Concatenate()([inp_gen, inp_spec])
    merged = tf.keras.layers.Dense(200, activation='sigmoid')(merged)


    #merged = tf.keras.layers.Dense(100, activation='sigmoid')(merged)
    #merged = tf.keras.layers.Dense(100, activation='sigmoid')(merged)
# drop out layer and dense layer
    merged = tf.keras.layers.Dropout(.2)(merged)
    merged = tf.keras.layers.Dense(1, activation='sigmoid')(merged)

    classifier = tf.keras.Model([inp_gen,inp_spec], merged)
    #classifier.summary()
    #classifier.summary()
    
    # training the model
    print(X_val_gen.shape, X_val_spec.shape, y_val.shape)
    classifier.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.01), metrics=['accuracy'])
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath="weights/classifier/classifier_without_al/certainty_sampling/classifier_domain_data_6"+str(k)+".h5")
    #print(X_train_gen.shape, X_train_spec.shape)
    history = classifier.fit([np.expand_dims(np.asarray(X_train_gen).astype(np.float32), 1), np.expand_dims(np.asarray(X_train_spec).astype(np.float32), 1)], np.asarray(y_train).astype(np.float32), epochs=40, validation_data = ([np.expand_dims(np.asarray(X_val_gen).astype(np.float32), 1), np.expand_dims(np.asarray(X_val_spec).astype(np.float32), 1)], np.asarray(y_val).astype(np.float32)), callbacks = [checkpoint, es], batch_size=32)

# evaluating the model
    score = classifier.evaluate([np.expand_dims(np.asarray(X_test).astype(np.float32), 1), np.expand_dims(np.asarray(X_test_spec).astype(np.float32), 1)], np.asarray(y_test).astype(np.float32), verbose=0) 
    print(k)
    return(k, 'Final accuracy score: '+str(score[1]))


# In[400]:


class TestCase:
   def __init__(self, name, i_range):
       self.name = name
       self.i_range = i_range
      # self.pars = pars
       
test_cases = [
   TestCase("peirama 1", range(0,16))
#     TestCase("peirama 1", range(0,16), [(1400, 2200), (2100, 3000)]),
  # TestCase("peirama 2", range(0,16), [1400, 2200])
  # TestCase("peirama 3", range(0,100), [(1400, 2200), (2100, 3000)]),
]

for test_case in test_cases:
   for i in test_case.i_range:
      # for pars in test_case.pars:
       print(test_case.name, i)
       x = return_results_without_AL_seperated_test(i)
       print(x)


# In[374]:


import keras_tuner as kt


# In[329]:


# importing the data for the general sentence embeddings, here corresponding data from domain 0 was chosen
def return_results_without_AL_seperated_test(k): 
    # set the target domain
    index_spec = k
    print(k)
    with open('data/sentence_embeddings/general/sorted/train/train_data_all_11_'+str(k)+'.p', 'rb') as f:
        X_train_gen = pkl.load(f)

    with open('data/sentence_embeddings/general/sorted/train/train_labels11_'+str(k)+'.p', 'rb') as f:
        y_train = pkl.load(f)

    with open('data/sentence_embeddings/general/sorted/val_test/test_data11_'+str(k)+'.p', 'rb') as f:
        X_test_spec = pkl.load(f)
    with open('data/sentence_embeddings/general/sorted/val_test/val_datα11_'+str(k)+'.p', 'rb') as f:
        X_val_spec = pkl.load(f)    

    with open('data/sentence_embeddings/general/sorted/val_test/test_labels11_'+str(k)+'.p', 'rb') as f:
        y_test = pkl.load(f)
        
    with open('data/sentence_embeddings/general/sorted/train/val_labels11_'+str(k)+'.p', 'rb') as f:
        y_val = pkl.load(f)    
   # print(y_train.shape)
    labels_total_train_val = y_train[:,:]
    labels_total_test = y_test
    X_train_gen, X_val_gen, X_test_gen = X_train_gen[:4200], X_val_spec, X_test_spec
    y_train= y_train[0,:4200]
    y_val= y_val[0,:]
    y_test= y_test[0,:]
   # X_test_spec = np1.delete(X_valid_gen1, ind, axis = 0)      
    #y_test_spec = np1.delete(y_valid_gen1, ind, axis = 0)  
    



# import the data from the specific sentence embeddings, here corresponding data from domain 0 was chosen
    with open('data/sentence_embeddings/specific/sentemb/sentemb_unlabeled11_'+str(k)+'.p', 'rb') as f:
        X_spec = pkl.load(f)
    print(X_spec.shape)
    X_test_spec = X_spec.transpose()[-400:]
    print(X_test_spec.shape)
    X_train_val_spec = X_spec.transpose()[:(X_spec.transpose().shape[0]-400)]
    print(X_train_val_spec.shape)
#X_train_spec, X_val_spec, X_test_spec = X_spec[:1400], X_spec[1400:1600], X_spec[1600:2000] 

    import numpy as np
    X_train_val_spec=np.repeat(X_train_val_spec,repeats=3, axis=0)

    
    X_train_spec, X_val_spec = X_train_val_spec[:4200], X_train_val_spec[4200:]
    print(X_train_gen.shape, X_train_spec.shape)
    #print(X_val_spec.shape)
    #X_test_spec, indices = np1.unique(X_test_spec, return_index=True, axis=0)
    #y_test = y_test[indices]
    
   # X_test_gen = X_test_gen[indices]
    #y_test_gen =  y_test_gen[indices]
 
    #print(X_test_spec.shape, X_test_gen.shape, y_test.shape)


    # load the original, unsorted data
    with open('data/sentence_embeddings/general/unsorted/sentemb/sentemb_unlabeled11.p', 'rb') as f:
        data_general = pkl.load(f)

    with open('data/sentence_embeddings/general/unsorted/label_domain/label_domain_train_sentemb_unlabeled11.p', 'rb') as f:
        labels_train = pkl.load(f)
    
    with open('data/sentence_embeddings/general/unsorted/label_domain/label_domain_test_sentemb_unlabeled11.p', 'rb') as f:
        labels_test = pkl.load(f)
        
    data_general_train = data_general[:,:25380]
    data_general_test = data_general[:,25380:]
    data_general_train = data_general_train.transpose()
    data_general_test = data_general_test.transpose()
# load the cleaned data
    with open('data/cleaned_data/merged_cleaned.p', 'rb') as f:
        df_train = pkl.load(f)
    with open('data/cleaned_data/test_cleaned.p', 'rb') as f:
        df_test = pkl.load(f)   
    
    list_unlabel = df_train.index[df_train['label'] == 3].to_list()

    df_train = df_train[~df_train.index.isin(list_unlabel)].reset_index(drop=True)



# create a list of data frames dfs, each data frame represents one domain
    df = pd.concat([df_train, df_test],ignore_index=True)
    #df = df_train
    dfs = [x for _, x in df.groupby('domain')]

# get the word distribution of each domain
# the frequency of each existing word is computed in every domain
    import collections
    import regex as re
    word_counter = []
    for df in dfs:
        counts = collections.Counter()
        words = re.compile(r'\w+')
        reviews = np.array([s for s in df['text']])
        for review in reviews:
            counts.update(words.findall(review.lower()))
        word_counter.append(counts)

# the rows of df are the 16 domains, the columns are all existing words
# the number of the cells of df is the word frequency for the word in the domain
    df_dist = pd.DataFrame(word_counter)
    df_dist = df_dist.fillna(0)
 


# get list js_d of jensen_shannon distances to the target domain
    js_d = []
    for i in range(df_dist.shape[0]):
        d = distance.jensenshannon(np.array(df_dist.iloc[index_spec]), np.array(df_dist.iloc[i]))
        js_d.append(d)
    
# take 5 most similiar distributions
# most_sim_dist is a list of 5 elements with the 5 closest domains to the target domain
    most_sim_dist = sorted(range(len(js_d)), key=lambda i: js_d[i], reverse=True)[-5:]
    most_sim_dist.remove(index_spec)
    #print(most_sim_dist)
# remove general embeddings from the training set that aren't from these 5 domains
    index_to_keep_train = [index for index, value in enumerate(labels_train[1]) if int(value) in most_sim_dist]
    labels_train, data_general_train = labels_train[:, index_to_keep_train], data_general_train[index_to_keep_train]
    #print(labels_train.shape, data_general_train.shape)

# remove general embeddings from test set that aren't from these 5 domains
    index_to_keep_test = [index for index, value in enumerate(labels_test[1]) if int(value) in most_sim_dist]
    labels_test, data_general_test = labels_test[:, index_to_keep_test], data_general_test[index_to_keep_test]
    #print(data_general, labels_general)    
    
# get indices for sorting the array
   # ind = sort_array(labels_general, labels_total)

# sort general sentence embeddings
    #data_general, labels_general = data_general[ind], labels_general[:, ind]

    print(labels_train.shape, labels_total_train_val.shape)
    ind_train = sort_array(labels_train, labels_total_train_val)
    #print(ind.shape)
# sort general sentence embeddings
    data_general_train, labels_train = data_general_train[ind_train], labels_train[:, ind_train]

    #print(data_general_train.shape)
    
    ind_test = sort_array(labels_test, labels_total_test)
    #print(ind.shape)
# sort general sentence embeddings
    data_general_test, labels_test = data_general_test[ind_test], labels_test[:, ind_test]
# data splitting
    #print(X_train_spec.shape)
    X_train_gen, X_val_gen = data_general_train[:4200], data_general_train[4200:]
    #print(data_general_train.shape, X_val_gen.shape)
   # y_train_gen, y_val_gen, y_test_gen = labels_general[:4200], labels_general[4200:4800], labels_general[4800:]
    X_test = data_general_test[:400]

    
     #THE LSTM Classifier

    INPUT_SIZE = 300
    LATENT_SIZE = 300

# domain-general model parts
    inp_gen = tf.keras.Input(shape=(1,INPUT_SIZE))
    #out_gen = tf.keras.layers.Dense(300, activation='sigmoid')(inp_gen)
#
    #out_gen = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(LATENT_SIZE, input_shape=(None,1,INPUT_SIZE)))(inp_gen)
    #out_gen, attn_weights_gen = SeqSelfAttention(return_attention = True)(out_gen1)
# domain-specific model parts
    inp_spec = tf.keras.Input(shape=(1,INPUT_SIZE))
   # out_spec = tf.keras.layers.Dense(300, activation='sigmoid')(inp_spec)
#inp_spec_att, attn_weights_spec = SeqSelfAttention(return_attention = True)(inp_spec)
   # out_spec = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(LATENT_SIZE, input_shape=(None,1,INPUT_SIZE)))(inp_spec)
   # out_spec, attn_weights_spec = SeqSelfAttention(return_attention = True)(out_spec1)
# concatenate domain-general and domain-specific results
    merged = tf.keras.layers.Concatenate()([inp_gen, inp_spec])
    merged = tf.keras.layers.Dense(300, activation='sigmoid')(merged)


    #merged = tf.keras.layers.Dense(100, activation='sigmoid')(merged)
    #merged = tf.keras.layers.Dense(100, activation='sigmoid')(merged)
# drop out layer and dense layer
    merged = tf.keras.layers.Dropout(.1)(merged)
    merged = tf.keras.layers.Dense(1, activation='sigmoid')(merged)

    classifier = tf.keras.Model([inp_gen,inp_spec], merged)
    #classifier.summary()
    #classifier.summary()
    
    # training the model
    print(X_val_gen.shape, X_val_spec.shape, y_val.shape)
    classifier.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.01), metrics=['accuracy'])
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath="weights/classifier/classifier_without_al/certainty_sampling/classifier_domain_data_11"+str(k)+".h5")
    #print(X_train_gen.shape, X_train_spec.shape)
    history = classifier.fit([np.expand_dims(np.asarray(X_train_gen).astype(np.float32), 1), np.expand_dims(np.asarray(X_train_spec).astype(np.float32), 1)], np.asarray(y_train).astype(np.float32), epochs=30, validation_data = ([np.expand_dims(np.asarray(X_val_gen).astype(np.float32), 1), np.expand_dims(np.asarray(X_val_spec).astype(np.float32), 1)], np.asarray(y_val).astype(np.float32)), callbacks = [checkpoint, es], batch_size=32)

# evaluating the model
    score = classifier.evaluate([np.expand_dims(np.asarray(X_test).astype(np.float32), 1), np.expand_dims(np.asarray(X_test_spec).astype(np.float32), 1)], np.asarray(y_test).astype(np.float32), verbose=0) 
    print(k)
    return(k, 'Final accuracy score: '+str(score[1]))


# In[375]:


class TestCase:
   def __init__(self, name, i_range):
       self.name = name
       self.i_range = i_range
      # self.pars = pars
       
test_cases = [
   TestCase("peirama 1", range(0,16))
#     TestCase("peirama 1", range(0,16), [(1400, 2200), (2100, 3000)]),
  # TestCase("peirama 2", range(0,16), [1400, 2200])
  # TestCase("peirama 3", range(0,100), [(1400, 2200), (2100, 3000)]),
]

for test_case in test_cases:
   for i in test_case.i_range:
      # for pars in test_case.pars:
       print(test_case.name, i)
       x = return_results_without_AL_seperated_test(i)
       print(x)


# In[390]:


index_spec = 5
#print(k)
with open('data/sentence_embeddings/general/sorted/train/train_data6_5.p', 'rb') as f:
    X_train_gen = pkl.load(f)

with open('data/sentence_embeddings/general/sorted/train/train_labels6_5.p', 'rb') as f:
    y_train = pkl.load(f)

with open('data/sentence_embeddings/general/sorted/val_test/vt_data6_5.p', 'rb') as f:
    X_val_test_spec = pkl.load(f)

with open('data/sentence_embeddings/general/sorted/val_test/vt_labels6_5.p', 'rb') as f:
    y_val_test = pkl.load(f)

labels_total = np1.hstack((y_train[:,:4200], y_val_test))
X_train_gen, X_val_gen, X_test_gen = X_train_gen[:4200], X_val_test_spec[:600], X_val_test_spec[600:]
y_train, y_val, y_test = y_train[0,:4200], y_val_test[0,:600], y_val_test[0,600:]




# import the data from the specific sentence embeddings, here corresponding data from domain 0 was chosen
with open('data/sentence_embeddings/specific/sentemb/sentemb_unlabeled6_5.p', 'rb') as f:
    X_spec = pkl.load(f)
    
#X_train_spec, X_val_spec, X_test_spec = X_spec[:1400], X_spec[1400:1600], X_spec[1600:2000] 

import numpy as np
X_spec=np.repeat(X_spec,repeats=3, axis=1)


X_train_spec, X_val_spec, X_test_spec = X_spec.transpose()[:4200], X_spec.transpose()[4200:4800], X_spec.transpose()[4800:]
    
 
    


    # load the original, unsorted data
with open('data/sentence_embeddings/general/unsorted/sentemb/sentemb_unlabeled6.p', 'rb') as f:
    data_general = pkl.load(f)

with open('data/sentence_embeddings/general/unsorted/label_domain/label_domain_train_sentemb_unlabeled6.p', 'rb') as f:
    labels_train = pkl.load(f)
    
with open('data/sentence_embeddings/general/unsorted/label_domain/label_domain_test_sentemb_unlabeled6.p', 'rb') as f:
    labels_test = pkl.load(f)
    
labels_general = np.hstack((labels_train, labels_test))

data_general = data_general.transpose()

# load the cleaned data
with open('data/cleaned_data/merged_cleaned.p', 'rb') as f:
    df_train = pkl.load(f)
with open('data/cleaned_data/test_cleaned.p', 'rb') as f:
    df_test = pkl.load(f)

# create a list of data frames dfs, each data frame represents one domain
#df = pd.concat([df_train, df_test],ignore_index=True)
#dfs = [x for _, x in df.groupby('domain')]

list_unlabel = df_train.index[df_train['label'] == 3].to_list()

df_train = df_train[~df_train.index.isin(list_unlabel)].reset_index(drop=True)



# create a list of data frames dfs, each data frame represents one domain
df = pd.concat([df_train, df_test],ignore_index=True)
dfs = [x for _, x in df.groupby('domain')]

# get the word distribution of each domain
# the frequency of each existing word is computed in every domain
import collections
import regex as re
word_counter = []
for df in dfs:
    counts = collections.Counter()
    words = re.compile(r'\w+')
    reviews = np.array([s for s in df['text']])
    for review in reviews:
        counts.update(words.findall(review.lower()))
    word_counter.append(counts)

# the rows of df are the 16 domains, the columns are all existing words
# the number of the cells of df is the word frequency for the word in the domain
df_dist = pd.DataFrame(word_counter)
df_dist = df_dist.fillna(0)
 


# get list js_d of jensen_shannon distances to the target domain
js_d = []
for i in range(df_dist.shape[0]):
    d = distance.jensenshannon(np.array(df_dist.iloc[index_spec]), np.array(df_dist.iloc[i]))
    js_d.append(d)
    
# take 5 most similiar distributions
# most_sim_dist is a list of 5 elements with the 5 closest domains to the target domain
most_sim_dist = sorted(range(len(js_d)), key=lambda i: js_d[i], reverse=True)[-5:]
most_sim_dist.remove(index_spec)

# remove general embeddings that aren't from these 5 domains
index_to_keep = [index for index, value in enumerate(labels_general[1]) if int(value) in most_sim_dist]
labels_general, data_general = labels_general[:, index_to_keep], data_general[index_to_keep]


# get indices for sorting the array
   # ind = sort_array(labels_general, labels_total)

# sort general sentence embeddings
    #data_general, labels_general = data_general[ind], labels_general[:, ind]


ind = sort_array(labels_general, labels_total)
    #print(ind.shape)
# sort general sentence embeddings
data_general, labels_general = data_general[ind], labels_general[:, ind]

# data splitting
    #print(X_train_spec.shape)
X_train_gen, X_val_gen, X_test_gen = data_general[:4200], data_general[4200:4800], data_general[4800:]



    #THE LSTM Classifier


# In[394]:


index_spec = 5
print(3)
with open('data/sentence_embeddings/general/sorted/train/train_data_all_6_5.p', 'rb') as f:
    X_train_gen = pkl.load(f)

with open('data/sentence_embeddings/general/sorted/train/train_labels6_5.p', 'rb') as f:
    y_train = pkl.load(f)

with open('data/sentence_embeddings/general/sorted/val_test/test_data6_5.p', 'rb') as f:
    X_test_spec = pkl.load(f)
with open('data/sentence_embeddings/general/sorted/val_test/val_data6_5.p', 'rb') as f:
    X_val_spec = pkl.load(f)    

with open('data/sentence_embeddings/general/sorted/val_test/test_labels6_5.p', 'rb') as f:
    y_test = pkl.load(f)
        
with open('data/sentence_embeddings/general/sorted/train/val_labels6_5.p', 'rb') as f:
    y_val = pkl.load(f)    
   # print(y_train.shape)
labels_total_train_val = y_train[:,:]
labels_total_test = y_test
X_train_gen, X_val_gen, X_test_gen = X_train_gen[:4200], X_val_spec, X_test_spec
y_train= y_train[0,:4200]
y_val= y_val[0,:]
y_test= y_test[0,:]
   # X_test_spec = np1.delete(X_valid_gen1, ind, axis = 0)      
    #y_test_spec = np1.delete(y_valid_gen1, ind, axis = 0)  
   



# import the data from the specific sentence embeddings, here corresponding data from domain 0 was chosen
with open('data/sentence_embeddings/specific/sentemb/sentemb_unlabeled6_5.p', 'rb') as f:
    X_spec = pkl.load(f)
#print(X_spec.shape)
X_test_spec = X_spec.transpose()[-400:]
#print(X_test_spec.shape)
X_train_val_spec = X_spec.transpose()[:(X_spec.transpose().shape[0]-400)]
#print(X_train_val_spec.shape)
#X_train_spec, X_val_spec, X_test_spec = X_spec[:1400], X_spec[1400:1600], X_spec[1600:2000] 

import numpy as np
X_train_val_spec=np.repeat(X_train_val_spec,repeats=3, axis=0)


X_train_spec, X_val_spec = X_train_val_spec[:4200], X_train_val_spec[4200:]
    
    #print(X_val_spec.shape)
    #X_test_spec, indices = np1.unique(X_test_spec, return_index=True, axis=0)
    #y_test = y_test[indices]
    
   # X_test_gen = X_test_gen[indices]
    #y_test_gen =  y_test_gen[indices]
 
    #print(X_test_spec.shape, X_test_gen.shape, y_test.shape)


    # load the original, unsorted data
with open('data/sentence_embeddings/general/unsorted/sentemb/sentemb_unlabeled6.p', 'rb') as f:
    data_general = pkl.load(f)

with open('data/sentence_embeddings/general/unsorted/label_domain/label_domain_train_sentemb_unlabeled6.p', 'rb') as f:
    labels_train = pkl.load(f)
    
with open('data/sentence_embeddings/general/unsorted/label_domain/label_domain_test_sentemb_unlabeled6.p', 'rb') as f:
    labels_test = pkl.load(f)
    
    #labels_general = np.hstack((labels_train, labels_test))

data_general_train = data_general[:,:25380]
data_general_test = data_general[:,25380:]
data_general_train = data_general_train.transpose()
data_general_test = data_general_test.transpose()
# load the cleaned data
with open('data/cleaned_data/merged_cleaned.p', 'rb') as f:
    df_train = pkl.load(f)
with open('data/cleaned_data/test_cleaned.p', 'rb') as f:
    df_test = pkl.load(f)

# create a list of data frames dfs, each data frame represents one domain
#df = pd.concat([df_train, df_test],ignore_index=True)
#dfs = [x for _, x in df.groupby('domain')]

list_unlabel = df_train.index[df_train['label'] == 3].to_list()

df_train = df_train[~df_train.index.isin(list_unlabel)].reset_index(drop=True)



# create a list of data frames dfs, each data frame represents one domain
df = pd.concat([df_train, df_test],ignore_index=True)
    #df = df_train
dfs = [x for _, x in df.groupby('domain')]

# get the word distribution of each domain
# the frequency of each existing word is computed in every domain
import collections
import regex as re
word_counter = []
for df in dfs:
    counts = collections.Counter()
    words = re.compile(r'\w+')
    reviews = np.array([s for s in df['text']])
    for review in reviews:
        counts.update(words.findall(review.lower()))
    word_counter.append(counts)

# the rows of df are the 16 domains, the columns are all existing words
# the number of the cells of df is the word frequency for the word in the domain
df_dist = pd.DataFrame(word_counter)
df_dist = df_dist.fillna(0)
 


# get list js_d of jensen_shannon distances to the target domain
js_d = []
for i in range(df_dist.shape[0]):
    d = distance.jensenshannon(np.array(df_dist.iloc[index_spec]), np.array(df_dist.iloc[i]))
    js_d.append(d)
    
# take 5 most similiar distributions
# most_sim_dist is a list of 5 elements with the 5 closest domains to the target domain
most_sim_dist = sorted(range(len(js_d)), key=lambda i: js_d[i], reverse=True)[-5:]
most_sim_dist.remove(index_spec)
    #print(most_sim_dist)
# remove general embeddings from the training set that aren't from these 5 domains
index_to_keep_train = [index for index, value in enumerate(labels_train[1]) if int(value) in most_sim_dist]
labels_train, data_general_train = labels_train[:, index_to_keep_train], data_general_train[index_to_keep_train]
    #print(labels_train.shape, data_general_train.shape)

# remove general embeddings from test set that aren't from these 5 domains
index_to_keep_test = [index for index, value in enumerate(labels_test[1]) if int(value) in most_sim_dist]
labels_test, data_general_test = labels_test[:, index_to_keep_test], data_general_test[index_to_keep_test]
    #print(data_general, labels_general)    
    
# get indices for sorting the array
   # ind = sort_array(labels_general, labels_total)

# sort general sentence embeddings
    #data_general, labels_general = data_general[ind], labels_general[:, ind]

print(labels_train.shape, labels_total_train_val.shape)
ind_train = sort_array(labels_train, labels_total_train_val)
    #print(ind.shape)
# sort general sentence embeddings
data_general_train, labels_train = data_general_train[ind_train], labels_train[:, ind_train]

    #print(data_general_train.shape)
    
ind_test = sort_array(labels_test, labels_total_test)
    #print(ind.shape)
# sort general sentence embeddings
data_general_test, labels_test = data_general_test[ind_test], labels_test[:, ind_test]
# data splitting
    #print(X_train_spec.shape)
X_train_gen, X_val_gen = data_general_train[:4200], data_general_train[4200:]
    #print(data_general_train.shape, X_val_gen.shape)
   # y_train_gen, y_val_gen, y_test_gen = labels_general[:4200], labels_general[4200:4800], labels_general[4800:]
X_test = data_general_test[:400]


# In[395]:


class MyHyperModel(kt.HyperModel):
   def build(self, hp):
       INPUT_SIZE = 300
       LATENT_SIZE = 300
  # hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
   
       hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
       hp_units1 = hp.Int('units1', min_value=100, max_value=500, step=100)
      # hp_units2 = hp.Int('units2', min_value=60, max_value=130, step=10)
# domain-general model parts
       inp_gen = tf.keras.Input(shape=(1,INPUT_SIZE))
     #  inp_gen_att, attn_weights_gen = SeqSelfAttention(return_attention = True)(inp_gen)
       #out_gen = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hp_units1, input_shape=(None,1,INPUT_SIZE)))(inp_gen)
       #out_gen = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hp_units2, input_shape=(None,1,INPUT_SIZE)))(out_gen1)
# domain-specific model parts
       inp_spec = tf.keras.Input(shape=(1,INPUT_SIZE))
      # inp_spec_att, attn_weights_spec = SeqSelfAttention(return_attention = True)(inp_spec)
       #out_spec = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(300, input_shape=(None,1,INPUT_SIZE)))(inp_spec)
      # out_spec = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(120, input_shape=(None,1,INPUT_SIZE)))(out_spec1)
# concatenate domain-general and domain-specific results
       merged = tf.keras.layers.Concatenate()([inp_gen, inp_spec])
       merged = tf.keras.layers.Dense(hp_units1, activation='sigmoid')(merged)
# drop out layer and dense layer
       merged = tf.keras.layers.Dropout(hp.Float('dropout', 0, 0.5, step=0.1, default=0.5))(merged)
       merged = tf.keras.layers.Dense(1, activation='sigmoid')(merged)

       classifier = tf.keras.Model([inp_gen,inp_spec], merged)
       classifier.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate), metrics=['accuracy'])
#classifier.summary()
       return classifier
       

   def fit(self, hp, model, *args, **kwargs):
       return model.fit(
           *args,
           batch_size=hp.Choice("batch_size", values=[32, 64]),
           epochs= hp.Int('epochs', min_value=20, max_value=70, step=10),
           **kwargs,
       )


# In[396]:


tuner2=kt.BayesianOptimization(
    MyHyperModel(),
    objective="val_accuracy",
    max_trials=50,
    overwrite=True,
    num_initial_points=25,
    alpha=0.01,
    beta=2.6
    
)


# In[397]:


#NUM_EPOCHS = 20
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
#checkpoint = ModelCheckpoint(filepath="weights/classifier/classifier_without_al/standard_model/classifier_domain_1.h5")

tuner2.search([np.expand_dims(np.asarray(X_train_gen).astype(np.float32), 1), np.expand_dims(np.asarray(X_train_spec).astype(np.float32), 1)], np.asarray(y_train).astype(np.float32), validation_data = ([np.expand_dims(np.asarray(X_val_gen).astype(np.float32), 1), np.expand_dims(np.asarray(X_val_spec).astype(np.float32), 1)], np.asarray(y_val).astype(np.float32)), callbacks = [es])


# In[398]:


# Get the optimal hyperparameters
best_hps=tuner2.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. The optimal batch_size
layer is {best_hps.get('batch_size')}, the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}, the optimal dropout rate is {best_hps.get('dropout')}, the optimal number of epochs is {best_hps.get('epochs')} the optimal number of units1 is {best_hps.get('units1')} and th.
""")


# In[ ]:


#hp_units1 = hp.Int('units', min_value=200, max_value=350, step=10)
#hp_units2 = hp.Int('units', min_value=50, max_value=180, step=10)
# domain-general model parts
inp_gen = tf.keras.Input(shape=(1,300))
#inp_gen_att, attn_weights_gen = SeqSelfAttention(return_attention = True)(inp_gen)
#out_gen = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(300, input_shape=(None,1,INPUT_SIZE)))(inp_gen)
#out_gen = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(120, input_shape=(None,1,INPUT_SIZE)))(out_gen1)


# domain-specific model parts
inp_spec = tf.keras.Input(shape=(1,300))
#inp_spec_att, attn_weights_spec = SeqSelfAttention(return_attention = True)(inp_spec)
#out_spec = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(300, input_shape=(None,1,INPUT_SIZE)))(inp_spec)
#out_spec = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(120, input_shape=(None,1,INPUT_SIZE)))(out_spec1)


# concatenate domain-general and domain-specific results
merged = tf.keras.layers.Concatenate()([inp_gen, inp_spec])
#merged = tf.keras.layers.AveragePooling1D()([out_gen, out_spec])
merged = tf.keras.layers.Dense(300, activation='sigmoid')(merged)
# drop out layer and dense layer
#merged = tf.keras.layers.Dense(300, activation='sigmoid')(merged)
merged = tf.keras.layers.Dropout(.0)(merged)
merged = tf.keras.layers.Dense(1, activation='sigmoid')(merged)

classifier = tf.keras.Model([inp_gen,inp_spec], merged)
classifier.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.01), metrics=['accuracy'])


# In[ ]:


# training the model
#print(X_train_gen.shape, X_train_spec.shape, y_train.shape)
#classifier.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.01), metrics=['accuracy'])
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath="weights/classifier/classifier_without_al/standard_model/classifier_domain_6_13.h5")
history = classifier.fit([np.expand_dims(np.asarray(X_train_gen).astype(np.float32), 1), np.expand_dims(np.asarray(X_train_spec).astype(np.float32), 1)], np.asarray(y_train).astype(np.float32), epochs=20, validation_data = ([np.expand_dims(np.asarray(X_val_gen).astype(np.float32), 1), np.expand_dims(np.asarray(X_val_spec).astype(np.float32), 1)], np.asarray(y_val).astype(np.float32)), callbacks = [checkpoint, es], batch_size=64)

# evaluating the model
score = classifier.evaluate([np.expand_dims(np.asarray(X_test_gen).astype(np.float32), 1), np.expand_dims(np.asarray(X_test_spec).astype(np.float32), 1)], np.asarray(y_test).astype(np.float32), verbose=1) 
print('Final accuracy score: '+str(score[1]))


# In[ ]:




