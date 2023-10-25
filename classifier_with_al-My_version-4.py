#!/usr/bin/env python
# coding: utf-8

# # Active Learning and Classification

# This file produces all the active learning results of the evaluation chapter. All cells needs to be executed until the headline "Executing Active Learning and training the model" is reached. Then, a specific model is chosen and executed. Note: Make sure to select the desired target domain when loading the data below.

# Make sure to adjust the checkpoint paths when training the models such that the weights are saved in the desired paths.

# ## Importing libraries and setting configurations

# In[4]:


# set the target domain
index_spec = 2


# In[5]:


# imports
# it's possible that a deprecation warning appears here for unmath_test, however unrelevant for this work
#from keras.models import Model
#from keras.layers.recurrent import LSTM
#from keras.callbacks import EarlyStopping
#from keras.callbacks import ModelCheckpoint
#from keras.layers.wrappers import Bidirectional
from keras_self_attention import SeqSelfAttention
#from keras.layers import Input, Dense, Concatenate, Dropout

import os
import numpy as np
import numpy as np1
import random as rn
import pickle as pkl
from random import randint
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.covariance import EllipticEnvelope
import tensorflow as tf
import re
import glob
import pandas as pd
import random as rn
    
from collections import Counter
import matplotlib.pyplot as plt
from scipy.spatial import distance


# In[969]:


# setting seeds in order to reproduce the results
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(1)
rn.seed(2)
tf.random.set_seed(3)


# configurations so we use a single thread
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)


# ## The necessary functions

# Make sure to load the data of the desired target domain here:

# In[2]:


# function for sorting two arrays such that both arrays have the same labels
def sort_array(labels, labels_ref):
    index_gen_zeros = []
    index_gen_ones = []
    
    # get indices when the general label is 0 (index_gen_zeros) and when it is 1 (index_gen_ones)
    for i in np.arange(len(list(labels))):
        if int(labels[i]) == 0:
            index_gen_zeros.append(i)
        if int(labels[i]) == 1: 
            index_gen_ones.append(i)
    cnt_0, cnt_1 = 0,0
    result_ind = np.zeros(len(list(labels_ref)))
    
    # sort array such that the first positive specific embedding gets paired with the first positive general embedding
    # as well as the first negative specific embeddings gets paired with the first negative general embedding
    for i in np.arange(len(list(labels_ref))):
        if int(labels_ref[i]) == 0:
            result_ind[i] = int(index_gen_zeros[cnt_0])
            cnt_0 = cnt_0+1
        else:
            result_ind[i] = int(index_gen_ones[cnt_1])
            cnt_1 = cnt_1+1
    
    return result_ind.astype(int)

def sort_array2(array_to_sort, array_ref):
    
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

# function for removing outliers
# choose parameter "algorithm" as 0 for no outlier detection, 1 for elliptic envelope
# and choose 2 for isolation forest
def outlier_removal(X, y, algorithm):

    # no outlier removal
    if algorithm == 0:
        outlier_removal = None
        
    # outlier removal
    else: 
        if algorithm == 1:
            outlier_removal = EllipticEnvelope(support_fraction=0.9, random_state = 2)
        elif algorithm == 2:
            outlier_removal = IsolationForest(random_state = 3)
        print(X)
        # fit and predict
        outlier_removal.fit(X)
        y_hat = outlier_removal.predict(X)

        # select all rows that are not outliers
        mask = y_hat != -1

        # remove outliers
        X, y = X[mask, :], y[mask]
    

    return X, y

# function for active learning
# parameter data and labels contain original train set
# parameter X_test and y_test are original validation and test set stacked
# max_query denotes the number of instances queried per iteration
# uncertainty sampling binarily encodes uncertainty sampling or certainty sampling
# outlier detection indicates the outlier detection technique
def AL(data_gen, data_spec, labels_gen, labels_spec, X_val_gen, X_test_gen, X_val_spec, X_test_spec, y_val_gen, y_test_gen, y_val_spec, y_test_spec, max_query1, max_query2, uncertainty_sampling, outlier_detection):
    
    # remove outliers, choose outlier detection algorithm (parameter "algo")
    data_gen, labels_gen = outlier_removal(data_gen, labels_gen, outlier_detection)
    data_spec, labels_spec = outlier_removal(data_spec, labels_spec, outlier_detection)
    # initializations
    k = 100
     
    continue_al = True
    X_train_gen, X_valid_gen, y_train_gen, y_valid_gen = data_gen[:150], data_gen[150:], labels_gen[:150], labels_gen[150:] 
    X_train_spec, X_valid_spec, y_train_spec, y_valid_spec = data_spec[:100], data_spec[100:], labels_spec[:100], labels_spec[100:] 
   
    
  
    
    #CLASSIFIER
    
    INPUT_SIZE = 300
    LATENT_SIZE = 300

# domain-general model parts
    inp_gen = tf.keras.Input(shape=(1,INPUT_SIZE))
    inp_spec = tf.keras.Input(shape=(1,INPUT_SIZE))
    merged = tf.keras.layers.Concatenate()([inp_gen, inp_spec])
    merged = tf.keras.layers.Dense(100, activation='sigmoid')(merged)
   # merged = tf.keras.layers.Dense(100, activation='sigmoid')(merged)
# drop out layer and dense layer
    merged = tf.keras.layers.Dropout(.5)(merged)
    merged = tf.keras.layers.Dense(1, activation='sigmoid')(merged)

    classifier = tf.keras.Model([inp_gen,inp_spec], merged)
    #classifier.summary()
     
    ind4 = sort_array(y_test_gen, y_test_spec)
    X_test_gen, y_test_gen = X_test_gen[ind4], y_test_spec
    #print(X_test_gen1.shape, y_test_gen1.shape, y_test_spec.shape, y_test_spec.shape)
    
    ind5 = sort_array(y_val_gen, y_val_spec)
    X_val_gen, y_val_gen = X_val_gen[ind5], y_val_spec
    #print(X_val_gen1.shape, y_val_gen1.shape, X_val_spec.shape, y_val_spec.shape)
    # apply AL on specific and general sentence embeddings
# make sure to pick the right parameters for uncertainty_sampling and outlier_detection 
# SIDENOTE: for certainty sampling experiment change max_query to 2200 (for domain 4 even to 3000) for general embeddings - due to different number of positive/negative samples
  
    


    # start active learning loop, execute loops until stopping criteria is fulfilled
    while continue_al:  
              
        
       # print(y_train_gen.shape, y_train_spec.shape)
        ind2 = sort_array(y_train_gen, y_train_spec)
        X_train_gen1, y_train1 = X_train_gen[ind2], y_train_spec
        #print(y_train1.shape)
    #THE LSTM Classifier
        #print(y_valid_gen.shape, y_valid_spec.shape)
        ind3 = sort_array(y_valid_gen, y_valid_spec)
        X_valid_gen1, y_valid_gen1 = X_valid_gen[ind3], y_valid_spec
        #print(y_test_gen.shape, y_test_spec.shape)
        
        
        # scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_train_gen1 = scaler.fit_transform(X_train_gen1)
        X_valid_gen1 = scaler.transform(X_valid_gen1)
        X_test_gen = scaler.transform(X_test_gen)
        X_val_gen = scaler.transform(X_val_gen)
        
        X_train_spec = scaler.fit_transform(X_train_spec)
        X_valid_spec = scaler.transform(X_valid_spec)
        X_test_spec = scaler.transform(X_test_spec)
        X_val_spec = scaler.transform(X_val_spec)
        # fit the classifier on training data
        #svclassifier.fit(X_train, y_train)
        
        # compute performance measure (using test data)
       # y_pred = svclassifier.predict(X_test)
        
    # training the model
        classifier.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.01), metrics=['accuracy'])
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath="weights/classifier/classifier_with_al/certainty_sampling/classifier_domain_6.h5")
        #print(X_train_gen1.shape, X_train_spec.shape,y_train1.shape)
        history = classifier.fit([np.expand_dims(np.asarray(X_train_gen1).astype(np.float32), 1), np.expand_dims(np.asarray(X_train_spec).astype(np.float32), 1)], np.asarray(y_train1).astype(np.float32), epochs=20,validation_data = ([np.expand_dims(np.asarray(X_val_gen).astype(np.float32), 1), np.expand_dims(np.asarray(X_val_spec).astype(np.float32), 1)], np.asarray(y_val_spec).astype(np.float32)), callbacks = [checkpoint, es], batch_size=64)

# evaluating the model
       # y_pred = classifier.predict([np.expand_dims(X_test_gen, 1), np.expand_dims(X_test_spec, 1)], verbose=0) 
        y_pred = classifier.predict([np.expand_dims(np.asarray(X_test_gen).astype(np.float32), 1), np.expand_dims(np.asarray(X_test_spec).astype(np.float32), 1)], verbose=0)
        #print(y_pred)
        y_pred = np.where(y_pred >= 0.5, 1,0)
        
        #print(y_pred)
        y_pred = np.squeeze(y_pred)
        #print(X_valid_gen1.shape)
        get_prob = classifier.predict([np.expand_dims(np.asarray(X_valid_gen1).astype(np.float32), 1), np.expand_dims(np.asarray(X_valid_spec).astype(np.float32), 1)], verbose=0) 
       # print(get_prob)
        
        get_prob = np.squeeze(get_prob)
       # print(get_prob)
        #print( 'Final accuracy score: '+str(y_pred_valid[1]))
        #print(y_test_spec, y_pred)
       # f1 = f1_score(y_test_spec, y_pred, average='macro')
        y_pred_valid = classifier.predict([np.expand_dims(np.asarray(X_valid_gen1).astype(np.float32), 1), np.expand_dims(np.asarray(X_valid_spec).astype(np.float32), 1)], verbose=0) 
        #print(y_pred_valid)
        y_pred_valid = np.where(y_pred >= 0.5, 1,0)
        y_pred_valid = np.squeeze(y_pred_valid)
        
        #print(y_test, y_pred)
        # compute uncertainty metric (using validation data)
        #classifier_export = classifier.export_model()
        #model_export.predict -> return probability
        #get_prob = classifier_export_predict([np.expand_dims(X_test_gen, 1), np.expand_dims(X_test_spec, 1)], y_test_spec, verbose=0) 
        #y_pred_valid = classifier.predict(X_valid)
        #prob0 = []
        #prob1=[]
        prob = []
        #print(prob)
        #print(get_prob.shape)
        #print(y_pred_valid.shape)
        #print(get_prob.shape)
        for i in range(get_prob.shape[0]):
            #prob.append((-get_prob[i] * np.log2(get_prob[i])).sum(axis=1))
            #print(get_prob[i])
            if get_prob[i]<0.5:
                prob.append(1-get_prob[i])
            else:    
                prob.append(get_prob[i])
   
            #print(prob)
        
        if uncertainty_sampling == True:
           # prob = filter(lambda x: x <= 0.7, prob)
            #prob2 = filter(lambda x: x <= 1, prob)
            #print(prob2)
            #prob = [x for x in prob if x < 0.8]
            
            ind = np.argsort(prob)[:k]
            #ind1 = np.argsort(prob1)[:100]
            #ind = ind0 + ind1
           # print(ind0, ind1)
            #print(ind)
            #for i in ind:
             #   print(prob[i])
            #ind = np.argsort(prob)[:k]
        else:
            ind = np.argsort(prob)[-k:]
        #print(y_train.shape,y_valid_gen[ind].shape)
        # move samples from the validation set to the training set
        X_train_gen = np.vstack((X_train_gen1,X_valid_gen1[ind]))
        y_train_gen = np.hstack((y_train1, y_valid_gen1[ind])) 
       # print(X_valid_gen1.shape)
        X_valid_gen = np.delete(X_valid_gen1, ind, axis = 0)      
        y_valid_gen = np.delete(y_valid_gen1, ind, axis = 0)  
        #print(X_valid_gen.shape)
        X_train_spec = np.vstack((X_train_spec,X_valid_spec[ind]))
        y_train_spec = np.hstack((y_train_spec, y_valid_spec[ind]))     
        X_valid_spec = np.delete(X_valid_spec, ind, axis = 0)      
        y_valid_spec = np.delete(y_valid_spec, ind, axis = 0) 
        
        # inverse scaling
        X_train_gen = scaler.inverse_transform(X_train_gen) 
        X_valid_gen = scaler.inverse_transform(X_valid_gen)
        X_test_gen = scaler.inverse_transform(X_test_gen)
        X_val_gen =scaler.inverse_transform(X_val_gen)
          # inverse scaling
        X_train_spec = scaler.inverse_transform(X_train_spec) 
        X_valid_spec = scaler.inverse_transform(X_valid_spec)
        X_test_spec = scaler.inverse_transform(X_test_spec)
        X_val_spec =scaler.inverse_transform(X_val_spec)
        
        # check stopping criteria
        print( X_train_spec.shape[0])
        #print(y_train_gen, y_train_spec)
        if X_train_spec.shape[0] > max_query1:
            continue_al = False
            
   # print(X_test_gen.shape, y_test_gen.shape, X_test_spec.shape, y_test_spec.shape)    
   # print(X_val_gen.shape, y_val_gen.shape, X_val_spec.shape, y_val_spec.shape) 
    return X_train_gen,X_train_spec, y_train_spec, X_val_gen, X_val_spec, y_val_spec, X_test_gen, X_test_spec, y_test_spec


# # Loading the split data

# In[3]:


# importing the data for the general sentence embeddings, here corresponding data from domain 0 was chosen
def return_results_AL(k, par0, par1): 
    # set the target domain
    index_spec = k
    print(k)
    with open('data/sentence_embeddings/general/sorted/train/train_data6_'+str(k)+'.p', 'rb') as f:
        X_train_gen_all = pkl.load(f)
    with open('data/sentence_embeddings/general/sorted/train/train_labels6_'+str(k)+'.p', 'rb') as f:
        y_train_gen_all = pkl.load(f)
    with open('data/sentence_embeddings/general/sorted/val_test/vt_data6_'+str(k)+'.p', 'rb') as f:
        X_val_test_spec = pkl.load(f)
    with open('data/sentence_embeddings/general/sorted/val_test/vt_labels6_'+str(k)+'.p', 'rb') as f:
        y_val_test = pkl.load(f)   
    labels_total = np1.hstack((y_train_gen_all[:,:4200].astype(int), y_val_test))
    X_val_gen, X_test_gen = X_val_test_spec[:600], X_val_test_spec[600:]
    y_train_gen_all = y_train_gen_all[0,:]
    y_train, y_val, y_test = y_train_gen_all[:4200], y_val_test[0,:600], y_val_test[0,600:]

# import the data from the specific sentence embeddings, here corresponding data from domain 0 was chosen
    with open('data/sentence_embeddings/specific/sentemb/sentemb_unlabeled6_'+str(k)+'.p', 'rb') as f:
        X_spec = pkl.load(f)   
    X_spec=np.repeat(X_spec,repeats=3, axis=1)
    X_train_spec, X_val_spec, X_test_spec = X_spec.transpose()[:4200], X_spec.transpose()[4200:4800], X_spec.transpose()[4800:
   
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
    labels_general = labels_general[0,:]
    labels_general = labels_general.transpose()
# data splitting
    X_train_gen, X_val_gen, X_test_gen = data_general[:5000], data_general[5000:6000], data_general[6000:]
    y_train_gen, y_val_gen, y_test_gen = labels_general[:5000], labels_general[5000:6000], labels_general[6000:]
                                                                                                                         
    #THE LSTM Classifier
    INPUT_SIZE = 300
    LATENT_SIZE = 300

# domain-general model parts
    inp_gen = tf.keras.Input(shape=(1,INPUT_SIZE))
# domain-specific model parts
    inp_spec = tf.keras.Input(shape=(1,INPUT_SIZE))
# concatenate domain-general and domain-specific results
    merged = tf.keras.layers.Concatenate()([inp_gen, inp_spec])
    merged = tf.keras.layers.Dense(100, activation='sigmoid')(merged)
    merged = tf.keras.layers.Dropout(.5)(merged)
    merged = tf.keras.layers.Dense(1, activation='sigmoid')(merged)
    classifier = tf.keras.Model([inp_gen,inp_spec], merged)

    # apply AL on specific and general sentence embeddings
# make sure to pick the right parameters for uncertainty_sampling and outlier_detection 
# SIDENOTE: for certainty sampling experiment change max_query to 2200 (for domain 4 even to 3000) for general embeddings - due to different number of positive/negative samples 
    classifier.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.01), metrics=['accuracy'])
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath="weights/classifier/classifier_with_al/certainty_sampling/classifier_domain_6.h5")
    history = classifier.fit([np.expand_dims(np.asarray(X_train_gen_al).astype(np.float32), 1), np.expand_dims(np.asarray(X_train_spec_al).astype(np.float32), 1)], np.asarray(y_train_spec_al).astype(np.float32), epochs=20, validation_data = ([np.expand_dims(np.asarray(X_val_gen2).astype(np.float32), 1), np.expand_dims(np.asarray(X_val_spec).astype(np.float32), 1)], np.asarray(y_val).astype(np.float32)), callbacks = [checkpoint, es], batch_size=64)

# evaluating the model
    score = classifier.evaluate([np.expand_dims(np.asarray(X_test_gen2).astype(np.float32), 1), np.expand_dims(np.asarray(X_test_spec).astype(np.float32), 1)], np.asarray(y_test).astype(np.float32), verbose=0) 
    return(k, pars[0], pars[1], 'Final accuracy score: '+str(score[1]))


# In[4]:


class TestCase:
    def __init__(self, name, i_range, pars):
        self.name = name
        self.i_range = i_range
        self.pars = pars
        
test_cases = [
    TestCase("peirama 1", range(0,16), [(1400, 2000)]),
#     TestCase("peirama 2", range(0,16), [(1400, 2200), (2100, 3000)]),
   # TestCase("peirama 3", range(0,100), [(1400, 2200), (2100, 3000)]),
]

for test_case in test_cases:
    for i in test_case.i_range:
        for pars in test_case.pars:
            print(test_case.name, i, pars)
            x = return_results_AL(i, pars[0], pars[1])
            print(x)


# In[ ]:


# plotting the results
import matplotlib.pyplot as plt
#from tf.keras.utils import plot_model

tf.keras.utils.plot_model(model, show_shapes=True, to_file='TRIAL_2_AL_full.png')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()
plt.plot(epochs, loss, 'ro', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()


# In[805]:



# importing the data for the general sentence embeddings, here corresponding data from domain 0 was chosen
def return_results_AL(k, par0, par1): 
    # set the target domain
    index_spec = k
    print(k)
    with open('data/sentence_embeddings/general/sorted/train/train_data6_'+str(k)+'.p', 'rb') as f:
        X_train_gen_all = pkl.load(f)

    with open('data/sentence_embeddings/general/sorted/train/train_labels6_'+str(k)+'.p', 'rb') as f:
        y_train_gen_all = pkl.load(f)

    with open('data/sentence_embeddings/general/sorted/val_test/vt_data6_'+str(k)+'.p', 'rb') as f:
        X_val_test_spec = pkl.load(f)

    with open('data/sentence_embeddings/general/sorted/val_test/vt_labels6_'+str(k)+'.p', 'rb') as f:
        y_val_test = pkl.load(f)
    #print(y_train_gen_all[:,:4200])
    #labels_total = np1.hstack((y_train_gen_all[:,:4200], y_val_test))
    
    labels_total = np1.hstack((y_train_gen_all[:,:4200].astype(int), y_val_test))
    X_val_gen, X_test_gen = X_val_test_spec[:600], X_val_test_spec[600:]
    y_train_gen_all = y_train_gen_all[0,:]
    y_train, y_val, y_test = y_train_gen_all[:4200], y_val_test[0,:600], y_val_test[0,600:]


# import the data from the specific sentence embeddings, here corresponding data from domain 0 was chosen
    with open('data/sentence_embeddings/specific/sentemb/sentemb_unlabeled6_'+str(k)+'.p', 'rb') as f:
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
    #print(labels_general, labels_total)
    #labels_general.dtype
    #print(labels_total.shape)
    #print(labels_general.shape)
    #print(labels_general.shape)
   # ind = sort_array2(labels_general, labels_total)
    #print(labels_general.shape)
    #print(ind.shape)
# sort general sentence embeddings
    #print(labels_general.shape)
    #data_general, labels_general = data_general[ind], labels_general[:, ind]
    print(labels_general)
    labels_general = labels_general[0,:]
    labels_general = labels_general.transpose()
    #print(labels_general)
# data splitting
    X_train_gen, X_val_gen, X_test_gen = data_general[:5000], data_general[5000:6000], data_general[6000:7700]
   # print(data_general, labels_general)
    y_train_gen, y_val_gen, y_test_gen = labels_general[:5000], labels_general[5000:6000], labels_general[6000:7700]
    
# data splitting
    #print( X_train_gen.shape, y_train_gen.shape, np.vstack((X_val_gen,X_test_gen)).shape, np.hstack((y_val_gen,y_test_gen)).shape)
    #X_train_spec_al, y_train_spec_al = AL(X_train_spec, y_train, np.vstack((X_val_spec,X_test_spec)), np.hstack((y_val,y_test)), pars[0], True, 2)
    #X_train_gen_al, y_train_gen_al = AL( X_train_gen, y_train_gen , np.vstack((X_val_gen1,X_test_gen1)), np.hstack((y_val_gen1,y_test_gen1)), pars[1], True, 2)
    X_train_gen_al,X_train_spec_al, y_train_gen_al, y_train_spec_al  = AL( X_train_gen, X_train_spec, y_train_gen, y_train, np.vstack((X_val_gen,X_test_gen)), np.vstack((X_val_spec,X_test_spec)), np.hstack((y_val_gen,y_test_gen)), np.hstack((y_val_gen,y_test_gen)),np.hstack((y_val,y_test)),pars[0], pars[1], True, 2)
    #print(X_train_gen_al.shape,X_train_spec_al.shape, y_train_gen_al.shape, y_train_spec_al)
    #print( X_train_gen_al.shape, y_train_spec_al.shape)
    #print(y_train_gen_al.shape, y_train_spec_al.shape)
    ind = sort_array(y_train_gen_al, y_train_spec_al)
    #print( X_train_gen_al.shape, y_train_spec_al.shape)
    X_train_gen_al, y_train = X_train_gen_al[ind], y_train_spec_al
    #y_train=y_train_spec_al
    
    ind2 = sort_array(y_val_gen, y_val)
    #print( X_train_gen_al.shape, y_train_spec_al.shape)
    X_val_gen = X_val_gen[ind2]

    ind3 = sort_array(y_test_gen, y_test)
    X_test_gen = X_test_gen[ind3]


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
    #out_spec = tf.keras.layers.Dense(300, activation='sigmoid')(inp_spec)
#inp_spec_att, attn_weights_spec = SeqSelfAttention(return_attention = True)(inp_spec)
   # out_spec = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(LATENT_SIZE, input_shape=(None,1,INPUT_SIZE)))(inp_spec)
   # out_spec, attn_weights_spec = SeqSelfAttention(return_attention = True)(out_spec1)
# concatenate domain-general and domain-specific results
    merged = tf.keras.layers.Concatenate()([inp_gen, inp_spec])
    merged = tf.keras.layers.Dense(300, activation='sigmoid')(merged)


    #merged = tf.keras.layers.Dense(100, activation='sigmoid')(merged)
    #merged = tf.keras.layers.Dense(100, activation='sigmoid')(merged)
# drop out layer and dense layer
    merged = tf.keras.layers.Dropout(.2)(merged)
    merged = tf.keras.layers.Dense(1, activation='sigmoid')(merged)

    classifier = tf.keras.Model([inp_gen,inp_spec], merged)
    #classifier.summary()
    
    
    
    
    # apply AL on specific and general sentence embeddings
# make sure to pick the right parameters for uncertainty_sampling and outlier_detection 
# SIDENOTE: for certainty sampling experiment change max_query to 2200 (for domain 4 even to 3000) for general embeddings - due to different number of positive/negative samples
  
    
    #print(X_train_gen_al.shape, X_train_spec_al.shape, y_train.shape, X_val_gen.shape, X_val_spec.shape, y_val.shape)
    # training the model
    classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath="weights/classifier/classifier_with_al/certainty_sampling/classifier_domain_3.h5")
    history = classifier.fit([np.expand_dims(X_train_gen_al, 1), np.expand_dims(X_train_spec_al, 1)], y_train, epochs=50, validation_data = ([np.expand_dims(X_val_gen, 1), np.expand_dims(X_val_spec, 1)], y_val), callbacks = [checkpoint, es], batch_size=32)

# evaluating the model
    print(X_test_gen.shape, X_test_spec.shape, y_test.shape)
    score = classifier.evaluate([np.expand_dims(X_test_gen, 1), np.expand_dims(X_test_spec, 1)], y_test, verbose=0) 
    print(k)
    return(k, pars[0], pars[1], 'Final accuracy score: '+str(score[1]))


# In[648]:


import numpy as np1
# importing the data for the general sentence embeddings, here corresponding data from domain 0 was chosen
def return_results_AL(k): 
    # set the target domain
    index_spec = k
    print(k)
    with open('data/sentence_embeddings/general/sorted/train/train_data8_'+str(k)+'.p', 'rb') as f:
        X_train_gen_all = pkl.load(f)

    with open('data/sentence_embeddings/general/sorted/train/train_labels8_'+str(k)+'.p', 'rb') as f:
        y_train_gen_all = pkl.load(f)

    with open('data/sentence_embeddings/general/sorted/val_test/vt_data8_'+str(k)+'.p', 'rb') as f:
        X_val_test_spec = pkl.load(f)

    with open('data/sentence_embeddings/general/sorted/val_test/vt_labels8_'+str(k)+'.p', 'rb') as f:
        y_val_test = pkl.load(f)

    labels_total = np1.hstack((y_train[:,:4200], y_val_test))
    #labels_total = np.hstack((y_train_gen_all[:,:4200], y_val_test))
    X_val_gen, X_test_gen = X_val_test_spec[:600], X_val_test_spec[600:]
    y_train_gen_all = y_train_gen_all[0,:]
    y_train, y_val, y_test = y_train_gen_all[:4200], y_val_test[0,:600], y_val_test[0,600:]

# import the data from the specific sentence embeddings, here corresponding data from domain 0 was chosen
    with open('data/sentence_embeddings/specific/sentemb/sentemb_unlabeled8_'+str(k)+'.p', 'rb') as f:
        X_spec = pkl.load(f)
    
#X_train_spec, X_val_spec, X_test_spec = X_spec[:1400], X_spec[1400:1600], X_spec[1600:2000] 


    X_spec=np.repeat(X_spec,repeats=3, axis=1)



    X_train_spec, X_val_spec, X_test_spec = X_spec.transpose()[:4200], X_spec.transpose()[4200:4800], X_spec.transpose()[4800:]
    
    

    # load the original, unsorted data
    with open('data/sentence_embeddings/general/unsorted/sentemb/sentemb_unlabeled8.p', 'rb') as f:
        data_general = pkl.load(f)

    with open('data/sentence_embeddings/general/unsorted/label_domain/label_domain_train_sentemb_unlabeled8.p', 'rb') as f:
        labels_train = pkl.load(f)
    
    with open('data/sentence_embeddings/general/unsorted/label_domain/label_domain_test_sentemb_unlabeled8.p', 'rb') as f:
        labels_test = pkl.load(f)
    
    labels_general = np1.hstack((labels_train, labels_test))

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
    # function for sorting two arrays such that both arrays have the same labels
# returns indeces_sorted which consists of indices and is used for sorting array_to_sort
  
    
      

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

# data splitting
    X_train_gen, X_val_gen, X_test_gen = data_general[:4200], data_general[4200:4800], data_general[4800:]






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
    #out_spec = tf.keras.layers.Dense(300, activation='sigmoid')(inp_spec)
#inp_spec_att, attn_weights_spec = SeqSelfAttention(return_attention = True)(inp_spec)
   # out_spec = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(LATENT_SIZE, input_shape=(None,1,INPUT_SIZE)))(inp_spec)
   # out_spec, attn_weights_spec = SeqSelfAttention(return_attention = True)(out_spec1)
# concatenate domain-general and domain-specific results
    merged = tf.keras.layers.Concatenate()([inp_gen, inp_spec])
    merged = tf.keras.layers.Dense(300, activation='sigmoid')(merged)


    #merged = tf.keras.layers.Dense(100, activation='sigmoid')(merged)
    #merged = tf.keras.layers.Dense(100, activation='sigmoid')(merged)
# drop out layer and dense layer
    merged = tf.keras.layers.Dropout(.4)(merged)
    merged = tf.keras.layers.Dense(1, activation='sigmoid')(merged)

    classifier = tf.keras.Model([inp_gen,inp_spec], merged)
    #classifier.summary()
    
    
    
    
    
    # apply AL on specific and general sentence embeddings
# make sure to pick the right parameters for uncertainty_sampling and outlier_detection 
# SIDENOTE: for certainty sampling experiment change max_query to 2200 (for domain 4 even to 3000) for general embeddings - due to different number of positive/negative samples
  
    
    
  # training the model
    classifier.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.001), metrics=['accuracy'])
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath="weights/classifier/classifier_with_al/random/classifier_domain3_15.h5")
    history = classifier.fit([np.expand_dims(X_train_gen[:1400], 1), np.expand_dims(X_train_spec[:1400], 1)], y_train[:1400], epochs=30, validation_data = ([np.expand_dims(X_val_gen, 1), np.expand_dims(X_val_spec, 1)], y_val), callbacks = [checkpoint, es], batch_size=32)

# evaluating the model
    score = classifier.evaluate([np.expand_dims(X_test_gen, 1), np.expand_dims(X_test_spec, 1)], y_test, verbose=0) 
    print('Final accuracy score: '+str(score[1]))
    return(k, 'Final accuracy score: '+str(score[1]))


# In[806]:


class TestCase:
   def __init__(self, name, i_range):
       self.name = name
       self.i_range = i_range
      # self.pars = pars
       
test_cases = [
   TestCase("peirama 1", range(0,16)),
#     TestCase("peirama 1", range(0,16), [(1400, 2200), (2100, 3000)]),
  # TestCase("peirama 2", range(0,16), [1400, 2200])
  # TestCase("peirama 3", range(0,100), [(1400, 2200), (2100, 3000)]),
]

for test_case in test_cases:
   for i in test_case.i_range:
      # for pars in test_case.pars:
       print(test_case.name, i)
       x = return_results_AL(i)
       print(x)


# In[2]:





# importing the data for the general sentence embeddings, here corresponding data from domain 0 was chosen
def return_results_AL(i, par0, par1): 
    with open('data/sentence_embeddings/general/sorted/train/train_data3_'+str(i)+'.p', 'rb') as f:
        X_train_gen_all = pkl.load(f)

    with open('data/sentence_embeddings/general/sorted/train/train_labels3_'+str(i)+'.p', 'rb') as f:
        y_train_gen_all = pkl.load(f)

    with open('data/sentence_embeddings/general/sorted/val_test/vt_data3_'+str(i)+'.p', 'rb') as f:
        X_val_test_spec = pkl.load(f)

    with open('data/sentence_embeddings/general/sorted/val_test/vt_labels3_'+str(i)+'.p', 'rb') as f:
        y_val_test = pkl.load(f)

    X_val_gen, X_test_gen = X_val_test_spec[:600], X_val_test_spec[600:]
    y_train_gen_all = y_train_gen_all[0,:]
    y_train, y_val, y_test = y_train_gen_all[:4200], y_val_test[0,:600], y_val_test[0,600:]


# import the data from the specific sentence embeddings, here corresponding data from domain 0 was chosen
    with open('data/sentence_embeddings/specific/sentemb/sentemb_unlabeled3_'+str(i)+'.p', 'rb') as f:
        X_spec = pkl.load(f)
    
#X_train_spec, X_val_spec, X_test_spec = X_spec[:1400], X_spec[1400:1600], X_spec[1600:2000] 

    import numpy as np
    X_spec=np.repeat(X_spec,repeats=3, axis=1)


    X_train_spec, X_val_spec, X_test_spec = X_spec.transpose()[:4200], X_spec.transpose()[4200:4800], X_spec.transpose()[4800:]
    
    
 #THE LSTM Classifier


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
    #out_spec = tf.keras.layers.Dense(300, activation='sigmoid')(inp_spec)
#inp_spec_att, attn_weights_spec = SeqSelfAttention(return_attention = True)(inp_spec)
   # out_spec = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(LATENT_SIZE, input_shape=(None,1,INPUT_SIZE)))(inp_spec)
   # out_spec, attn_weights_spec = SeqSelfAttention(return_attention = True)(out_spec1)
# concatenate domain-general and domain-specific results
    merged = tf.keras.layers.Concatenate()([inp_gen, inp_spec])
    merged = tf.keras.layers.Dense(300, activation='sigmoid')(merged)


    #merged = tf.keras.layers.Dense(100, activation='sigmoid')(merged)
    #merged = tf.keras.layers.Dense(100, activation='sigmoid')(merged)
# drop out layer and dense layer
    merged = tf.keras.layers.Dropout(.0)(merged)
    merged = tf.keras.layers.Dense(1, activation='sigmoid')(merged)

    classifier = tf.keras.Model([inp_gen,inp_spec], merged)
    #classifier.summary()

    classifier = tf.keras.Model([inp_gen,inp_spec], merged)
    #classifier.summary()
    
    #X_train_spec_al, y_train_spec_al = AL(X_train_spec, y_train, np.vstack((X_val_spec,X_test_spec)), np.hstack((y_val,y_test)), 1400, True, 2)
    X_train_gen_al,  X_train_spec_al, y_train_gen_al, y_train_spec_al = AL(X_train_gen_all, X_train_spec, y_train_gen_all, y_train, np.vstack((X_val_gen,X_test_gen)), np.vstack((X_val_spec,X_test_spec)), np.hstack((y_val,y_test)), np.hstack((y_val,y_test)), pars[0], pars[1], True, 2)
    # sort general sentence embeddings so that general and specific sentence embeddings have the same labels 
# and number of instances
    ind = sort_array(y_train_gen_al, y_train_spec_al)
    X_train_gen_al, y_train = X_train_gen_al[ind], y_train_spec_al
    
    # apply AL on specific and general sentence embeddings
# make sure to pick the right parameters for uncertainty_sampling and outlier_detection 
# SIDENOTE: for certainty sampling experiment change max_query to 2200 (for domain 4 even to 3000) for general embeddings - due to different number of positive/negative samples
   
    
    
# training the model
   # classifier.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.001), metrics=['accuracy'])
   # es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
   # checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath="weights/classifier/classifier_with_al/random/classifier_domain3_15.h5")
   # history = classifier.fit([np.expand_dims(X_train_gen_all[:pars[0]], 1), np.expand_dims(X_train_spec[:pars[0]], 1)], y_train[:pars[0]], epochs=30, validation_data = ([np.expand_dims(X_val_gen, 1), np.expand_dims(X_val_spec, 1)], y_val), callbacks = [checkpoint, es], batch_size=32)

# evaluating the model
   # score = classifier.evaluate([np.expand_dims(X_test_gen, 1), np.expand_dims(X_test_spec, 1)], y_test, verbose=0) 
   # print('Final accuracy score: '+str(score[1]))
   # return(i, 'Final accuracy score: '+str(score[1]))



# training the model
    classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath="weights/classifier/classifier_with_al/certainty_sampling/classifier_domain_3.h5")
    history = classifier.fit([np.expand_dims(X_train_gen_al, 1), np.expand_dims(X_train_spec_al, 1)], y_train, epochs=50, validation_data = ([np.expand_dims(X_val_gen, 1), np.expand_dims(X_val_spec, 1)], y_val), callbacks = [checkpoint, es], batch_size=32)

# evaluating the model
    score = classifier.evaluate([np.expand_dims(X_test_gen, 1), np.expand_dims(X_test_spec, 1)], y_test, verbose=0) 
    return('Final accuracy score: '+str(score[1]))


# In[3]:


class TestCase:
    def __init__(self, name, i_range, pars):
        self.name = name
        self.i_range = i_range
        self.pars = pars
        
test_cases = [
    TestCase("peirama 1", range(0,16), [(1400, 2000)]),
#     TestCase("peirama 1", range(0,16), [(1400, 2200), (2100, 3000)]),
#     TestCase("peirama 2", range(0,16), [(1400, 2200), (2100, 3000)]),
   # TestCase("peirama 3", range(0,100), [(1400, 2200), (2100, 3000)]),
]

for test_case in test_cases:
    for i in test_case.i_range:
        for pars in test_case.pars:
            print(test_case.name, i, pars)
            x = return_results_AL(i, pars[0], pars[1])
            print(x)
           # data = pd.DataFrame(x)
#data.to_excel('sample_data.xlsx', sheet_name='sheet1', index=False)


# ## Initializing the model

# ## Executing Active Learning and training the model

# In the following cell, a model is trained on 420 random samples as comparison. If a specific active learning algorithm is desired to be executed, this section can be skipped.

# In[312]:


# training the model
classifier.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.001), metrics=['accuracy'])
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath="weights/classifier/classifier_with_al/random/classifier_domain3_15.h5")
history = classifier.fit([np.expand_dims(X_train_gen_all[:1400], 1), np.expand_dims(X_train_spec[:1400], 1)], y_train[:1400], epochs=30, validation_data = ([np.expand_dims(X_val_gen, 1), np.expand_dims(X_val_spec, 1)], y_val), callbacks = [checkpoint, es], batch_size=32)

# evaluating the model
score = classifier.evaluate([np.expand_dims(X_test_gen, 1), np.expand_dims(X_test_spec, 1)], y_test, verbose=0) 
print('Final accuracy score: '+str(score[1]))


# The following cells need to be executed for applying active learning and then training the classifier. In order to execute the different experiments, the input parameters of the function "AL()" simply need to be adjusted  in the next cell according to the desired configuration. For this please take a look at how the parameters of the function "AL()" are defined.

# In[494]:


# sort general sentence embeddings so that general and specific sentence embeddings have the same labels 
# and number of instances
ind = sort_array(y_train_gen_al, y_train_spec_al)
X_train_gen_al, y_train = X_train_gen_al[ind], y_train_spec_al


# In[495]:


# training the model
classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath="weights/classifier/classifier_with_al/certainty_sampling/classifier_domain_3.h5")
history = classifier.fit([np.expand_dims(X_train_gen_al, 1), np.expand_dims(X_train_spec_al, 1)], y_train, epochs=30, validation_data = ([np.expand_dims(X_val_gen, 1), np.expand_dims(X_val_spec, 1)], y_val), callbacks = [checkpoint, es], batch_size=32)

# evaluating the model
score = classifier.evaluate([np.expand_dims(X_test_gen, 1), np.expand_dims(X_test_spec, 1)], y_test, verbose=0) 
print('Final accuracy score: '+str(score[1]))


# In[ ]:





# In[ ]:




