# # Active Learning and Classification

# This file produces all the active learning results of the evaluation chapter. All cells needs to be executed until the headline "Executing Active Learning and training the model" is reached. Then, a specific model is chosen and executed. Note: Make sure to select the desired target domain when loading the data below.

# Make sure to adjust the checkpoint paths when training the models such that the weights are saved in the desired paths.

# ## Importing libraries and setting configurations

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

# Defining constants
DEFAULT_INDEX_SPEC = 5

# Reading from environment variable or using default
index_spec = int(os.getenv('INDEX_SPEC', DEFAULT_INDEX_SPEC))

INPUT_SIZE = 300

TRAIN_GEN_EMBEDDINGS_PATH = 'data/sentence_embeddings/general/unsorted/sentemb/sentemb_unlabeled3.p'
TRAIN_LABELS_PATH = 'data/sentence_embeddings/general/unsorted/label_domain/label_domain_train_sentemb_unlabeled3.p'
TEST_LABELS_PATH = 'data/sentence_embeddings/general/unsorted/label_domain/label_domain_test_sentemb_unlabeled3.p'
TRAIN_CLEANED_DATA_PATH = 'data/cleaned_data/merged_cleaned.p'
TEST_CLEANED_DATA_PATH = 'data/cleaned_data/test_cleaned.p'



def set_seeds_and_configurations():
    # Setting seeds to reproduce results
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(1)
    rn.seed(2)
    tf.random.set_seed(3)
    # Configurations to use a single thread
    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    tf.compat.v1.keras.backend.set_session(sess)


# function for sorting two arrays such that both arrays have the same labels
def sort_array(labels, labels_ref):
    index_gen_zeros = np.where(labels == 0)[0]
    index_gen_ones = np.where(labels == 1)[0]
    
    result_ind = np.zeros_like(labels_ref, dtype=int)
    result_ind[labels_ref == 0] = index_gen_zeros
    result_ind[labels_ref == 1] = index_gen_ones

    return result_ind

def sort_for_jensen(array_to_sort, array_ref):
    y, y_ref = array_to_sort[0].astype(int), array_ref[0].astype(int)

    # Get indices where y is 0 and 1
    indeces_zeros = np.where(y == 0)[0]
    indeces_ones = np.where(y == 1)[0]

    # Ensure that the number of zeros and ones in y matches those in y_ref
    if not (len(indeces_zeros) == np.sum(y_ref == 0) and len(indeces_ones) == np.sum(y_ref == 1)):
        raise ValueError("The distribution of 0s and 1s in y doesn't match with y_ref.")

    # Create an array to store sorted indices
    indeces_sorted = np.zeros_like(y_ref, dtype=int)

    # Assign indices from indeces_zeros and indeces_ones based on the values in y_ref
    indeces_sorted[y_ref == 0] = indeces_zeros
    indeces_sorted[y_ref == 1] = indeces_ones

    return indeces_sorted

# function for removing outliers
# choose parameter "algorithm" as 0 for no outlier detection, 1 for elliptic envelope
# and choose 2 for isolation forest
def outlier_removal(X, y, algorithm):
    outlier_detector = None
    
    if algorithm == 1:
        outlier_detector = EllipticEnvelope(support_fraction=0.9, random_state=2)
    elif algorithm == 2:
        outlier_detector = IsolationForest(random_state=3)
    
    if outlier_detector:
        outlier_detector.fit(X)
        mask = outlier_detector.predict(X) != -1
        X, y = X[mask], y[mask]
    
    return X, y

def scale_data(data, scaler, fit=True):
    return scaler.fit_transform(data) if fit else scaler.transform(data)

def preprocess_data(data_gen, data_spec, labels_gen, labels_spec, outlier_detection):
    # Remove outliers
    data_gen, labels_gen = outlier_removal(data_gen, labels_gen, outlier_detection)
    data_spec, labels_spec = outlier_removal(data_spec, labels_spec, outlier_detection)
    
    return data_gen, data_spec, labels_gen, labels_spec

def create_classifier(input_size):
    inp_gen = tf.keras.Input(shape=(1, input_size))
    inp_spec = tf.keras.Input(shape=(1, input_size))
    merged = tf.keras.layers.Concatenate()([inp_gen, inp_spec])
    merged = tf.keras.layers.Dense(100, activation='sigmoid')(merged)
    merged = tf.keras.layers.Dropout(.5)(merged)
    merged = tf.keras.layers.Dense(1, activation='sigmoid')(merged)
    
    return tf.keras.Model([inp_gen, inp_spec], merged)

def train_classifier(classifier, X_train_gen, X_train_spec, y_train, X_val_gen, X_val_spec, y_val_spec):
    classifier.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.01), metrics=['accuracy'])
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath="weights/classifier/classifier_with_al/certainty_sampling/classifier_domain_6.h5")
    history = classifier.fit(
        [np.expand_dims(np.asarray(X_train_gen).astype(np.float32), 1), 
         np.expand_dims(np.asarray(X_train_spec).astype(np.float32), 1)], 
        np.asarray(y_train).astype(np.float32), epochs=20,
        validation_data=([np.expand_dims(np.asarray(X_val_gen).astype(np.float32), 1), 
                          np.expand_dims(np.asarray(X_val_spec).astype(np.float32), 1)], 
                         np.asarray(y_val_spec).astype(np.float32)),
        callbacks=[checkpoint, es], batch_size=64)
    return classifier


def AL(data_gen, data_spec, labels_gen, labels_spec, X_val_gen, X_test_gen, X_val_spec, X_test_spec, y_val_gen, y_test_gen, y_val_spec, y_test_spec, max_query1, max_query2, uncertainty_sampling, outlier_detection):
    # Preprocess and initializations
    data_gen, data_spec, labels_gen, labels_spec = preprocess_data(data_gen, data_spec, labels_gen, labels_spec, outlier_detection)
    X_train_gen, X_valid_gen, y_train_gen, y_valid_gen = data_gen[:150], data_gen[150:], labels_gen[:150], labels_gen[150:]
    X_train_spec, X_valid_spec, y_train_spec, y_valid_spec = data_spec[:100], data_spec[100:], labels_spec[:100], labels_spec[100:]
    classifier = create_classifier(INPUT_SIZE)
    ind1 = sort_array(y_test_gen, y_test_spec)
    X_test_gen, y_test_gen = X_test_gen[ind1], y_test_spec
    ind2 = sort_array(y_val_gen, y_val_spec)
    X_val_gen, y_val_gen = X_val_gen[ind2], y_val_spec

    scaler = MinMaxScaler(feature_range=(0, 1))

    # Active learning loop
    while True:           
        ind3 = sort_array(y_train_gen, y_train_spec)
        X_train_gen1, y_train1 = X_train_gen[ind3], y_train_spec
        ind4 = sort_array(y_valid_gen, y_valid_spec)
        X_valid_gen1, y_valid_gen1 = X_valid_gen[ind4], y_valid_spec
        
        # Scaling
        X_train_gen1 = scale_data(X_train_gen1, scaler)
        X_valid_gen1 = scale_data(X_valid_gen1, scaler)
        X_test_gen = scale_data(X_test_gen, scaler)
        X_val_gen = scale_data(X_val_gen, scaler)
        X_train_spec = scale_data(X_train_spec, scaler)
        X_valid_spec = scale_data(X_valid_spec, scaler)
        X_test_spec = scale_data(X_test_spec, scaler)
        X_val_spec = scale_data(X_val_spec, scaler)

        # Training
        train_classifier(classifier, X_train_gen1, X_train_spec, y_train1, X_valid_gen, X_valid_spec, y_val_spec)
        
        # Get predictions for validation set
        get_prob = np.squeeze(classifier.predict([np.expand_dims(X_valid_gen, 1), np.expand_dims(X_valid_spec, 1)]))
        prob = [1-get_prob[i] if get_prob[i]<0.5 else get_prob[i] for i in range(get_prob.shape[0])]
        
        # Sample based on uncertainty
        ind = np.argsort(prob)[:100] if uncertainty_sampling else np.argsort(prob)[-100:]
        
        # Update training and validation sets
        X_train_gen = np.vstack((X_train_gen1, X_valid_gen1[ind]))
        y_train_gen = np.hstack((y_train1, y_valid_gen1[ind]))
        X_valid_gen = np.delete(X_valid_gen1, ind, axis=0)
        y_valid_gen = np.delete(y_valid_gen1, ind, axis=0) 
        X_train_spec = np.vstack((X_train_spec, X_valid_spec[ind]))
        y_train_spec = np.hstack((y_train_spec, y_valid_spec[ind]))
        X_valid_spec = np.delete(X_valid_spec, ind, axis=0)
        y_valid_spec = np.delete(y_valid_spec, ind, axis=0)
        
        # Check stopping criteria
        if X_train_spec.shape[0] > max_query1:
            break
    return X_train_gen, X_train_spec, y_train_spec, X_val_gen, X_val_spec, y_val_spec, X_test_gen, X_test_spec, y_test_spec

# importing the data for the general sentence embeddings, here corresponding data from domain 0 was chosen
def return_results_AL(index_spec, par0, par1): 
    print(index_spec)
    X_train_gen_all = load_from_file('data/sentence_embeddings/general/sorted/train/train_data6_'+str(index_spec)+'.p')
    y_train_gen_all = load_from_file('data/sentence_embeddings/general/sorted/train/train_labels6_'+str(index_spec)+'.p')
    X_val_test_spec = load_from_file('data/sentence_embeddings/general/sorted/val_test/vt_data6_'+str(index_spec)+'.p')
    y_val_test = load_from_file('data/sentence_embeddings/general/sorted/val_test/vt_labels6_'+str(index_spec)+'.p')   
    labels_total = np1.hstack((y_train_gen_all[:,:4200].astype(int), y_val_test))
    X_val_gen, X_test_gen = X_val_test_spec[:600], X_val_test_spec[600:]
    y_train_gen_all = y_train_gen_all[0,:]
    y_train, y_val, y_test = y_train_gen_all[:4200], y_val_test[0,:600], y_val_test[0,600:]

    # import the data from the specific sentence embeddings, here corresponding data from domain 0 was chosen

    X_spec = load_data_from_file('data/sentence_embeddings/specific/sentemb/sentemb_unlabeled_'+str(k)+'.p'')
    X_spec = np.repeat(X_spec, repeats=3, axis=1)    
    X_train_spec, X_val_spec, X_test_spec = X_spec.transpose()[:4200], X_spec.transpose()[4200:4800], X_spec.transpose()[4800:
    # load the original, unsorted data
    data_general = data_general.transpose()
    data_general = load_data_from_file('data/sentence_embeddings/general/unsorted/sentemb/sentemb_unlabeled3.p')
    labels_train = load_data_from_file('data/sentence_embeddings/general/unsorted/label_domain/label_domain_train_sentemb_unlabeled3.p')
    labels_test = load_data_from_file('data/sentence_embeddings/general/unsorted/label_domain/label_domain_test_sentemb_unlabeled3.p')
    labels_general = np.hstack((labels_train, labels_test))
    # load the cleaned data
    df_train = load_data_from_file('data/cleaned_data/merged_cleaned.p')
    df_test = load_data_from_file('data/cleaned_data/test_cleaned.p')

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


class TestCase:
    def __init__(self, name, i_range, pars):
        self.name = name
        self.i_range = i_range
        self.pars = pars
        
test_cases = [
    TestCase("peirama 1", range(0,16), [(1400, 2000)]),
]

for test_case in test_cases:
    for i in test_case.i_range:
        for pars in test_case.pars:
            print(test_case.name, i, pars)
            x = return_results_AL(i, pars[0], pars[1])
            print(x)



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


# importing the data for the general sentence embeddings, here corresponding data from domain 0 was chosen
def return_results_AL(k, par0, par1): 
    # set the target domain
    
# data splitting
    X_train_gen_al,X_train_spec_al, y_train_gen_al, y_train_spec_al  = AL( X_train_gen, X_train_spec, y_train_gen, y_train, np.vstack((X_val_gen,X_test_gen)), np.vstack((X_val_spec,X_test_spec)), np.hstack((y_val_gen,y_test_gen)),np.hstack((y_val,y_test)),pars[0], pars[1], True, index_spec)
    ind = sort_array(y_train_gen_al, y_train_spec_al)
    X_train_gen_al, y_train = X_train_gen_al[ind], y_train_spec_al 
    ind2 = sort_array(y_val_gen, y_val)
    X_val_gen = X_val_gen[ind2]
    ind3 = sort_array(y_test_gen, y_test)
    X_test_gen = X_test_gen[ind3]

    #THE LSTM Classifier

    INPUT_SIZE = 300
    LATENT_SIZE = 300

# domain-general model parts
    inp_gen = tf.keras.Input(shape=(1,INPUT_SIZE))
    inp_spec = tf.keras.Input(shape=(1,INPUT_SIZE))
    merged = tf.keras.layers.Concatenate()([inp_gen, inp_spec])
    merged = tf.keras.layers.Dense(300, activation='sigmoid')(merged)
    merged = tf.keras.layers.Dropout(.2)(merged)
    merged = tf.keras.layers.Dense(1, activation='sigmoid')(merged)
    classifier = tf.keras.Model([inp_gen,inp_spec], merged)

    
    
    # apply AL on specific and general sentence embeddings
# make sure to pick the right parameters for uncertainty_sampling and outlier_detection 
# SIDENOTE: for certainty sampling experiment change max_query to 2200 (for domain 4 even to 3000) for general embeddings - due to different number of positive/negative samples
  

    classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath="weights/classifier/classifier_with_al/certainty_sampling/classifier_domain_3.h5")
    history = classifier.fit([np.expand_dims(X_train_gen_al, 1), np.expand_dims(X_train_spec_al, 1)], y_train, epochs=50, validation_data = ([np.expand_dims(X_val_gen, 1), np.expand_dims(X_val_spec, 1)], y_val), callbacks = [checkpoint, es], batch_size=32)

# evaluating the model
    print(X_test_gen.shape, X_test_spec.shape, y_test.shape)
    score = classifier.evaluate([np.expand_dims(X_test_gen, 1), np.expand_dims(X_test_spec, 1)], y_test, verbose=0) 
    print(k)
    return(k, pars[0], pars[1], 'Final accuracy score: '+str(score[1]))

import numpy as np1
# importing the data for the general sentence embeddings, here corresponding data from domain 0 was chosen
def return_results_AL(k): 
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




