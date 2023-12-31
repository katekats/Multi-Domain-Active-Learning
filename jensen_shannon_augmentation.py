import os
import numpy as np
import random as rn
import tensorflow as tf
from hypermodel import MyHyperModel
import pickle as pkl
import pandas as pd
import collections
import re
from scipy.spatial import distance
import argparse


def load_data_from_path(filepath):
    try:
        with open(filepath, 'rb') as f:
            return pkl.load(f)
    except FileNotFoundError:
        print(f"File {filepath} not found!")
        return None


def load_general_embeddings():
    # Load general embeddings
    X_train_gen_ = load_from_file('data/sentence_embeddings/general/sorted/train/train_data_'+str(spec_index)+'.pkl')
    y_train = load_from_file('data/sentence_embeddings/general/sorted/train/train_labels_'+str(spec_index)+'.pkl')
    X_val_test_spec = load_from_file('data/sentence_embeddings/general/sorted/val_test/vt_data_'+str(spec_index)+'.pkl')
    y_val_test = load_from_file('data/sentence_embeddings/general/sorted/val_test/vt_labels_'+str(spec_index)+'.pkl')  
    labels_total = np.hstack((y_train[:,:1400], y_val_test))
    # Split and return data
    return X_train_gen[:4200], X_val_test_spec[:600], X_val_test_spec[600:], y_train[0,:4200], y_val_test[0,:600], y_val_test[0,600:], labels_total

def load_specific_embeddings():
    X_spec = load_data_from_file('data/sentence_embeddings/specific/sentemb/sentemb_unlabeled_'+str(spec_index)+'.pkl')
    X_spec = np.repeat(X_spec, repeats=3, axis=1)
    # Split and return data
    return X_spec.transpose()[:4200], X_spec.transpose()[4200:4800], X_spec.transpose()[4800:]

def load_unsorted_general_data():
    data_general = load_data_from_file('data/sentence_embeddings/general/unsorted/sentemb/sentemb_unlabeled.pkl')
    labels_train = load_data_from_file('data/sentence_embeddings/general/unsorted/label_domain/label_domain_train_sentemb_unlabeled.pkl')
    labels_test = load_data_from_file('data/sentence_embeddings/general/unsorted/label_domain/label_domain_test_sentemb_unlabeled.pkl')
    labels_general = np.hstack((labels_train, labels_test))
    data_general = data_general.transpose()
    return data_general, labels_general

def load_cleaned_data():
    df_train = load_data_from_file('data/cleaned_data/merged_cleaned.p')
    df_test = load_data_from_file('data/cleaned_data/test_cleaned.p') 
    # Remove unlabeled data from train
    list_unlabel = df_train.index[df_train['label'] == 3].to_list()
    df_train = df_train[~df_train.index.isin(list_unlabel)].reset_index(drop=True)
    return df_train, df_test


def word_distribution(df_train, df_test):
    """
    Get the word distribution of each domain.
    The frequency of each existing word is computed in every domain.
    
    Args:
    - df_train (DataFrame): The training dataframe.
    - df_test (DataFrame): The test dataframe.

    Returns:
    - DataFrame: Word distribution where rows are the domains and columns are words, 
                 and cell values are the word frequency for the word in the domain.
    """
    # Create a list of data frames dfs, each data frame represents one domain
    df = pd.concat([df_train, df_test], ignore_index=True)
    dfs = [x for _, x in df.groupby('domain')]
    word_counter = []
    words = re.compile(r'\w+')
    for df in dfs:
        counts = collections.Counter()
        reviews = np.array([s for s in df['text']]) 
        for review in reviews:
            counts.update(words.findall(review.lower()))
        word_counter.append(counts)
        df_dist = pd.DataFrame(word_counter)
        df_dist = df_dist.fillna(0)
    return df_dist



def sort_array(array_to_sort, array_ref):
    """Sort `array_to_sort` based on `array_ref` and return the sorted indices."""
<<<<<<< HEAD
    
    # Get the indices for 0s and 1s from array_to_sort
    indeces_zeros = np.where(array_to_sort[0] == 0)[0]
    indeces_ones = np.where(array_to_sort[0] == 1)[0]
    
    # Create an empty array to hold the sorted indices
    indeces_sorted = np.empty_like(array_ref[0], dtype=int)
    
    # Replace values in indeces_sorted with corresponding indices from indeces_zeros and indeces_ones
    indeces_sorted[array_ref[0] == 0] = indeces_zeros
    indeces_sorted[array_ref[0] == 1] = indeces_ones
    
    return indeces_sorted

=======
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
>>>>>>> de764fb3fa93558734efc7909d89b34387bebcfe

def filter_and_sort_data(df_dist, labels_general, data_general, labels_total, spec_index):
    # Compute Jensen-Shannon distances between the spec_index distribution and all others  
    js_d = [distance.jensenshannon(np.array(df_dist.iloc[spec_index]), row) for _, row in df_dist.iterrows()]
    # Get indices of the 5 most similar distributions based on JS distances
    most_sim_dist = sorted(range(len(js_d)), key=lambda i: js_d[i], reverse=True)[-4:]
    # Remove the original spec_index as we are interested in the most similar ones excluding itself
    most_sim_dist.remove(spec_index)

    # Filter the data and labels to keep only those corresponding to the most similar distributions
    indices_to_keep = [i for i, value in enumerate(labels_general[1]) if int(value) in most_sim_dist]
    labels_general, data_general = labels_general[:, indices_to_keep], data_general[indices_to_keep]
    
    # Sort the general data based on labels_total for consistency
    ind = sort_array(labels_general, labels_total)

    # Rearrange `data_general` and `labels_general` based on the sorted indices
    data_general, labels_general = data_general[ind], labels_general[:, ind]
    
    # Partition the general data into training, validation, and test sets
    X_train_gen, X_val_gen, X_test_gen = data_general[:4200], data_general[4200:4800], data_general[4800:]
    return X_train_gen, X_val_gen, X_test_gen

def save_to_file(data, filename):
    with open(filename, "wb") as file:
        pkl.dump(data, file)


def jensen_shannon(spec_index):
    # Load general sentence embeddings
    X_train_gen, X_val_gen, X_test_gen, y_train, y_val, y_test, labels_total = load_general_embeddings()
    # Load specific embeddings
    X_train_spec, X_val_spec, X_test_spec = load_specific_embeddings()
    # Load unsorted general data
    data_general, labels_general = load_unsorted_general_data()
    # Load cleaned data
    df_train, df_test = load_cleaned_data()
    # Load unsorted general data
    data_general, labels_general = load_unsorted_general_data()
    # Load cleaned data
    df_test, df_train = load_cleaned_data()
    # Usage:
    df_word_dist = word_distribution(df_train, df_test)   
    X_train_gen, X_val_gen, X_test_gen = filter_and_sort_data(df_word_dist, labels_general, data_general, labels_total,  spec_index)
    save_to_file(X_train_gen, "X_train_gen_"+str(spec_index)+".pkl")
    save_to_file(X_val_gen, "X_val_gen_"+str(spec_index)+".pkl")
    save_to_file(X_test_gen, "X_rest_gen_"+str(spec_index)+".pkl")
    save_to_file(X_train_spec, "X_train_spec_"+str(spec_index)+".pkl")
    save_to_file(X_val_spec, "X_val_spec_"+str(spec_index)+".pkl")
    save_to_file(X_test_spec, "X_test_spec_"+str(spec_index)+".pkl")
    save_to_file(y_train, "y_train_"+str(spec_index)+".pkl")
    save_to_file(y_val, "y_val_"+str(spec_index)+".pkl")
    save_to_file(y_test, "y_test_"+str(spec_index)+".pkl")
    return X_train_gen, X_val_gen, X_test_gen, X_train_spec, X_val_spec, X_test_spec, y_train, y_val, y_test

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Calculate the Jensen-Shannon Distance.')
    parser.add_argument('spec_index', type=int, help='Index for the domain.')

    args = parser.parse_args()

    print(f"Running Jensen-Shannon distance for the domain with spec_index: {args.spec_index}")
    x = jensen_shannon(args.spec_index)
    print(x)
