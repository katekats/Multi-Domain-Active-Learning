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
import argparse
from jensen_shannon_augmentation_AL import jensen_shannon_with_AL
# Defining constants
DEFAULT_INDEX_SPEC = 5

# Reading from environment variable or using default
index_spec = int(os.getenv('INDEX_SPEC', DEFAULT_INDEX_SPEC))

INPUT_SIZE = 300


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

def reverse_scale_data(data, scaler):
    return scaler.inverse_transform(data)

def preprocess_data(data_gen, data_spec, labels_gen, labels_spec, outlier_detection):
    # Remove outliers
    data_gen, labels_gen = outlier_removal(data_gen, labels_gen, outlier_detection)
    data_spec, labels_spec = outlier_removal(data_spec, labels_spec, outlier_detection)
    
    return data_gen, data_spec, labels_gen, labels_spec

def preprocess_data_classifier(data):
    return np.expand_dims(np.asarray(data), 1)

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
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath="weights/classifier/classifier_with_al/certainty_sampling/classifier_domain__"+str(index_spec)+".h5")
    history = classifier.fit(
        [preprocess_data_classifier(X_train_gen), 
         preprocess_data_classifier(X_train_spec)], 
        preprocess_data_classifier(y_train), epochs=20,
        validation_data=(preprocess_data_classifier(X_val_gen), 
                          preprocess_data_classifier(X_val_spec)], 
                         preprocess_data_classifier(y_val_spec),
        callbacks=[checkpoint, es], batch_size=64)
    return classifier

def load_hyperparameters_from_file(filename="best_hyperparameters.pkl"):
    with open(filename, "rb") as file:
        hyperparameters = pickle.load(file)
    return hyperparameters

def train_model_with_best_hyperparameters(X_train_gen, X_train_spec, y_train, X_val_gen, X_val_spec, y_val, hyperparameters):
    """
    Build and train a model using the best hyperparameters.

    Args:
    - X_train_gen, X_train_spec: Training data for general and specific inputs respectively.
    - y_train: Training labels.
    - X_val_gen, X_val_spec: Validation data for general and specific inputs respectively.
    - y_val: Validation labels.
    - hyperparameters: Best hyperparameters obtained from Bayesian optimization.

    Returns:
    - model: Trained model.
    - history: History object from training.
    """
    
    # Model building
    inp_gen = tf.keras.Input(shape=(1, INPUT_SIZE))
    inp_spec = tf.keras.Input(shape=(1, INPUT_SIZE))
    merged = tf.keras.layers.Concatenate()([inp_gen, inp_spec])
    merged = tf.keras.layers.Dense(hyperparameters['units1'], activation='sigmoid')(merged)
    merged = tf.keras.layers.Dropout(hyperparameters['dropout'])(merged)
    merged = tf.keras.layers.Dense(1, activation='sigmoid')(merged)

    model = tf.keras.Model([inp_gen, inp_spec], merged)
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(hyperparameters['learning_rate']), metrics=['accuracy'])

    # Model training
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath="weights/classifier/classifier_with_al/standard_model/classifier_domain_"+str(index_spec)+".h5")
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    rlr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
    
    history = model.fit([preprocess_data_classifier(X_train_gen), preprocess_data_classifier(X_train_spec), y_train, 
                        epochs=hyperparamepers['epochs'], 
                        validation_data=([preprocess_data_classifier(X_val_gen), preprocess_data_classifier(X_val_spec)], y_val), 
                        callbacks=[checkpoint, es, rlr], 
                        batch_size=hyperparameters['batch_size'])
    
    return model, history

def evaluate_model(model, X_test_gen, X_test_spec, y_test):
    """Evaluate the given model."""
    score = model.evaluate(
        [preprocess_data_classifier(X_test_gen), preprocess_data_classifier(X_test_spec)],
        preprocess_data_classifier(y_test),
        verbose=0
    )
    return score

def AL_algorithm(data_gen, data_spec, labels_gen, labels_spec, X_val_gen, X_test_gen, X_val_spec, X_test_spec, y_val_gen, y_test_gen, y_val_spec, y_test_spec, max_query1, uncertainty_sampling, outlier_detection):
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
        
          # inverse scaling
        X_train_gen = reverse_scale_data(X_train_gen) 
        X_valid_gen = reverse_scale_data(X_valid_gen)
        X_test_gen = reverse_scale_data(X_test_gen)
        X_val_gen =reverse_scale_data(X_val_gen)
          # inverse scaling
        X_train_spec = reverse_scale_datam(X_train_spec) 
        X_valid_spec = reverse_scale_data(X_valid_spec)
        X_test_spec = reverse_scale_data(X_test_spec)
        X_val_spec =reverse_scale_data(X_val_spec)
        # Check stopping criteria
        if X_train_spec.shape[0] > max_query1:
            break
    return X_train_gen, X_train_spec, y_train_spec, X_val_gen, X_val_spec, y_val_spec, X_test_gen, X_test_spec, y_test_spec


def classifier_with_AL(spec_index, par0): 
    # set the target domain
    set_seeds_and_configurations()
    hyperparameters = load_hyperparameters_from_file()
    X_train_gen, X_val_gen, X_test_gen, X_train_spec, X_val_spec, X_test_spec, y_train_gen, y_train, y_val_gen, y_test_gen, y_val, y_test = jensen_shannon_with_AL(spec_index)
    
    # data splitting
    X_train_gen_updated, X_train_spec_updated, y_train_gen_updated, y_train_spec_updated = AL_algorithm(
        X_train_gen, X_train_spec, y_train_gen, y_train, 
        np.vstack((X_val_gen, X_test_gen)), np.vstack((X_val_spec, X_test_spec)), 
        np.hstack((y_val_gen, y_test_gen)), np.hstack((y_val, y_test)), 
        par0, True, spec_index)   
    
    ind = sort_array(y_train_gen_updated, y_train_spec_updated)
    X_train_gen_updated, y_train = X_train_gen_updated[ind], y_train_spec_updated 
    ind2 = sort_array(y_val_gen, y_val)
    X_val_gen = X_val_gen[ind2]
    ind3 = sort_array(y_test_gen, y_test)
    X_test_gen = X_test_gen[ind3]
    model, history = train_model_with_best_hyperparameters(X_train_gen_updated, X_train_spec_updated, y_train, X_val_gen, X_val_spec, y_val, hyperparameters)
    score = evaluate_model(model, X_test_gen, X_test_spec, y_test)
    return (spec_index, par0, 'Final accuracy score: '+str(score[1]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run classifier with given parameters.')
    parser.add_argument('spec_index', type=int, help='Index for the classifier.')
    parser.add_argument('par0', type=int, help='First parameter.')

    args = parser.parse_args()

    print(f"Running classifier_with_AL with spec_index: {args.spec_index}, par0: {args.par0}")
    x = classifier_with_AL(args.spec_index, args.par0)
    print(x)





