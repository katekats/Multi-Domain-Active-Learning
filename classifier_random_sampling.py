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
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath="weights/classifier/classifier_without_al/standard_model/classifier_domain_3_15.h5")
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    rlr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
    
    history = model.fit([process_data(X_train_gen), process_data(X_train_spec), y_train, 
                        epochs=hyperparamepers['epochs'], 
                        validation_data=([process_data(X_val_gen), process_data(X_val_spec)], y_val), 
                        callbacks=[checkpoint, es, rlr], 
                        batch_size=hyperparameters['batch_size'])
    
    return model, history

    def evaluate_model(model, X_test_gen, X_test_spec, y_test):
    """Evaluate the given model."""
    score = model.evaluate(
        [preprocess_data(X_test_gen), preprocess_data(X_test_spec)],
        preprocess_data(y_test),
        verbose=0
    )
    return score

    def main():
        set_seeds_and_configurations()
        # Load general and specific sentence embeddings
        X_train_gen = load_from_file("X_train_gen.pkl")
        X_val_gen = load_from_file("X_val_gen.pkl")
        X_train_spec = load_from_file("X_train_spec.pkl")
        X_val_spec = load_from_file("X_val_spec.pkl")
        y_train = load_from_file("y_train.pkl")
        X_val = load_from_file("y_val.pkl")
        hyperparameters = load_hyperparameters_from_file()
        model, history = train_model_with_best_hyperparameters(X_train_gen, X_train_spec, y_train, X_val_gen, X_val_spec, y_val, hyperparameters)
        score = evaluate_model(model, X_test_gen, X_test_spec, y_test)