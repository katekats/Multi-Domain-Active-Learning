import os
import numpy as np
import random as rn
import tensorflow as tf
import pickle
import pandas as pd
import re
import argparse
from jensen_shannon_augmentation import jensen_shannon

# Constants
INPUT_SIZE = 300
TRAINING_SIZE = 4200
VALIDATION_SIZE = 600
TOTAL_SIZE = TRAINING_SIZE + VALIDATION_SIZE

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

def load_hyperparameters_from_file(filename="best_hyperparameters.pkl"):
    with open(filename, "rb") as file:
        hyperparameters = pickle.load(file)
    return hyperparameters

def preprocess_data(data):
    return np.expand_dims(np.asarray(data), 1)

def build_model(hyperparameters):
    inp_gen = tf.keras.Input(shape=(1, INPUT_SIZE))
    inp_spec = tf.keras.Input(shape=(1, INPUT_SIZE))
    merged = tf.keras.layers.Concatenate()([inp_gen, inp_spec])
    merged = tf.keras.layers.Dense(hyperparameters['units1'], activation='sigmoid')(merged)
    merged = tf.keras.layers.Dropout(hyperparameters['dropout'])(merged)
    merged = tf.keras.layers.Dense(1, activation='sigmoid')(merged)
    model = tf.keras.Model([inp_gen, inp_spec], merged)
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(hyperparameters['learning_rate']), metrics=['accuracy'])
    return model

def train_model(model, X_train_gen, X_train_spec, y_train, X_val_gen, X_val_spec, y_val, hyperparameters):
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath="weights/classifier/classifier_without_al/standard_model/classifier_domain_3_15.h5")
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    rlr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

    history = model.fit(
        [preprocess_data(X_train_gen), preprocess_data(X_train_spec)], y_train, 
        epochs=hyperparameters['epochs'], 
        validation_data=([preprocess_data(X_val_gen), preprocess_data(X_val_spec)], y_val), 
        callbacks=[checkpoint, es, rlr], 
        batch_size=hyperparameters['batch_size']
    )
    return history

def evaluate_model(model, X_test_gen, X_test_spec, y_test):
    """Evaluate the given model."""
    score = model.evaluate(
        [preprocess_data(X_test_gen), preprocess_data(X_test_spec)],
        preprocess_data(y_test),
        verbose=0
    )
    return score

def run_the_classifier(spec_index):
    # Initialize random seeds and configurations for reproducibility
    set_seeds_and_configurations()
    ## Load hyperparameters for training the model
    hyperparameters = load_hyperparameters_from_file()
    # Get the  domain with the 4 most similar distributions using the Jensen-Shannon method with the specified index
    X_train_gen, X_val_gen, X_test_gen, X_train_spec, X_val_spec, X_test_spec, y_train, y_val, y_test = jensen_shannon(spec_index)
    # Build and train model
    model = build_model(hyperparameters)
    history = train_model(model, X_train_gen, X_train_spec, y_train, X_val_gen, X_val_spec, y_val, hyperparameters)
    
    # Evaluation
    score = evaluate_model(model, X_test_gen, X_test_spec, y_test)

    # Summarize the results
    print(f"Original Model Accuracy: {score[1]*100:.2f}%")
    # return the results
    return (spec_index, f"Final accuracy score: {score[1]*100:.2f}%")

if __name__ == "__main__":
    # Argument parser setup to enable command line inputs
    parser = argparse.ArgumentParser(description='Run classifier with given parameters.')
    parser.add_argument('spec_index', type=int, help='Index for the classifier.')

    args = parser.parse_args()

    print(f"Running classifier with spec_index: {args.spec_index}")
    x = classifier_with_AL(args.spec_index)
    print(x)





if __name__ == '__main__':

     # Argument parser setup to enable command line inputs
    parser = argparse.ArgumentParser(description='Run classifier with given parameters.')
    parser.add_argument('spec_index', type=int, help='Index for the classifier.')

    args = parser.parse_args()

    print(f"Running classifier_without_AL with spec_index: {args.spec_index}")
    x = run_the_classifier(args.spec_index)
    print(x)

