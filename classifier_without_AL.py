import os
import numpy as np
import random as rn
import tensorflow as tf
import pickle as pkl
import pandas as pd
import re
import argparse


TRAIN_GEN_EMBEDDINGS_PATH = 'data/sentence_embeddings/general/unsorted/sentemb/sentemb_unlabeled3.p'
TRAIN_LABELS_PATH = 'data/sentence_embeddings/general/unsorted/label_domain/label_domain_train_sentemb_unlabeled3.p'
TEST_LABELS_PATH = 'data/sentence_embeddings/general/unsorted/label_domain/label_domain_test_sentemb_unlabeled3.p'
TRAIN_CLEANED_DATA_PATH = 'data/cleaned_data/merged_cleaned.p'
TEST_CLEANED_DATA_PATH = 'data/cleaned_data/test_cleaned.p'

# Constants
INPUT_SIZE = 300
TRAINING_SIZE = 4200
VALIDATION_SIZE = 600
TOTAL_SIZE = TRAINING_SIZE + VALIDATION_SIZE  # assuming 4800 total for training and validation

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

def load_from_file(filename):
    with open(filename, "rb") as file:
        data = pkl.load(file)
    return data 

def load_hyperparameters_from_file(filename="best_hyperparameters.pkl"):
    with open(filename, "rb") as file:
        hyperparameters = pickle.load(file)
    return hyperparameters

def preprocess_data(data):
    return np.expand_dims(np.asarray(data), 1) 

def train_model_with_best_hyperparameters(X_train_gen, X_train_spec, y_train, X_val_gen, X_val_spec, y_val, hyperparameters):
    """
    Build and train a model using the best hyperparameters.

    Args:
    - X_train_gen, X_train_spec: Training data for general and specific inputs respectively.
    - y_train: Training labels.
    - X_val_gen, X_val_spec: Validation data for general and specific inputs respectively.
    - y_val: Validation labels.
    - hyperparrameters: Best hyperparameters obtained from Bayesian optimization.

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
    parser = argparse.ArgumentParser(description='Run the classifier with given parameters.')
    parser.add_argument('--spec-index', type=int, default=DEFAULT_INDEX_SPEC, help='Index for the classifier.')    
    args = parser.parse_args()
    spec_index = args.spec_index
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

    # Summarize the results
    print(f"Original Model Accuracy: {score[1]*100:.2f}%")
    print(f"Model Accuracy after filtering and sorting: {score_new[1]*100:.2f}%")

    # Visualize training history for both models (if necessary)
    visualize_training_history(history, title="Training History")

    # Save model (if necessary)
    model.save('path_to_save_model/model.h5')

    
if __name__ == '__main__':
    main()
