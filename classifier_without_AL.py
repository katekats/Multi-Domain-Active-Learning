import os
import numpy as np
import random as rn
import tensorflow as tf
import pickle as pkl
import pandas as pd
import re


# Defining constants
DEFAULT_INDEX_SPEC = 5

# Reading from environment variable or using default
INDEX_SPEC = int(os.getenv('INDEX_SPEC', DEFAULT_INDEX_SPEC))

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

import numpy as np
import tensorflow as tf

def train_model_with_best_hyperparameters(X_train_gen, X_train_spec, y_train, X_val_gen, X_val_spec, y_val, hyperparameters):
    """
    Build and train a model using the best hyperparameters.

    Args:
    - X_train_gen, X_train_spec: Training data for general and specific inputs respectively.
    - y_train: Training labels.
    - X_val_gen, X_val_spec: Validation data for general and specific inputs respectively.
    - y_val: Validation labels.
    - best_hps: Best hyperparameters obtained from Bayesian optimization.

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
    
    history = model.fit([np.expand_dims(X_train_gen, 1), np.expand_dims(X_train_spec, 1)], y_train, 
                        epochs=hyperparamepers['epochs'], 
                        validation_data=([np.expand_dims(X_val_gen, 1), np.expand_dims(X_val_spec, 1)], y_val), 
                        callbacks=[checkpoint, es, rlr], 
                        batch_size=hyperparameters['batch_size'])
    
    return model, history



def preprocess_data(data):
    return np.expand_dims(np.asarray(data).astype(np.float32), 1)    

def load_hyperparameters_from_file(filename="best_hyperparameters.pkl"):
    with open(filename, "rb") as file:
        hyperparameters = pickle.load(file)
    return hyperparameters

def main():
    set_seeds_and_configurations()
    

    hyperparameters = load_hyperparameters_from_file()
    # Extract individual hyperparameters
    dropout = hyperparameters['dropout']
    units1 = hyperparameters['units1']
    learning_rate = hyperparameters['learning_rate']
    epochs = hyperparameters['epochs']
    batch_size = hyperparameters['batch_size']
    # Create and compile the model
    # Assuming best_hps is already loaded from the pickle file
    model, history = train_model_with_best_hyperparameters(X_train_gen, X_train_spec, y_train, X_val_gen, X_val_spec, y_val, hyperparameters)

    score_new = evaluate_model(classifier4, X_test_gen, X_test_spec, y_test)

    # Summarize the results
    print(f"Original Model Accuracy: {score[1]*100:.2f}%")
    print(f"Model Accuracy after filtering and sorting: {score_new[1]*100:.2f}%")

    # Visualize training history for both models (if necessary)
    visualize_training_history(history, title="Original Training History")
    visualize_training_history(history_new, title="Training History after Filtering and Sorting")

    # Save model (if necessary)
    classifier4.save('path_to_save_model/model.h5')

   

# Evaluating the model
    score = classifier.evaluate([np.expand_dims(X_test_gen, 1), np.expand_dims(X_test_spec, 1)], y_test, verbose=1)
    print(f'Final accuracy score: {score[1]}')  
if __name__ == '__main__':
    main()
