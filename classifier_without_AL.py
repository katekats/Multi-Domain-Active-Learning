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
    
def load_from_file(filepath):
    try:
        with open(filepath, 'rb') as f:
            return pkl.load(f)
    except FileNotFoundError:
        print(f"File {filepath} not found!")
        return None



def create_model():
    # domain-general and domain-specific model parts
    inp_gen = tf.keras.Input(shape=(1, INPUT_SIZE))
    inp_spec = tf.keras.Input(shape=(1, INPUT_SIZE))

    # concatenate domain-general and domain-specific results
    merged = tf.keras.layers.Concatenate()([inp_gen, inp_spec])
    merged = tf.keras.layers.Dense(INPUT_SIZE, activation='sigmoid')(merged)
    merged = tf.keras.layers.Dropout(0.5)(merged)
    merged = tf.keras.layers.Dense(1, activation='sigmoid')(merged)
    
    return tf.keras.Model([inp_gen, inp_spec], merged)

def evaluate_model(model, X_test_gen, X_test_spec, y_test):
    """Evaluate the given model."""
    score = model.evaluate(
        [preprocess_data(X_test_gen), preprocess_data(X_test_spec)],
        preprocess_data(y_test),
        verbose=0
    )
    return score


def preprocess_data(data):
    return np.expand_dims(np.asarray(data).astype(np.float32), 1)    

def load_hyperparameters_from_file(filename="best_hyperparameters.pkl"):
    with open(filename, "rb") as file:
        hyperparameters = pickle.load(file)
    return hyperparameters

def main():
    set_seeds_and_configurations()
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
    
    X_train_gen, X_val_gen, X_test_gen = filter_and_sort_data(df_word_dist, labels_general, data_general, labels_total,  index_spec)
    save_to_file(X_train_gen, "X_train_gen.pkl")
    save_to_file(X_val_gen, "X_val_gen.pkl")
    save_to_file(X_train_spec, "X_train_spec.pkl")
    save_to_file(X_val_spec, "X_val_spec.pkl")
    save_to_file(y_train, "y_train_.pkl")
    save_to_file(y_val, "y_val.pkl")

     hyperparameters = load_hyperparameters_from_file()
     # Create and compile the model
    classifier4 = create_model()
    compile_model(classifier4)

    # Train the model
    history = train_model(classifier4, X_train_gen, X_train_spec, y_train, X_val_gen, X_val_spec)

    # Evaluate the model
    score = evaluate_model(classifier4, X_test_gen, X_test_spec, y_test)
    history_new = train_model(classifier4, X_train_gen, X_train_spec, y_train, X_val_gen, X_val_spec)
    score_new = evaluate_model(classifier4, X_test_gen, X_test_spec, y_test)

    # Summarize the results
    print(f"Original Model Accuracy: {score[1]*100:.2f}%")
    print(f"Model Accuracy after filtering and sorting: {score_new[1]*100:.2f}%")

    # Visualize training history for both models (if necessary)
    visualize_training_history(history, title="Original Training History")
    visualize_training_history(history_new, title="Training History after Filtering and Sorting")

    # Save model (if necessary)
    classifier4.save('path_to_save_model/model.h5')

    tuner2 = kt.BayesianOptimization(
    MyHyperModel(),
    objective="val_accuracy",
    max_trials=50,
    overwrite=True,
    num_initial_points=25,
    alpha=0.001,
    beta=2.6
)

    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    tuner2.search([np.expand_dims(X_train_gen, 1), np.expand_dims(X_train_spec, 1)], y_train, validation_data=([np.expand_dims(X_val_gen, 1), np.expand_dims(X_val_spec, 1)], y_val), callbacks=[es])

    best_hps = tuner2.get_best_hyperparameters(num_trials=1)[0]

    # Building the model using best hyperparameters
    inp_gen = tf.keras.Input(shape=(1, INPUT_SIZE))
    inp_spec = tf.keras.Input(shape=(1, INPUT_SIZE))
    merged = tf.keras.layers.Concatenate()([inp_gen, inp_spec])
    merged = tf.keras.layers.Dense(best_hps.get('units1'), activation='sigmoid')(merged)
    merged = tf.keras.layers.Dropout(best_hps.get('dropout'))(merged)
    merged = tf.keras.layers.Dense(1, activation='sigmoid')(merged)

    classifier = tf.keras.Model([inp_gen, inp_spec], merged)
    classifier.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(best_hps.get('learning_rate')), metrics=['accuracy'])

    # Training the model
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath="weights/classifier/classifier_without_al/standard_model/classifier_domain_3_15.h5")
    history = classifier.fit([np.expand_dims(X_train_gen, 1), np.expand_dims(X_train_spec, 1)], y_train, epochs=best_hps.get('epochs'), 
                         validation_data=([np.expand_dims(X_val_gen, 1), np.expand_dims(X_val_spec, 1)], y_val), 
                         callbacks=[checkpoint, es, rlr], batch_size=best_hps.get('batch_size'))

# Evaluating the model
    score = classifier.evaluate([np.expand_dims(X_test_gen, 1), np.expand_dims(X_test_spec, 1)], y_test, verbose=1)
    print(f'Final accuracy score: {score[1]}')  
if __name__ == '__main__':
    main()
