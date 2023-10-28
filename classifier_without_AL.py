import os
import numpy as np
import random as rn
import tensorflow as tf
import pickle as pkl
import pandas as pd
import re
import argparse

# Constants
INPUT_SIZE = 300
TRAINING_SIZE = 4200
VALIDATION_SIZE = 600
TOTAL_SIZE = TRAINING_SIZE + VALIDATION_SIZE
MODEL_PATH = 'path_to_save_model/model.h5'

# ... [the rest of the helper functions remain unchanged]

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

def run_the_classifier(spec_index):
    set_seeds_and_configurations()

    # Load data and hyperparameters
    X_train_gen = load_from_file("X_train_gen.pkl")
    X_train_spec = load_from_file("X_train_spec.pkl")
    y_train = load_from_file("y_train.pkl")
    X_val_gen = load_from_file("X_val_gen.pkl")
    X_val_spec = load_from_file("X_val_spec.pkl")
    y_val = load_from_file("y_val.pkl")  # Fixed this line (was X_val)
    hyperparameters = load_hyperparameters_from_file()

    # Build and train model
    model = build_model(hyperparameters)
    history = train_model(model, X_train_gen, X_train_spec, y_train, X_val_gen, X_val_spec, y_val, hyperparameters)
    
    # Evaluation
    score = evaluate_model(model, X_test_gen, X_test_spec, y_test)

    # Summarize the results
    print(f"Original Model Accuracy: {score[1]*100:.2f}%")

    # Visualize training history for both models (if necessary)
    visualize_training_history(history, title="Training History")

    # Save model (if necessary)
    model.save(MODEL_PATH)

if __name__ == '__main__':
    # ... [argparse section remains unchanged]

    print(f"Running classifier_without_AL with spec_index: {args.spec_index}")
    run_the_classifier(args.spec_index)
