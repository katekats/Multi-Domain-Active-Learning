from keras_self_attention import SeqSelfAttention
import os
import h5py
import numpy as np
import pandas as pd
import random as rn
import pickle as pkl
import tensorflow as tf
import sys
import argparse

# Set up argparse
parser = argparse.ArgumentParser(description="Your Script Description")
parser.add_argument('--embedding_type', default='GENERAL', help="Type of embedding: GENERAL or SPECIFIC")
args = parser.parse_args()

# Define embedding_type as a global variable
embedding_type = args.embedding_type

SEQUENCE_LEN = 50
EMBED_SIZE = 300
LATENT_SIZE = 300
encoding_dim = 100



def load_data_from_file(filename):
    """Load data from a file."""
    with open(filename, 'rb') as f:
        return pkl.load(f)

def shuffle_data(data, labels, seed):
    """Shuffle data and labels."""
    idx = np.random.RandomState(seed=seed).permutation(data.shape[0])
    return data[idx], labels[:, idx]

def filter_data_by_domain(X, label_domain, domain):
    """Filter data by domain."""
    index_domain = [i for i, e in enumerate(label_domain[1]) if e == domain]
    return X[index_domain], label_domain[:, index_domain]

# Build autoencoder
def build_autoencoder():
    # encoder
    inp = tf.keras.Input(shape=(SEQUENCE_LEN, EMBED_SIZE))
    enc_out1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=LATENT_SIZE, input_shape=(SEQUENCE_LEN, EMBED_SIZE), return_sequences=True))(inp)
    inp_att, attn_weights = SeqSelfAttention(return_attention=True)(enc_out1)
    enc_out = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=300, input_shape=(SEQUENCE_LEN, EMBED_SIZE)), merge_mode='sum')(inp_att)

    # encoder model (to extract sentence embeddings later)
    encoder_model = tf.keras.Model(inputs=inp, outputs=enc_out)

    rep_vec = tf.keras.layers.RepeatVector(SEQUENCE_LEN)(enc_out)

    # decoder
    dec_lstm_out = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=300, return_sequences=True), merge_mode='sum')(rep_vec)
    dec_dense_out = tf.keras.layers.Dense(EMBED_SIZE)(dec_lstm_out)

    # entire autoencoder model
    autoencoder = tf.keras.Model(inp, dec_dense_out)
    return autoencoder

def remove_unlabeled_entries(embeddings, label_domain_data):
    """
    Remove unlabeled entries from the embeddings and label_domain_data.
    """
    df = pd.DataFrame(label_domain_data.transpose(), columns=['label', 'domain', 'idx_domain'])
    list_unlabel = df.index[df['label'] == 3].tolist()

    # Delete the rows with label=3 (unlabeled)
    df_filtered = df[~df.index.isin(list_unlabel)].reset_index(drop=True)
    label_domain_filtered = df_filtered.to_numpy().transpose()

    df_embeddings = pd.DataFrame(embeddings)
    df_embeddings_filtered = df_embeddings[~df_embeddings.index.isin(list_unlabel)].reset_index(drop=True)
    embeddings_filtered = df_embeddings_filtered.to_numpy().transpose()

    return embeddings_filtered, label_domain_filtered

def save_embeddings(embedding_type, embeddings, label_domain_merged, label_domain_test):
    """
    Save the embeddings and label domain data based on the embedding type.
    """
    global embedding_type
    if embedding_type == 'GENERAL':
        path_prefix = "data/sentence_embeddings/general/unsorted"
    else:  # SPECIFIC
        path_prefix = "data/sentence_embeddings/specific"

    pkl.dump(embeddings, open(f"{path_prefix}/sentemb/sentemb_unlabeled14.pkl", "wb"))
    pkl.dump(label_domain_merged, open(f"{path_prefix}/label_domain/label_domain_train_sentemb_unlabeled.pkl", "wb"))
    pkl.dump(label_domain_test, open(f"{path_prefix}/label_domain/label_domain_test_sentemb_unlabeled14.pkl", "wb"))

def process_and_save_embeddings(encoder_model, data, label_domain_merged, label_domain_test, embedding_type, domain=None):
    
    # Use encoder to generate sentence embeddings
    sentence_embeddings = encoder_model.predict(data)

    # Create DataFrame from label_domain_merged
    df_labels = pd.DataFrame(label_domain_merged.transpose(), columns=['label', 'domain', 'idx_domain'])

    # Identify and filter out unlabeled rows
    list_unlabel = df_labels.index[df_labels['label'] == 3].to_list()
    df_labels_filtered = df_labels[~df_labels.index.isin(list_unlabel)].reset_index(drop=True)
    df_embeddings_filtered = pd.DataFrame(sentence_embeddings)[~pd.DataFrame(sentence_embeddings).index.isin(list_unlabel)].reset_index(drop=True)

    # Convert DataFrames back to numpy arrays
    filtered_labels = df_labels_filtered.to_numpy().transpose()
    filtered_embeddings = df_embeddings_filtered.to_numpy().transpose()

     # Determine save paths based on embedding type
    if embedding_type == 'GENERAL':
        base_path = "data/sentence_embeddings/general/unsorted/"
        filename_suffix = "unlabeled14"
    else:  # SPECIFIC
        if domain is None:
            raise ValueError("For specific embeddings, the domain number must be provided.")
        base_path = "data/sentence_embeddings/specific/unsorted/"
        filename_suffix = f"unlabeled{domain}_14"

    # Save the processed embeddings and labels
    pkl.dump(filtered_embeddings, open(os.path.join(base_path, f"sentemb/sentemb_{filename_suffix}.pkl"), "wb"))
    pkl.dump(filtered_labels, open(os.path.join(base_path, f"label_domain/label_domain_train_sentemb_{filename_suffix}.pkl"), "wb"))
    pkl.dump(label_domain_test, open(os.path.join(base_path, f"label_domain/label_domain_test_sentemb_{filename_suffix}.pkl"), "wb"))

    return filtered_embeddings, filtered_labels

def load_data_from_file(filename):
    with open(filename, 'rb') as f:
        return pkl.load(f)

def main():
    global embedding_type
    # Load the common data for all domains
    with h5py.File('data/fully_preprocessed_data/X_merged_preprocessed_new14.h5', 'r') as g:
        X_merged = np.zeros((55824, 50, 300), dtype='float64')
        g['data'].read_direct(X_merged)

    with h5py.File('data/fully_preprocessed_data/X_test_preprocessed_new14.h5', 'r') as h:
        X_test = np.zeros((6400, 50, 300), dtype='float64')
        h['data'].read_direct(X_test)

    # Load domain-specific labels
    label_domain_test = load_data_from_file(f'domain_and_label_test.pkl')
    label_domain_merged = load_data_from_file(f'domain_and_label_merged.pkl')    

    # Determine embedding type
    embedding_type = os.environ.get('EMBEDDING_TYPE', 'GENERAL')  # default is 'GENERAL'

    if embedding_type == 'SPECIFIC':
        domains = range(0, 16)  # Assuming domains are numbered from 1 to 16
        for domain in domains:
            print(f"Processing for domain {domain}...")

            # Filter data based on the domain
            X_merged_domain, label_domain_merged_domain = filter_data_by_domain(X_merged, label_domain_merged, domain)
            X_test_domain, label_domain_test_domain = filter_data_by_domain(X_test, label_domain_test, domain)
            # Concatenate train and test data
            data_domain = np.concatenate([X_merged_domain, X_test_domain])

            # Train and process for the specific domain
            autoencoder = build_autoencoder()
            autoencoder.compile(optimizer='adam', loss='mse')

            checkpoint_path = f"weights/autoencoder/specific/autoencoder_weights_with_unlabeled_{domain}.h5"
            checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path)
            es = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

            history = autoencoder.fit(data_domain, epochs=25, callbacks=[checkpoint, es], batch_size=16)
            processed_embeddings, processed_labels = process_and_save_embeddings(autoencoder, data_domain, label_domain_merged_domain, label_domain_test_domain, embedding_type)
    

    else:  # For 'GENERAL'
        # Load domain-specific labels for GENERAL type
        label_domain_test = load_data_from_file('domain_and_label_test.p')
        label_domain_merged = load_data_from_file('domain_and_label_merged.p')
        # Shuffle data
        X_merged, label_domain_merged = shuffle_data(X_merged, label_domain_merged, seed=42)
        X_test, label_domain_test = shuffle_data(X_test, label_domain_test, seed=43)

        # Concatenate train and test data
        data = np.concatenate([X_merged, X_test])

        # Train and process for general embeddings
        autoencoder = build_autoencoder()
        autoencoder.compile(optimizer='adam', loss='mse')

        checkpoint_path = "weights/autoencoder/general/autoencoder_weights_with_unlabeled.h5"
        checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path)
        es = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

        history = autoencoder.fit(data, data, epochs=25, callbacks=[checkpoint, es], batch_size=32)
        processed_embeddings, processed_labels = process_and_save_embeddings(autoencoder, data, label_domain_merged, label_domain_test, embedding_type, domain=domain)

if __name__ == "__main__":
    main()