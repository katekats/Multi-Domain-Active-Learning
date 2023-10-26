import numpy as np
import tensorflow as tf
import kerastuner as kt
import pickle as pkl

TRAIN_GEN_EMBEDDINGS_PATH = 'data/sentence_embeddings/general/unsorted/sentemb/sentemb_unlabeled3.p'
TRAIN_LABELS_PATH = 'data/sentence_embeddings/general/unsorted/label_domain/label_domain_train_sentemb_unlabeled3.p'
TEST_LABELS_PATH = 'data/sentence_embeddings/general/unsorted/label_domain/label_domain_test_sentemb_unlabeled3.p'
TRAIN_CLEANED_DATA_PATH = 'data/cleaned_data/merged_cleaned.p'
TEST_CLEANED_DATA_PATH = 'data/cleaned_data/test_cleaned.p



class MyHyperModel(kt.HyperModel):
    def build(self, hp):
        INPUT_SIZE = 300
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        hp_units1 = hp.Int('units1', min_value=100, max_value=500, step=100)
      
        inp_gen = tf.keras.Input(shape=(1, INPUT_SIZE))
        inp_spec = tf.keras.Input(shape=(1, INPUT_SIZE))
        merged = tf.keras.layers.Concatenate()([inp_gen, inp_spec])
        merged = tf.keras.layers.Dense(hp_units1, activation='sigmoid')(merged)
        merged = tf.keras.layers.Dropout(hp.Float('dropout', 0, 0.5, step=0.1, default=0.5))(merged)
        merged = tf.keras.layers.Dense(1, activation='sigmoid')(merged)

        model = tf.keras.Model([inp_gen, inp_spec], merged)
        model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate), metrics=['accuracy'])
        return model

def run_hyperparameter_tuning():
    tuner2 = kt.BayesianOptimization(
    MyHyperModel(),
    objective="val_accuracy",
    max_trials=50,
    overwrite=True,
    num_initial_points=25,
    alpha=0.001,
    beta=2.6
)

def load_from_file(filename):
    with open(filename, "rb") as file:
        data = pkl.load(file)
    return data

def preprocess_data(data):
    return np.expand_dims(np.asarray(data), 1) 
    
def main():    
    # Load general sentence embeddings
    X_train_gen = load_from_file("X_train_gen.pkl")
    X_val_gen = load_from_file("X_val_gen.pkl")
    X_train_spec = load_from_file("X_train_spec.pkl")
    X_val_spec = load_from_file("X_val_spec.pkl")
    y_train = load_from_file("y_train.pkl")
    X_val = load_from_file("y_val.pkl")
    
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    tuner.search([preprocess_data(X_train_gen), preprocess_data(X_train_spec)], y_train, 
                 validation_data=([preprocess_data(X_val_gen), preprocess_data(X_val_spec)], y_val),
        callbacks=[es]
    )
    )
    # Get the best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"""
    The hyperparameter search is complete. The optimal batch_size
    layer is {best_hps.get('batch_size')}, the optimal learning rate for the optimizer
    is {best_hps.get('learning_rate')}, the optimal dropout rate is {best_hps.get('dropout')}, the optimal number of epochs is {best_hps.get('epochs')} the optimal number of units1 is {best_hps.get('units1')} and th.
    """)
    
    with open("best_hyperparameters.pkl", "wb") as f:
        pkl.dump(best_hps, f)

if __name__ == "__main__":
    main()

