# hypermodel.py

import tensorflow as tf
import kerastuner as kt

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

