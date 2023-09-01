import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import gc
import pandas as pd
import gzip
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from io import StringIO
from datetime import datetime
import farmhash

class EpsilonGreedyBiddingModel:
    def __init__(self, num_arms, initial_epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01, epsilon_reset=1000):
        self.num_arms = num_arms
        self.initial_epsilon = initial_epsilon
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.epsilon_reset = epsilon_reset
        
        # Input layers
        continuous_input = keras.Input(shape=(8,), name='continuous_input')
        ohe_input = keras.Input(shape=(11,), name='ohe_input')  # Adjust shape based on total number of OHE columns
        embedding_input = keras.Input(shape=(1,), name='embedding_input')


        # Embedding for the website
        embedded_website = layers.Embedding(input_dim=10000, output_dim=50)(embedding_input)
        flattened_website = layers.Flatten()(embedded_website)

        # Concatenate all inputs
        concatenated_inputs = layers.Concatenate()([continuous_input, ohe_input, flattened_website])

        # Deep Neural Network Layers
        dense1 = layers.Dense(128, activation='relu')(concatenated_inputs)
        dense2 = layers.Dense(64, activation='relu')(dense1)
        dense3 = layers.Dense(32, activation='relu')(dense2)
        outputs = layers.Dense(num_arms)(dense3)

        self.model = keras.Model(inputs=[continuous_input, ohe_input, embedding_input], outputs=outputs)

        self.model.compile(optimizer='adam', loss='mse')

    def select_arm(self, continuous_input_val, ohe_input_val, embedding_input_val):
        # Exploration phase
        if tf.random.uniform(shape=(), minval=0, maxval=1) < self.epsilon:
            return tf.random.uniform(shape=(1,), minval=0, maxval=self.num_arms, dtype=tf.int32).numpy()[0]
        # Exploitation phase
        else:
            arm_values = self.model.predict({
                "continuous_input": np.array([continuous_input_val]),
                "ohe_input": np.array([ohe_input_val]),
                "embedding_input": np.array([embedding_input_val])
            },  verbose=3)
            return tf.argmax(arm_values, axis=1).numpy()[0]

    def update(self, continuous_input_val, ohe_input_val, embedding_input_val, chosen_arm, reward):

        y = self.model.predict({
            "continuous_input": np.array([continuous_input_val]),
            "ohe_input": np.array([ohe_input_val]),
            "embedding_input": np.array([embedding_input_val])
        },  verbose=3)
        y[0][chosen_arm] = reward  # Use the observed reward as the target for the chosen arm
        self.model.fit({
            "continuous_input": np.array([continuous_input_val]),
            "ohe_input": np.array([ohe_input_val]),
            "embedding_input": np.array([embedding_input_val])
        }, y, verbose=0)
        
        # Decay epsilon
        self.decay_epsilon()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)

    def reset_epsilon(self):
        self.epsilon = self.initial_epsilon
        
    def checkpoint(self):
        self.model.save('keras_model.h5')
        # Redirect standard output to a file
        with open('model_summary.txt', 'w') as f:
            # Temporarily set the standard output to the file
            original = sys.stdout
            sys.stdout = f
            self.model.summary()
            # Reset the standard output back to its original value
            sys.stdout = original