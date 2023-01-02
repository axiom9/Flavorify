import tensorflow as tf
import pandas as pd
import numpy as np
import importlib
import re
import sys
base_dir = '/Users/anasputhawala/Desktop/Winterproj/'
sys.path.insert(0, base_dir)

from src.utils import pre_processing
importlib.reload(pre_processing)

# def custom_standardize(text):
#         text = pre_processing.hashtag_mentions_removal(text) #removing hashtags and mentions
#         text = re.sub(r"[^\w\s]", "", text) # removing punctuation
#         text = pre_processing.remove_extra_spaces(text) # removing extra spaces that may result from removing punctuation
#         return tf.strings(text.lower())

# Define model
class VanillaModel(tf.keras.Model):
    def __init__(self, vocab_size, maxlen, embedding_dim, X_train) -> None:
        super(VanillaModel, self).__init__()
        self.vocab_size = vocab_size
        self.maxlen = maxlen
        self.embedding_dim = embedding_dim

        self.textvec = tf.keras.layers.TextVectorization(
            standardize='lower_and_strip_punctuation',
            max_tokens=vocab_size,
            output_mode='int',
            output_sequence_length=maxlen) 

        self.textvec.adapt(X_train) # adapting to our training data

        self.emb = tf.keras.layers.Embedding(vocab_size, embedding_dim, name="embedding")
        self.lstm = tf.keras.layers.LSTM(256, return_sequences=True, dropout=0.2)
        self.lstm2 = tf.keras.layers.LSTM(128, return_sequences=False, dropout=0.2)
        self.bn = tf.keras.layers.BatchNormalization()
        self.flat = tf.keras.layers.Flatten()
        self.do = tf.keras.layers.Dropout(rate=0.30)
        self.dense = tf.keras.layers.Dense(64)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.Activation('relu')
        self.do2 = tf.keras.layers.Dropout(rate=0.25)
        self.clf = tf.keras.layers.Dense(16, 'softmax')

    def call(self, input):
        x = self.textvec(input)
        x = self.emb(x)
        x = self.lstm(x)
        x = self.lstm2(x)
        x = self.bn(x)
        x = self.flat(x)
        x = self.do(x)
        x = self.dense(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.do2(x)
        x = self.clf(x)
        return x
