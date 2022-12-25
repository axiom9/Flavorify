'''Config file for tfidf training'''
import tensorflow as tf
import pandas as pd

path = '/Users/anasputhawala/Desktop/Winterproj/src/utils/scraped_tweets2.csv'
num_classes = len(pd.read_csv(path).Label.value_counts())
vocab_size = 5000
standardize='lower_and_strip_punctuation'
split= 'whitespace'

loss = tf.keras.losses.CategoricalCrossentropy()
metrics=[tf.keras.metrics.CategoricalAccuracy()]