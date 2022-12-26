'''Config file for tfidf training'''
import tensorflow as tf
import pandas as pd

path = '/Users/anasputhawala/Desktop/Winterproj/src/utils/scraped_tweets2.csv'
path_flag_christmastree = '/Users/anasputhawala/Desktop/Winterproj/src/utils/scraped_tweets3_flag_christmas.csv'
num_classes = len(pd.read_csv(path_flag_christmastree).Label.value_counts())
vocab_size = 5000
standardize='lower_and_strip_punctuation'
split= 'whitespace'

loss = tf.keras.losses.CategoricalCrossentropy()
metrics=[tf.keras.metrics.CategoricalAccuracy()]