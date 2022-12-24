'''Config file for tfidf training'''
import tensorflow as tf

num_classes = 18
vocab_size = 5000
standardize='lower_and_strip_punctuation'
split= 'whitespace'
lr = 0.001
batch_size=256
epochs=100

optimizer=tf.keras.optimizers.Adam(learning_rate=lr)
loss = tf.keras.losses.SparseCategoricalCrossentropy()
metrics=[tf.keras.metrics.CategoricalAccuracy()]