# Get labels
# Get text
# Split data in to training, validaiton and test
import sys
sys.path.insert(0, '/Users/anasputhawala/Desktop/Winterproj')
import numpy as np
from importlib import reload
import tensorflow as tf

from src.utils import pre_processing
import config_tfidf
reload(pre_processing)
reload(config_tfidf)

#import pandas as pd

def load_and_split(df, train_ratio:float, validation_ratio:float, test_ratio:float, shuffle:bool=True):
    assert "Text" and "Label" in df.columns, "Your dataframe must have a column for text and its corresponding label"
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = pre_processing.split_data(df, 
                                                                                    train_ratio,
                                                                                    validation_ratio,
                                                                                    test_ratio)
    print(f'Shape of X_train: {len(x_train)}\nShape of y_train: {len(y_train)}\n\nShape of X_val: {len(x_val)}\nShape of y_val: {len(y_val)}\n\nShape of X_test: {len(x_test)}\nShape of y_test: {len(y_test)}')
    return (np.array(x_train), np.array(y_train)), (np.array(x_val), np.array(y_val)), (np.array(x_test), np.array(y_test))

class TfidfModel():
    def __init__(self):
        self.num_classes = config_tfidf.num_classes
        self.vocab_size = config_tfidf.vocab_size
        self.standardize= config_tfidf.standardize
        self.split = config_tfidf.split
        self.optimizer = config_tfidf.optimizer
        self.loss = config_tfidf.loss
        self.metrics = config_tfidf.metrics
        self.batch_size = config_tfidf.batch_size
        self.epochs= config_tfidf.epochs

        self.Model = None

    def build(self, X):

        '''Need to pass in "X train" as a np array in order to adapt and build tfidf vectorizer'''
        tfidf_layer = tf.keras.layers.TextVectorization(
                                            standardize = self.standardize,
                                            split = self.split,
                                            max_tokens=self.vocab_size,
                                            output_mode='tf-idf',
                                            pad_to_max_tokens=False)
        
        tfidf_layer.adapt(X)

        # Build model
        inputs = tf.keras.Input(shape=(1,), dtype='string', name='Input_Layer')
        x = tfidf_layer(inputs)
        x = tf.keras.layers.Dense(units=1024, name='Dense1')(x)
        x = tf.nn.relu(tf.keras.layers.BatchNormalization()(x))
        x = tf.keras.layers.Dropout(rate=0.35, name='dropout1')(x)
        x = tf.keras.layers.Dense(units=512, name='Dense2')(x)
        x = tf.nn.relu(tf.keras.layers.BatchNormalization()(x))
        x = tf.keras.layers.Dense(units=256, name='Dense3')(x)
        x = tf.nn.relu(tf.keras.layers.BatchNormalization()(x))
        x = tf.keras.layers.Dropout(rate=0.25)(x)
        x = tf.keras.layers.Dense(units=64, name='Dense4')(x)
        x = tf.nn.relu(tf.keras.layers.BatchNormalization()(x))
        x = tf.keras.layers.Dense(units=self.num_classes, name='Classifier')(x)
        outputs = tf.keras.activations.softmax(x, axis=-1)
        self.Model = tf.keras.Model(inputs=inputs, outputs=outputs, name="Tfidf_Model")
        print('Model Successfully Built')
        
    def train(self, x_train, y_train, x_val, y_val):
        # compile model
        self.Model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=5),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.75, patience=3)
        ]

        history = self.Model.fit(x_train, 
                                y_train, 
                                batch_size=self.batch_size, 
                                epochs=5, 
                                callbacks=callbacks,
                                validation_data=(x_val, y_val))

        return history

    def test(self, x, y):
        pass

    def reset_weights(self):
        '''Resetting weights in model'''
        for l in self.Model.layers:
            if hasattr(l, "kernel_initializer"):
                l.kernel.assign(l.kernel_initializer(tf.shape(l.kernel)))
            if hasattr(l, "bias_initializer"):
                l.bias.assign(l.bias_initializer(tf.shape(l.bias)))
            if hasattr(l, "recurrent_initializer"):
                l.recurrent_kernel.assign(l.recurrent_initializer(tf.shape(l.recurrent_kernel)))
        print("Weights Reset")

    def summary(self):
        return self.Model.summary()
