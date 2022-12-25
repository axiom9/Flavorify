# Get labels
# Get text
# Split data in to training, validaiton and test
import sys
sys.path.insert(0, '/Users/anasputhawala/Desktop/Winterproj')
import numpy as np
from importlib import reloads

import tensorflow as tf
from tensorflow.keras import regularizers


from src.utils import pre_processing
import config_tfidf
reload(pre_processing)
reload(config_tfidf)

class TfidfModel():
    def __init__(self, dir):
        self.num_classes = config_tfidf.num_classes
        self.vocab_size = config_tfidf.vocab_size
        self.standardize= config_tfidf.standardize
        self.split = config_tfidf.split
        self.loss = config_tfidf.loss
        self.metrics = config_tfidf.metrics
        self.dir = dir

        self.Model = None

    def build(self, X):

        '''Need to pass in "X train" as a np array in order to adapt and build tfidf vectorizer'''
        tfidf_layer = tf.keras.layers.TextVectorization(
                                            standardize = self.standardize,
                                            split = self.split,
                                            max_tokens=self.vocab_size,
                                            output_mode='tf-idf',
                                            pad_to_max_tokens=False)
        
        tfidf_layer.adapt(X, batch_size=256)

        # Build model
        inputs = tf.keras.Input(shape=(1,), dtype='string', name='Input_Layer')
        x = tfidf_layer(inputs)
        x = tf.keras.layers.Dense(units=1024, name='Dense1')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Dropout(rate=0.35, name='dropout1')(x)
        x = tf.keras.layers.Dense(units=512, name='Dense2', kernel_regularizer=regularizers.L1(l1=1e-4))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Dense(units=256, name='Dense3', kernel_regularizer=regularizers.L1(l1=1e-4))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Dropout(rate=0.35)(x)
        x = tf.keras.layers.Dense(units=64, name='Dense4', kernel_regularizer=regularizers.L1(l1=1e-4))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        outputs = tf.keras.layers.Dense(units=self.num_classes, activation='softmax', name='Classifier')(x)
        # outputs = tf.keras.activations.softmax(x, axis=-1)

        self.Model = tf.keras.Model(inputs=inputs, outputs=outputs, name="Tfidf_Model")
        print('Model Successfully Built')
        
    def train(self, x_train, y_train, x_val, y_val, epochs, lr, batch_size, val_model:bool=True):
        # compile model
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr)
        self.Model.compile(optimizer=optimizer, loss=self.loss, metrics=self.metrics)
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=5),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.75, patience=3)
        ]

        if val_model:
            history = self.Model.fit(x_train, 
                                    y_train, 
                                    batch_size=batch_size, 
                                    epochs=epochs, 
                                    callbacks=callbacks,
                                    validation_data=(x_val, y_val))
        
        if not val_model:
            history = self.Model.fit(np.concatenate((x_train, x_val)), # since we aren't in validation mode, we concatenate the data to provide better data for model to learn with
                                    np.concatenate((y_train, y_val)), 
                                    batch_size=batch_size, 
                                    epochs=epochs, 
                                    callbacks=callbacks)
                                    # validation_data=(x_val, y_val))

        return history

    def test(self, x, y, batch_size:int=128):
        return self.Model.evaluate(x, y, batch_size)

    def predict(self, x, batch_size:int=128):
        '''Note: need to apply argmax over predictions to get actual labels'''
        return self.Model.predict(x, batch_size=batch_size)

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

    def save_model(self):
        # fpath=self.dir+'.h5'
        fpath=self.dir
        self.Model.save(filepath=fpath, save_format='tf')
        print(f'Model successfully saved at {fpath}')

    def load_model(self):
        # fpath=self.dir+'.h5'
        fpath=self.dir
        print(f'Attempting to load model from {fpath}')
        self.Model = tf.keras.models.load_model(fpath)
        print(f'Model successfully loaded')
