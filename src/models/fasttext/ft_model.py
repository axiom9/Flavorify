import tensorflow as tf
from gensim.models import FastText
from gensim.utils import simple_preprocess
import numpy as np

class FastTextEmbedding(tf.keras.layers.Layer):
# class FastTextEmbedding(tf.keras.layers.experimental.preprocessing.PreprocessingLayer):
    def __init__(self, trained_ft_model_dir:str) -> None:
        super(FastTextEmbedding, self).__init__()
        self.trained_ft_model = FastText.load(trained_ft_model_dir)
        
    # INPUT IS A LIST example: ["Hey my name is anas", "this is a string"]
    # def call(self, input_list):
    #     out = np.zeros(shape=(len(input_list),60)) # num elements in inp list = len(input_list)
    #     for idx, inp in enumerate(input_list):
    #         sent_tokenized = simple_preprocess(inp)
    #         # we'll use mean pooling
    #         for word in sent_tokenized:
    #             out[idx,:] += self.trained_ft_model.wv[word]
    #         out[idx, :] /= len(sent_tokenized)
    #     return tf.convert_to_tensor(out)

     # INPUT IS A STRING : "Hey my name is Anas"
    def call(self, input):
        out = np.zeros(shape=(1,60))
        sent_tokenized = simple_preprocess(input)

        # we'll use mean pooling
        for word in input:
            out[:,:] += self.trained_ft_model.wv[word]
            
        return tf.convert_to_tensor(out / len(sent_tokenized))

    #changing shape
    # def call(self, input):
    #     out = np.zeros(shape=(60,))
    #     # sent_tokenized = simple_preprocess(input)

    #     # we'll use mean pooling
    #     for word in input:
    #         out[:] += self.trained_ft_model.wv[word]
        
    #     print(len(input))
    #     return tf.convert_to_tensor(out / len(input))

        # return tf.convert_to_tensor(out / len(sent_tokenized))
    # # INPUT IS A STRING : "Hey my name is Anas"
    # def call(self, input):
    #     out = np.zeros(shape=(1,60))
    #     sent_tokenized = simple_preprocess(input)

    #     # we'll use mean pooling
    #     for word in sent_tokenized:
    #         out[:,:] += self.trained_ft_model.wv[word]
        
    #     print(len(sent_tokenized))
    #     return tf.convert_to_tensor(out / len(sent_tokenized))
    #     # return tf.convert_to_tensor(out / len(sent_tokenized))


    # def call(self, input):
    #     # print(type(input))
    #     # input = tf.constant(input)
    #     # assert type(input) == str
    #     out = np.zeros(shape=(60,))
    #     sent_tokenized = simple_preprocess(input)

    #     # we'll use mean pooling
    #     for word in sent_tokenized:
    #         out += self.trained_ft_model.wv[word]
        
    #     print(len(sent_tokenized))
    #     return tf.expand_dims(tf.convert_to_tensor(out / len(sent_tokenized)), axis=1)
    #     # return tf.convert_to_tensor(out / len(sent_tokenized))

class FastTextModel(tf.keras.Model):
    def __init__(self, trained_ft_model_dir, num_classes:int=16) -> None:
        super(FastTextModel, self).__init__()
        # self.tokenizer=tf.keras.layers.TextVectorization(
        #     max_tokens=20,
        #     standardize='lower_and_strip_punctuation',
        #     split='whitespace',
        #     ngrams=None,
        # )
        self.fasttext_embeddings = FastTextEmbedding(trained_ft_model_dir)
        self.relu = tf.keras.layers.Activation('relu')
        self.softmax = tf.keras.layers.Activation('softmax')
        self.dense1 = tf.keras.layers.Dense(units=60)
        self.do = tf.keras.layers.Dropout(rate=0.35)
        self.dense2 = tf.keras.layers.Dense(units=num_classes)
        self.bn = tf.keras.layers.BatchNormalization()
        self.flatten = tf.keras.layers.Flatten()

    def call(self, input):
        x = self.fasttext_embeddings(input)
        print('Got here')
        x = self.flatten(x)
        x = self.dense1(x, training=True)
        print(x.shape)
        x = self.bn(x, training=True)
        print(x.shape)
        x = self.relu(x)
        x = self.do(x, training=False)
        print(x.shape)
        x = self.dense2(x, training=True)
        print(x.shape)
        return self.softmax(x)


