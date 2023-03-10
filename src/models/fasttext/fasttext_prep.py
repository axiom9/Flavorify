# from gensim.models import FastText
from gensim.utils import simple_preprocess
from sklearn.model_selection import train_test_split
import os
import importlib
import pandas as pd
import sys

base_dir = '/Users/anasputhawala/Desktop/Winterproj/'
sys.path.insert(0, base_dir)

from src.utils import pre_processing
from src.models.tfidf import config_tfidf
importlib.reload(pre_processing)
importlib.reload(config_tfidf)

df = pd.read_csv(config_tfidf.path_flag_christmastree, index_col=[0])

#Load data + custom pre-processing for fasttext
def custom_preprocess(text):
    return ' '.join(simple_preprocess(text))

def load_prep_save(df):
    df.Text = df.Text.apply(lambda row: pre_processing.remove_extra_spaces(row)) # remove extra spaces
    df.Text = df.Text.apply(lambda row: pre_processing.hashtag_mentions_removal(row)) # Remove mentions & hashtags
    df.Text = df.Text.apply(lambda row: custom_preprocess(row)) # simple pre-processing
    df.Label = df.Label.apply(lambda x: '__label__' + str(x))
    # (X_train, y_train), (X_val, y_val), (X_test, y_test) = pre_processing.load_and_split(df, 
    #                                                                                     train_ratio=0.7, 
    #                                                                                     validation_ratio=0.2,
    #                                                                                     test_ratio=0.1)
    train, test = train_test_split(df, test_size=0.2)
    train[['Label', 'Text']].to_csv('train.txt', 
                                    index = False, 
                                    sep = ' ',
                                    header = None)

    test[['Label', 'Text']].to_csv('test.txt', 
                                    index = False, 
                                    sep = ' ',
                                    header = None)

    print(f'Train and Test .txt files have successfully been saved at {os.getcwd()}')

def load_gensim():
    return df

def prep_gensim(df, tokenize:False, train_ratio, val_ratio, test_ratio):
    df.Text = df.Text.apply(lambda row: pre_processing.remove_extra_spaces(row)) # remove extra spaces
    df.Text = df.Text.apply(lambda row: pre_processing.hashtag_mentions_removal(row)) # Remove mentions & hashtags
    if tokenize:
        df.Text = df.Text.apply(lambda row: simple_preprocess(row)) # tokenize
    elif not tokenize:
        df.Text = df.Text.apply(lambda row: ' '.join(simple_preprocess(row))) # do not tokenize

    # df_train, df_test = train_test_split(df, test_size=0.15)
    # print(f'Length of training set: {len(df_train)}\nLength of testing set: {len(df_test)}')
    (xtr, ytr), (xvl, yvl), (xtst, ytst)= pre_processing.load_and_split(df, 
                                                                        train_ratio=train_ratio, 
                                                                        validation_ratio=val_ratio, 
                                                                        test_ratio=test_ratio, 
                                                                        shuffle=True, 
                                                                        num_classes=config_tfidf.num_classes)
    return (xtr, ytr), (xvl, yvl), (xtst, ytst)