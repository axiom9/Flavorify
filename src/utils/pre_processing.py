'''Pre-processing for data'''

import pandas as pd
import numpy as np
import nltk
import tensorflow as tf
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import unicodedata
from scipy.stats import zscore
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, '/Users/anasputhawala/Desktop/Winterproj')
from src.models.tfidf import config_tfidf
import importlib
importlib.reload(config_tfidf)


# add appropriate words that will be ignored in the analysis
ADDITIONAL_STOPWORDS = []
import re

def sculpt_df(text_file_loc: str, labels_loc: str) -> pd.DataFrame:
    '''Pass in the location of the .text file and the .labels and this function will output a dataframe
    with two columns:
        1. Text (cleaned tweet using CodaLab's pre-processer)
        2. Labels'''
    with open(text_file_loc) as f:
        lines = [line.rstrip() for line in f]

    with open(labels_loc) as f2:
        labels = [line.rstrip() for line in f2]

    return pd.DataFrame(list(zip(lines, labels)), columns=['Text', 'Label'])

def combine_labels(df: pd.DataFrame, labels_to_combine:list) -> pd.DataFrame:
    '''Pass in the entire dataframe along with a list of labels to combine and this function will perform the
    combining and return out the dataframe.
    
    Note: When you pass labels to combine, the LAST entry in your list should be what you want to
    combine all the previous labels with. Example:
    [13, 8, 0] --> You want labels 13 and 8 to simply be over-written to label 0.'''

    assert 'Label' in df.columns, "If you're applying this function to combine labels there must be a column titled 'Label'"

    def apply_func(val):
        if val in labels_to_combine[:-1]:
            val = labels_to_combine[-1]
        return val

    df.Label = df.Label.apply(lambda row: apply_func(row))
    return df


def convert_to_int(df: pd.DataFrame, column='Label') -> pd.DataFrame:
    '''Convert strings in a given column to integers'''
    df[column] = df[column].apply(lambda row: int(row))
    return df


def clean_text(text: list) -> list:
    '''Pass in the text as a list and this function will perform the following cleaning:
    1. Using partition method to remove the "@" and everything that follows it
    2.
    3.
    4.
    5.
    The function will return a list with the values cleaned in the same locations as passed in'''
    head, sep, tail = text.partition('@')
    return head

def remove_empty_strings(original_df:pd.DataFrame) -> pd.DataFrame:
    '''After performing some cleaning, some of the text columns may have empty strings. This function simply
    locates those rows where there is an empty string and drops that entire row. It will
    return a new dataframe where it deals with the empty text rows (tweets) by dropping the row'''
    df = original_df.copy()
    df.Text.replace('', np.nan, inplace=True)
    df.dropna(subset=['Text'], inplace=True)
    df = df.reset_index(drop=True)

    print(f'Found {len(original_df) - len(df)} containing empty string for text and dropped them')
    return df



def tokenize(text: str, lowercase:bool) -> list:
    '''Pass in UNTOKENIZED text as a string (note; this is the default format of the column
    "Text" in the dataframe and obtain a tokenized version of the string'''
    assert lowercase in [True, False]

    #Removing punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    if lowercase:
        #Tokenizing and returning
        return word_tokenize((text).lower())
    elif not lowercase:
        return word_tokenize(text)


def basic_clean(text):
        """
        A simple function to clean up the data. All the words that
        are not designated as a stop word is then lemmatized after
        encoding and basic regex parsing are performed.
        """
        wnl = nltk.stem.WordNetLemmatizer()
        stopwords = nltk.corpus.stopwords.words('english') + ADDITIONAL_STOPWORDS
        text = (unicodedata.normalize('NFKD', text)
        .encode('ascii', 'ignore')
        .decode('utf-8', 'ignore')
        .lower())
        words = re.sub(r'[^\w\s]', '', text).split()
        return [wnl.lemmatize(word) for word in words if word not in stopwords]


def drop_outliers(original_df: pd.DataFrame, labels_to_clean:list, tokenized_text:pd.Series, z:int) -> pd.DataFrame:
    '''This function drops outliers by identifying which tweets in a given label are for example greater than 3
    z-score for mean average length of tweets in that specific label
    This function will return a copy of the original df'''
    
    assert "Label" in original_df.columns
    idxs_to_drop = []

    for label in labels_to_clean:
        tmp = tokenized_text[original_df.Label==label].apply(lambda row: len(row)) # Get the number of tokens
        # now we can filter for > 3 or < -3 z-score for outliers
        to_remove = (zscore(tmp) > z) | (zscore(tmp) < -z)
        idxs_to_drop.extend(np.where(to_remove)[0].tolist()) # append all indices that need to be dropped into
                                                            # 'idxs_to_drop'
        print(f'For label: {label} there are a total of {sum(to_remove)} outliers when considering z={z}')

    print(f'Total number of data points cleaned / removed {len(idxs_to_drop)}')
    
    return original_df.drop(idxs_to_drop, axis=0).reset_index(drop=True)

def drop_random_rows(original_df: pd.DataFrame, labels_to_clean: list, target_size) -> pd.DataFrame:
    assert "Label" in original_df.columns
    idxs_to_drop = []
    for label in labels_to_clean:
        num_to_drop = sum(original_df.Label == label) - target_size
        if num_to_drop <= 0: #i.e. we're trying to drop more than currently exist for that current label
            print(f'Skipping label {label} since there are less values in this class to begin with')
            break
        class_indices = original_df[original_df.Label == label].index

        idcs_drop = np.random.choice(class_indices, size=num_to_drop, replace=False)
        print(f'For label = {label} there are a total of {len(idcs_drop)} rows that will be dropped to get to a target size for this class to {target_size}')

        idxs_to_drop.extend(idcs_drop.tolist())
    return original_df.drop(labels=idxs_to_drop, axis=0).reset_index(drop=True)


def split_data(df, train_ratio:float, validation_ratio:float, test_ratio:float, shuffle:bool=True):
    '''Splits data into training, validation, test for training the model:
    i.e. training can be .70, val can be .20, test can be .10'''
    assert "Text" and "Label" in df.columns, "Your dataframe must have a column for text and its corresponding label"
    dataX = df.Text
    dataY = df.Label

    # train is now 75% of the entire data set
    # the _junk suffix means that we drop that variable completely
    x_train, x_test, y_train, y_test = train_test_split(dataX, dataY, test_size=1 - train_ratio, shuffle=shuffle)

    # test is now 10% of the initial data set
    # validation is now 15% of the initial data set
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio), shuffle=shuffle) 

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def load_and_split(df, train_ratio:float, validation_ratio:float, test_ratio:float, shuffle:bool=True, num_classes:int=config_tfidf.num_classes):
    assert "Text" and "Label" in df.columns, "Your dataframe must have a column for text and its corresponding label"
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = split_data(df, 
                                                                      train_ratio,
                                                                      validation_ratio,
                                                                      test_ratio)

    # to categorical for CCE loss
    y_train = tf.keras.utils.to_categorical(np.array(y_train), num_classes=num_classes)
    y_val = tf.keras.utils.to_categorical(np.array(y_val), num_classes=num_classes)
    y_test = tf.keras.utils.to_categorical(np.array(y_test), num_classes=num_classes)

    print(f'Shape of X_train: {x_train.shape}\nShape of y_train: {y_train.shape}\n\nShape of X_val: {x_val.shape}\nShape of y_val: {y_val.shape}\n\nShape of X_test: {x_test.shape}\nShape of y_test: {y_test.shape}')
    return (np.array(x_train), y_train), (np.array(x_val), y_val), (np.array(x_test), y_test)

def remove_extra_spaces(text):
    '''This function will remove any extra spaces within a tweet, example: "hey   my  name is    anas" -> hey my name is anas'''
    return " ".join(text.split())


def re_order_labels(label):
    '''Need to re-order labels because we combined some. This will impact us later when we declare loss as
    Sparse Categorical Crossentropy unless we fix it via this function!
    New mapping is as follows:
    see emoji_labels_updated.jpg in /data/emoji_labels_updated.jpg'''

    map = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 9:9, 10:13, 11:11, 12:12, 14:14, 15:15, 16:5, 17:17, 18:13, 19:8}
    return map[label]

def drop_specific_labels(df:pd.DataFrame, labels):
    '''Given the original dataframe and labels this will clean the dataframe some more and drop the label(s)
    labels must be of type list (example: you want to drop label 5 --> [5], you want to drop 5 and 10 --> [5,10]'''
    indcs_to_drop = []

    assert isinstance(labels, list), 'Labels passed in must be of type list'
    assert 'Label' in df.columns, 'Label column must exist in the dataframe you pass in'

    for label in labels:
        to_drop = df.Label==label
        lst_indcs_label = list(np.where(to_drop)[0])
        indcs_to_drop.extend(lst_indcs_label)
        print(f'For label: {label} there were a total of {len(lst_indcs_label)} rows that will be dropped')

    return df.drop(labels=indcs_to_drop, axis=0).reset_index(drop=True)

def hashtag_mentions_removal(text):
    # mentions = re.findall("@([a-zA-Z0-9_]{1,50})", text)
    # hashtags = re.findall("#([a-zA-Z0-9_]{1,50})", text)
    clean_tweet = re.sub("@[A-Za-z0-9_]+","", text)
    clean_tweet = re.sub("#[A-Za-z0-9_]+","", clean_tweet)
    clean_tweet = re.sub('[@#]', "", clean_tweet)
    return remove_extra_spaces(clean_tweet)