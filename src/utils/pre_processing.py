'''Pre-processing for data'''

import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import unicodedata
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