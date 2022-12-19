'''Pre-processing for data'''

import pandas as pd
# import nltk
from nltk.tokenize import word_tokenize
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
    1.
    2.
    3.
    4.
    5.
    The function will return a list with the values cleaned in the same locations as passed in'''
    pass

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