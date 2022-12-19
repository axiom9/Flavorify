'''Various different forms of analysis'''

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time

import pre_processing
import importlib
import config
importlib.reload(pre_processing)
importlib.reload(importlib)
importlib.reload(config)

def plot_average_num_words(num_elements: pd.Series, original_df: pd.DataFrame, labels_to_analyze:list, num_bins:int):
    '''This function takes in a list of labels (i.e. can be from 0 - 19 for all 20 labels of emojis)
    and analyzes the average number of words in a given TOKENIZED tweet (NOTE it must be tokenized,
    so call pre_processing.tokenize() first if needed) by plotting histograms and overlaying some basic summary statistics
    (such as mean, median, std). 

    :param num_element must be a pandas series which contains number of elements in each respective
    tweet (note the rows in num elements correspond to the rows from original_df)
    '''

    assert len(num_elements) == len(original_df), 'Something went wrong, the number of elements in "num_elements" must be equivalent to the len of original_df'
    assert 'Label' in original_df.columns, 'You need to have a "Label" column in the dataframe for which to do the lookup'
    for label in labels_to_analyze:
        ser = num_elements[original_df.Label == label]
        fig, ax = plt.subplots()
        ax.hist(ser, bins=20);
        # print(f'Currently analyzing label: {label} -> {config.mapping_rev[label]}')
        plt.xlabel('Number of words');
        plt.ylabel('Count');
        plt.title(f'Histogram of the number of words for label: {label}');
        statistics = f'Mean: {round(np.mean(ser), 2)}, Std: {round(np.std(ser), 2)}, Median: {round(np.median(ser), 2)}, Label: {label}, {config.mapping_rev[label]}';
        plt.annotate(text=statistics, xy=(round(max(ser) * 0.5), round(int(max(ax.get_yticks())) * 0.5)));
        



    
