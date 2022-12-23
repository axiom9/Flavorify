# Get labels
# Get text
# Split data in to training, validaiton and test
import sys
sys.path.insert(0, '/Users/anasputhawala/Desktop/Winterproj')
import numpy as np
from importlib import reload

from src.utils import pre_processing
reload(pre_processing)

#import pandas as pd

def load_and_split(df, train_ratio:float, validation_ratio:float, test_ratio:float, shuffle:bool=True):
    assert "Text" and "Label" in df.columns, "Your dataframe must have a column for text and its corresponding label"
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = pre_processing.split_data(df, 
                                                                                    train_ratio,
                                                                                    validation_ratio,
                                                                                    test_ratio)
    print(f'Shape of X_train: {len(x_train)}\nShape of y_train: {len(y_train)}\n\nShape of X_val: {len(x_val)}\nShape of y_val: {len(y_val)}\n\nShape of X_test: {len(x_test)}\nShape of y_test: {len(y_test)}')
    return (x_train, np.array(y_train)), (x_val, np.array(y_val)), (x_test, np.array(y_test))


# Build tfidf vectorizer
# Get tfidf of the text
# Pass through dense layers, softmax classifier, and train the model
# evaluate the performance


