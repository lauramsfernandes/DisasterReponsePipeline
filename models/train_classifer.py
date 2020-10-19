# Import Libraries
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import re
import numpy as np
import pandas as pd
import sqlite3
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sqlalchemy import create_engine
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# Load data from database
engine = create_engine('sqlite:///disaster_messages.db')
df = pd.read_sql_table("message", con = engine)

# Tokenization function
url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

def tokenize(text):
    """This function replace all url adresses for a string and

    Args:
        text ([type]): [description]

    Returns:
        [type]: [description]
    """
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# Classification Report Function
def df_classification_report(y_true,y_pred):
    """[summary]

    Args:
        y_true ([float]): float list of all true results
        y_pred ([float]): float list of all predictions

    Returns:
        [type]: a MultiIndex DataFrame cointaining all results of classification_report function from sklearn.
        The classification_report function builds a text report showing the precision, recall, f1_score and support.
    """

    # Set levels of MultiIndex DataFrame
    first = y_true.columns.values.tolist()
    second = ['0.0','1.0','avg/total']

    # Set column names
    cols = ['precision','recall','f1_score', 'support']

    # Set MultiIndex
    mult_ind = [first,second]
    mult_ind = pd.MultiIndex.from_product(mult_ind, names=['first', 'second'])

    # Create a dict to be used as data
    d = {}

    ## Iterates classification_report through each column and updates the dict data
    j=0
    for i in np.arange(0,36,1):
        lenght = len(classification_report(y_true.iloc[i].tolist(), y_pred[i].tolist()))
        if lenght == 214:
            d[mult_ind[j]] = classification_report(y_true.iloc[i].tolist(), y_pred[i].tolist()).split()[5:9]
            j+=1
            d[mult_ind[j]] = classification_report(y_true.iloc[i].tolist(), y_pred[i].tolist()).split()[10:14]
            j+=1
            d[mult_ind[j]] = classification_report(y_true.iloc[i].tolist(), y_pred[i].tolist()).split()[-4:]
            j+=1
        elif lenght == 161:
            true_or_false = classification_report(y_true.iloc[i].tolist(), y_pred[i].tolist()).split()[4]
            if true_or_false == '0.0':
                d[mult_ind[j]] = classification_report(y_true.iloc[i].tolist(), y_pred[i].tolist()).split()[5:9]
                j+=1
                d[mult_ind[j]] = [np.nan, np.nan, np.nan, np.nan]
                j+=1
                d[mult_ind[j]] = classification_report(y_true.iloc[i].tolist(), y_pred[i].tolist()).split()[-4:]
                j+=1
            if true_or_false == '1.0':
                d[mult_ind[j]] = [np.nan, np.nan, np.nan, np.nan]
                j+=1
                d[mult_ind[j]] = classification_report(y_true.iloc[i].tolist(), y_pred[i].tolist()).split()[5:9]
                j+=1
                d[mult_ind[j]] = classification_report(y_true.iloc[i].tolist(), y_pred[i].tolist()).split()[-4:]
                j+=1
    # Creates the DataFrame
    df_metric = pd.DataFrame(data=d, index=cols)
    df_metric = df_metric.T

    return df_metric

# Create a ML pipeline with NLTK, scikit-learn's Pipeline and GridSearchCV
# OUTPUT: a final model that uses 'message' column to predict classifications for 36 categories (multi-output classification)

# Export model to a pickle file