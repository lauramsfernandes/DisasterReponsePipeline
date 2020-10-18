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
def df_classification_report(y_pred, y_true):
    """Runs trought each column from y_true series sklearn's classification_report and returns
    a new MultiIndex DataFrame cointaining the values.

    Args:
        y_pred ([float]): predicted values list.
        y_true ([float]): true value series.

    Returns:
        [pd.DataFrame]: Classification Report with all values from each column.
    """

    first = y_true.columns.values.tolist()
    second = ['0.0','1.0','avg/total']

    cols = ['precision','recall','f1_score', 'support']

    mult_ind = [first,second]
    mult_ind = pd.MultiIndex.from_product(mult_col, names=['first', 'second'])

    # Dict data
    d = {}

    # Iterates classification_report through each column
    for i in np.arange(0,36,1):
        if mult_ind[i][1] == '0.0':
            res = classification_report(y_true.iloc[i].tolist(), y_pred[i].tolist()).split()[5:9]
        elif mult_ind[i][1] == '1.0':
            res = classification_report(y_true.iloc[i].tolist(), y_pred[i].tolist()).split()[10:14]
        elif mult_ind[i][1] == 'avg/total':
            res = classification_report(y_true.iloc[i].tolist(), y_pred[i].tolist()).split()[-4:]

        # Update the dict data
        d[mult_ind[i]] = res
    # Creates the DataFrame
    df_metric = pd.DataFrame(data=d, index=ind)
    df_metric = df_metric.T

    return df_metric

# Create a ML pipeline with NLTK, scikit-learn's Pipeline and GridSearchCV
# OUTPUT: a final model that uses 'message' column to predict classifications for 36 categories (multi-output classification)

# Export model to a pickle file