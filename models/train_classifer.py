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
# Split the data into train and test set


# Create a ML pipeline with NLTK, scikit-learn's Pipeline and GridSearchCV
# OUTPUT: a final model that uses 'message' column to predict classifications for 36 categories (multi-output classification)

# Export model to a pickle file