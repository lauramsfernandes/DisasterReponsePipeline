# Import libraries
import pandas as pd
import numpy as np
import matplotlib as plt
from sqlalchemy import create_engine

# Summary: read the dataset, clean the data, and then store it in a SQLite database

## E - Extracting data

# Load messages dataset
messages = pd.read_csv(r'data\messages.csv')

# Load categories dataset
categories = pd.read_csv(r'data\categories.csv')

## T - Transforming data

# Merge messages and categories dataset into a new dataframe
df = pd.merge(categories, messages)

# Split `categories` into separate category columns
categories = categories.loc[:,'categories'].str.split(pat=';', expand=True)

# Select the first row of the categories dataframe in order to set the columns name
row = categories.iloc[0].values

# Clean the columns name removing the last two characters
category_colnames = list(map(lambda name: name[slice(-2)], row))

# Rename `categories` columns
categories.columns = category_colnames

# Convert category values to just numbers 0 or 1
for column in categories:
    # Set each value to be the last character of the string
    categories[column] = categories[column].str.get(-1)
    # Convert each value from string to numeric
    categories[column] = pd.to_numeric(categories[column])

# Replace `categories` column in df with new category columns
df.drop('categories', axis=1, inplace=True)
df = pd.concat([df,categories], axis=1)

#Remove duplicates
df.drop_duplicates(inplace=True)

# L - Loading data
engine = create_engine('sqlite:///InsertDatabaseName.db')
df.to_sql('InsertTableName', engine, index=False)

