import sys
import sqlite3
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    
    #Pull messages and categories data
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    #Merge messages and categories data
    df = pd.concat([messages, categories], axis=1)
    
    return df


def clean_data(df):
    
    # [1] Clean categories data
    
    # split data between word and number by ';'
    categories = df['categories'].str.split(pat = ';', expand=True)

    # get title name to use as column name of categories
    title = {}
    for i in range(categories.shape[1]):
        title[i] = categories.iloc[0].str.split('-')[i][0]
    
    # replace column name with title
    categories.rename(columns = title, inplace = True)
    
    # convert category value to 0 or 1 (except 'related' column
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x : x.split('-')[1])
        # convert column from string to numeric
        categories[column] = categories[column].astype('int')
 
    # drop the original categories column from `df`
    df = df.drop(['categories'], axis = 1)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    # drop duplicates
    df = df[df.duplicated() == False]
    
    return df

def save_data(df, database_filename):
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('NPdatabase', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()