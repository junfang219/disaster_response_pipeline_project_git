import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """ load the csv data from data file

    :param messages_filepath: disaster message data path
    :param categories_filepath: messages' corresponding categories data path
    :return: a merged dataframe
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on=['id'])

    return df


def clean_data(df):
    """ clean the merged dataframe and get the desired values

    :param df: read the dataframe after loading and merging the data files
    :return: a clean dataframe
    """

    # separate the categories columns by ;
    categories = df['categories'].str.split(";", expand=True)

    # assign column names by using the first few words (excluding last two numbers)
    row = categories.iloc[1]
    category_colnames = row.apply(lambda x: x[0:-2]).tolist()
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str.get(-1)
        # convert column from string to numeric
        categories[column] = categories[column].astype(np.int)

    # drop the original categories column from `df`
    df = df.drop(['categories'], axis=1)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df = df.drop_duplicates()

    # make relate a binary variable
    df['related'] = df['related'].map(lambda x: 1 if x == 2 else x)

    return df

def save_data(df, database_filename):
    """save cleaned dataframe to a database by using the sqlalchemy library create_engine function
    arg: df is the cleaned dataframe that needs to be stored
    arg: database_filename, the desired database file name
    """
    engine = create_engine('sqlite:///' + database_filename)
    table_name = database_filename.replace(".db", "") + "_table"
    df.to_sql(table_name, engine, index=False, if_exists='replace')


def main():
  """
  arg: message.csv, category.csv, database filepath
  Three steps in the main()
  1.   load data by laod_data()
  2.   clean data
  3.   save dataframe to a database
  :return: a dataframe store in database
  """
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
            'disaster_response.db')


if __name__ == '__main__':
    main()
