import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    ''' load and merge two datasets.
    Args:
    messages_filepath: string; filepath for csv file containing messages dataset.
    categories_filepath: string; filepath for csv file containing categories dataset.
       
    Returns:
    df: dataframe; a merged dataframe containing messages and categories datasets.
    '''

    # Load messages and categories datasets
    messages = pd.read_csv(messages_filepath) # 'messages.csv'
    categories = pd.read_csv(categories_filepath) # 'categories.csv'

    # Merge two datasets
    df = messages.merge(categories, on = 'id', how = 'inner')

    return df


def clean_data(df):
    ''' clean dataframe by 
    - converting categories from string to numeric values;
    - removing columns with unique value or removing records with non-binary values; and
    - removing duplicated records

    Args:
    df: dataframe; a merged dataframe from load_data() 
       
    Returns:
    df: dataframe; a cleaned dataframe containing messages and categories information.
    '''
    # Create a dataframe of the 36 individual category columns 
    categories = df.categories.str.split(pat = ';', expand = True)

    # Select the first row of the categories dataframe
    # and use this row to extract a list of new column names for categories.
    row = categories.iloc[0, :]
    category_colnames = row.apply(lambda x: x[:-2]).tolist()
    categories.columns = category_colnames

    # Convert string values to a numeric value
    for column in categories:
        # replace original value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1:])
        # convert dtype from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # Replace the original caategories column with the new categories dataframe
    df = pd.concat([df.drop('categories', axis = 1), categories], axis = 1)

    # Remove duplicates 
    df.drop_duplicates(inplace = True)

    # Drop columns if the column has only one (unique) value
    # and drop records if any of the column has non-binary values (vales other than 0 or 1)
    for col in df.columns[4:]: 
        # display(col)
        if len(pd.unique(df[col])) == 1:
            df.drop(col, axis = 1, inplace = True)
            print('    {} column with a unigue value is dropped'.format(col))
            continue
        
        if len(pd.unique(df[col])) != 2:
            target = pd.unique(df[col])[-1]
            df = df.loc[df[col] != target, :]
            print('    Records with value {} in the {} column are dropped'.format(target, col))


    return df


def save_data(df, database_filename):

    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('msg_categories', engine, index = False, if_exists = 'replace')
    pass  


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