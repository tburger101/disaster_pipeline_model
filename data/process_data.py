import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''This method loads both the messages and categories DF, merges them together, and breaks up the
    messages' cateogry into 36 different binary classes to be able to be used for a multioutput classifier
    '''
    
    #Loading the data sets
    categories = pd.read_csv(categories_filepath)
    categories.set_index('id', inplace=True)
    
    messages =  pd.read_csv(messages_filepath)
    messages.set_index('id', inplace=True)
    
    #Merging the Datasets
    df = pd.merge(messages, categories, left_index=True, right_index=True)
   
    return(df)

def clean_data(df):
    '''Changing the single category variable into 36 individual variables so
    a different model can be trained to predict each class'''
    
    #Splitting the category column into 36 different columns
    categories_split = df['categories'].str.split(";", expand=True)
    row = categories_split.iloc[0].values
    category_colnames = [x[:-2] for x in row]
    categories_split.columns = category_colnames
    
    #Changing the value of the categories to be binary
    for column in categories_split:
    # set each value to be the last character of the string
        categories_split[column] =categories_split[column].apply(lambda x : x[-1])    
    # convert column from string to numeric
        categories_split[column] =categories_split[column].apply(lambda x : int(x))
    

    #Merging the two dataframes together and drop duplicate column
    df.drop('categories', axis=1, inplace=True)
    df = pd.merge(df, categories_split, left_index=True, right_index=True)
    
    #Removing duplicates
    df.drop_duplicates(inplace=True)
    return(df)

def save_data(df, database_filename):
    engine_text="sqlite:///"+str(database_filename)
    engine = create_engine(engine_text)
    df.to_sql('messages', engine, index=False)  


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