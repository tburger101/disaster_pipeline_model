import sys, pickle, re,  nltk
nltk.download(['punkt', 'wordnet'])
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier

def load_data(database_filepath):
    '''Loading data from the message table in the database'''
    
    from sqlalchemy import create_engine
    engine = create_engine("sqlite:///"+str(database_filepath))
    df = pd.read_sql('SELECT * FROM messages', engine)
    X = df['message'].values
    Y = df.drop(['message', 'original', 'genre'], axis=1).values
    category_names=df.drop(['message', 'original', 'genre'], axis=1).columns
    return(X,Y, category_names)

def tokenize(text):
    '''Breaking the messages into a list of single words to be used in
    nlp pipeline'''
    
    
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize andremove stop words
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words("english")]

    return tokens


def build_model():
    '''Creating a pipeline and grid search model which can be optimized'''
    
    #Bulding the pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_jobs=-1)))])
    
    #Giving the model parameters to tune.
    parameters = {'tfidf__use_idf': (True, False),
        'vect__max_features': (None, 5000),
        'clf__estimator__min_samples_split': [2, 3, 4],
        'clf__estimator__class_weight': ['balanced']
                 }

    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1)
    return(cv)

def evaluate_model(model, X_test, Y_test, category_names):
    '''prints out the classification report for each of the
    36 differnet categories'''
    
    predictions = model.predict(X_test)
    predictions_df=pd.DataFrame(predictions, columns = category_names)
    y_test_df=pd.DataFrame(Y_test, columns = category_names)
    
    for column in predictions_df.columns:
        print(column)
        print(classification_report(y_test_df[column], predictions_df[column]))


def save_model(model, model_filepath):
    '''pickle the model to be used later'''
    
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()