import sys
import pandas as pd
from sqlalchemy import create_engine
import sqlite3
import numpy as np
import re
import pickle

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('wordnet')

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


def load_data(database_filepath):
    
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    print('Table name: ', engine.table_names())
    df = pd.read_sql_table(engine.table_names()[0], engine)
    
    X = df['message'].values
    
    category_names = ['related', 'request',
       'offer', 'aid_related', 'medical_help', 'medical_products',
       'search_and_rescue', 'security', 'military', 'child_alone', 'water',
       'food', 'shelter', 'clothing', 'money', 'missing_people', 'refugees',
       'death', 'other_aid', 'infrastructure_related', 'transport',
       'buildings', 'electricity', 'tools', 'hospitals', 'shops',
       'aid_centers', 'other_infrastructure', 'weather_related', 'floods',
       'storm', 'fire', 'earthquake', 'cold', 'other_weather',
       'direct_report']
        
    Y = df[category_names].values
    

    
    return X, Y, category_names

def tokenize(text):
    text = text.lower()
    text = re.sub(r'\d+?', '', text)
    text = re.sub(r'[^\w]', ' ', text)
    text = re.sub(r'\ +', ' ', text)
    text = word_tokenize(text)
    text = [WordNetLemmatizer().lemmatize(word, pos='v') for word in text]
    return text

def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'tfidf__norm': ('l1', 'l2'),
        'clf__estimator__n_estimators': [5, 10, 20]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    
    y_pred = model.predict(X_test).reshape(-1, 1)

    print("Accuracy: ", accuracy_score(Y_test.reshape(-1, 1), y_pred))
    print("Precision: ", precision_score(Y_test.reshape(-1, 1), y_pred.reshape(-1, 1), average='macro'))
    print("Recal: ", recall_score(Y_test.reshape(-1, 1), y_pred.reshape(-1, 1), average='macro'))
    print("F1 Score: ", f1_score(Y_test.reshape(-1, 1), y_pred.reshape(-1, 1), average='macro'))
    
    try:
        print("\nBest Parameters:", model.best_params_)
    except:
        print("no calculation of best parameters")


def save_model(model, model_filepath):
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))


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