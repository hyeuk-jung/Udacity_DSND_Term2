import sys
import nltk
import numpy as np
import pickle
import pandas as pd
import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize

from sklearn.base import BaseEstimator, TransformerMixin # For custom transformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion

from sqlalchemy import create_engine

nltk.download(['punkt', 'wordnet', 'stopwords'])

def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('msg_categories', con = engine)

    X = df.message
    Y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)
    category_names = Y.columns

    return X, Y, category_names


def tokenize(text):
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize and remove stop words
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens


def build_model():
    # build a ml pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(), n_jobs = -1))
    ])

    # set a dictionary of parameters for hyperparameter tuning
    parameters = {'clf__estimator__max_depth': [2, 3, 5], \
                  'clf__estimator__min_samples_split': [2, 3, 5], \
                  'clf__estimator__n_estimators': [50, 100]}

    # create a model
    model = GridSearchCV(pipeline, param_grid = parameters, cv = 3, \
                         refit = True, verbose = 5) # , scoring = 'f1'
    
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    # predict the result
    Y_pred = model.predict(X_test)
    Y_pred = pd.DataFrame(Y_pred, columns = category_names)

    # result = []
    # for column in category_names:
    #     accuracy = accuracy_score(Y_test[column], Y_pred[column])
    #     precision = precision_score(Y_test[column], Y_pred[column], average = 'micro') 
    #     recall = recall_score(Y_test[column], Y_pred[column], average = 'micro')
    #     f1 = f1_score(Y_test[column], Y_pred[column], average = 'micro')
        
    #     result.append([accuracy, precision, recall, f1])
        
    # result = pd.DataFrame(result, index = category_names, \
    #                       columns = ['Accuracy', 'Precision', 'Recall', 'F1'])
    # print(result)          
            
    print(classification_report(Y_test.values, Y_pred.values, target_names = category_names), \
          '\nAccuracy: {}'.format(accuracy_score(Y_test.values, Y_pred.values)) )


def save_model(model, model_filepath):
    # filename = 'ML_pipeline_tunedmodel.pkl'
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