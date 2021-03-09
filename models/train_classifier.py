# import libraries
import os
import sys
import pandas as pd
from sqlalchemy import create_engine
import pickle

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import nltk

nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report


class VerbCount(BaseEstimator, TransformerMixin):

  def verb_count(self, text):
    verb_count = 0
    sentence_list = nltk.sent_tokenize(text)
    for sentence in sentence_list:
      pos_tags = nltk.pos_tag(tokenize(sentence))

      for word_tag in pos_tags:
        word, tag = word_tag
        if tag in ['VB', 'VBP'] or word == 'RT':
          verb_count += 1
    return verb_count

  def fit(self, x, y=None):
    return self

  def transform(self, X):
    X_tagged = pd.Series(X).apply(self.verb_count)
    return pd.DataFrame(X_tagged)


def load_data(database_filepath):
  """
    load data from the database
    :param database_filepath: the file path for the stored database
    :return: X the text message, y the corresponding labels, category_name records all label names
  """
  engine = create_engine('sqlite:///' + database_filepath)
  table_name = os.path.basename(database_filepath).replace(".db", "") + "_table"
  df = pd.read_sql_table(table_name, engine)

  # drop unnecessary variables
  df = df.drop(['id', 'original'], axis=1)

  # make relate a binary variable
  df['related'] = df['related'].map(lambda x: 1 if x == 2 else x)

  # create labels
  y = df.drop(['message', 'genre'], axis=1)
  # store features (text)
  X = df['message']
  # get category names
  category_names = y.columns

  return X, y, category_names


def tokenize(text):
  """the customized transform function from train_classifier.py.
    The customized transform function needs to be imported to this file where the classifier.pkl is used
    """
  tokens = word_tokenize(text)
  lemmatizer = WordNetLemmatizer()

  clean_tokens = []
  for tok in tokens:
    clean_tok = lemmatizer.lemmatize(tok).lower().strip()
    clean_tokens.append(clean_tok)

  return clean_tokens


def build_model():
  """
  A model contains pipelines, hyperparameter search, and a ML model to fit.

  Pipeline containing two parallel pipelines (by using featureUnion), the text_pipeline transform text by CountVector and tfidf
  the start_verb is the function running parallel to the text pipeline that code the number of verb each message has.

  Parameters contains the predefined hyperparameters to search in CV

  The ML is decision trees with adaboost
  :return: a trained ML model that can be used to make predictions on unseen test data.
  """
  pipeline = Pipeline([
    ('features', FeatureUnion([

      ('text_pipeline', Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer())
      ])),

      ('starting_verb', VerbCount())
    ])),

    ('clf', MultiOutputClassifier(AdaBoostClassifier()))
  ])

  parameters = {
    'clf__estimator__learning_rate': [0.01, 0.02, 0.05],
    'clf__estimator__n_estimators': [10, 20, 40]
  }

  cv = GridSearchCV(pipeline, param_grid=parameters)

  return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
  """

  :param model: the trained ML model
  :param X_test: test message data
  :param Y_test: test message labels
  :param category_names: labels' names
  :return: classification report of the trained ML model on the test data, which output precision, f1 scores etc.
  """
  Y_pred = model.predict(X_test)
  print(classification_report(Y_test.values, Y_pred, target_names=category_names))


def save_model(model, model_filepath):
  """this is pickle method to store trained ML model to a given file path"""
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
    print('Please provide the filepath of the disaster messages database ' \
          'as the first argument and the filepath of the pickle file to ' \
          'save the model to as the second argument. \n\nExample: python ' \
          'train_classifier.py ../data/disaster_response.db classifier.pkl')


if __name__ == '__main__':
  main()
