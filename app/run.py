import json
import plotly
import pandas as pd

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine

app = Flask(__name__)


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


# load data
engine = create_engine('sqlite:///disaster.db')
df = pd.read_sql_table('disastertable', engine)
# print(df)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
  # extract data needed for visuals
  # TODO: Below is an example - modify to extract data for your own visuals
  genre_counts = df.groupby('genre').count()['message']
  genre_names = list(genre_counts.index)

  category_names = df.columns[4:]
  category_counts = df.iloc[:,4:].sum().values

  # create visuals
  # TODO: Below is an example - modify to create your own visuals
  graphs = [
    # graph 1: genre category bar chart
    {
      'data': [
        Bar(
          x=genre_names,
          y=genre_counts
        )
      ],

      'layout': {
        'title': 'Distribution of Message Genres',
        'yaxis': {
          'title': "Count"
        },
        'xaxis': {
          'title': "Genre"
        }
      }
    },
    # graph 2: category bar chart
    {
      'data': [
        Bar(
          x=category_names,
          y=category_counts
        )
      ],

      'layout': {
        'title': 'Distribution of Message categories',
        'yaxis': {
          'title': "Count"
        },
        'xaxis': {
          'title': "Categories"
        }
      }
    }
  ]

  # encode plotly graphs in JSON
  ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
  graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

  # render web page with plotly graphs
  return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
  # save user input in query
  query = request.args.get('query', '')

  # use model to predict classification for query
  classification_labels = model.predict([query])[0]
  classification_results = dict(zip(df.columns[4:], classification_labels))

  # This will render the go.html Please see that file.
  return render_template(
    'go.html',
    query=query,
    classification_result=classification_results
  )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
  main()
