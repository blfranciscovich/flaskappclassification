from flask import Flask, render_template, request
from flask_wtf import FlaskForm
from wtforms import Form, SubmitField
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import re
from nltk.stem.porter import PorterStemmer
import nltk
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from stats import groupairline
import pickle
from sklearn.model_selection import train_test_split

app = Flask(__name__)


def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',
                           text)
    text = (re.sub('[\W]+', ' ', text.lower()) +
            ' '.join(emoticons).replace('-', ''))
    return text

porter = PorterStemmer()
nltk.download('stopwords')

def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

def tokenizer(text):
    return text.split()

def unpickle():
	mypickle_path = 'fb_pickle.pkl'
	model_unpickle = open(mypickle_path, 'rb')
	clf_new = pickle.load(model_unpickle)
	return clf_new

class fbForm(Form):
	submit = SubmitField("Send")
@app.route("/")
def main():
	form = fbForm(request.form)
	return render_template('main.html', form=form)

@app.route('/analysis', methods=['POST'])
def result():
    form = fbForm(request.form)
    if request.method == 'POST' and form.validate():
        message = request.form['sentimentTextarea']
        myname = request.form['nameText']
        review_this = unpickle()
        category = review_this.predict([message])

        return render_template('analysis.html', content=message, category=category[0], myname=myname)
    return render_template('main.html', form=form)

if __name__ == "__main__":
    app.run(debug=True)
