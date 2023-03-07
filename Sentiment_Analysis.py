!pip install nltk

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
import re

nltk.download('punkt')
nltk.download('stopwords')

# Load data
with open('train.csv', 'r') as f:
    data = f.readlines()

# Split data into labels and paragraphs
labels = [d[:2] for d in data]
paragraphs = [d[3:] for d in data]

# Tokenize, remove stopwords and stem paragraphs
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()
tokens = []
for para in paragraphs:
    tokens.append([ps.stem(token) for token in nltk.word_tokenize(para) if token not in stop_words])

# Remove unnecessary symbols and blank entries
strs_after_clean = []
points = []
for i, rev in enumerate(tokens):
    rev = " ".join(rev)
    rev = re.sub('[^A-Za-z]', ' ',rev).lower()
    if rev:
        strs_after_clean.append(rev)
        points.append(labels[i])

# Create dataframe and train model
df = pd.DataFrame({'review': strs_after_clean, 'points': points})
model = CountVectorizer(max_features=1000)
X = model.fit_transform(df['review'])
clf = MultinomialNB().fit(X, df['points'])

# Make predictions on validation data
with open('valid.csv', 'r') as f:
    dataValid = f.readlines()
pointsValid = [d[:2] for d in dataValid]
paragraphsValid = [d[3:] for d in dataValid]
tokensValid = []
for para in paragraphsValid:
    tokensValid.append([ps.stem(token) for token in nltk.word_tokenize(para) if token not in stop_words])
strs_after_clean_Valid = []
for rev in tokensValid:
    rev = " ".join(rev)
    rev = re.sub('[^A-Za-z]', ' ',rev).lower()
    if rev:
        strs_after_clean_Valid.append(rev)
dfValid = pd.DataFrame({'review': strs_after_clean_Valid, 'points': pointsValid})
XValid = model.transform(dfValid['review'])
y_pred_valid = clf.predict(XValid)
score_valid = clf.score(XValid, dfValid['points'])

# Write predictions to file
with open("pred_test_post_valid.csv", 'w') as f:
    for label in y_pred_valid:
        f.write(str(label) + "\n")
