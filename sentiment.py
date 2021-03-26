import urllib.request
import os
import pandas as pd
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

df = pd.read_csv('movie_data.csv', encoding='utf-8')
df.head()

# init Objects
tokenizer=RegexpTokenizer(r'\w+')
en_stopwords=set(stopwords.words('english'))
ps=PorterStemmer()
def getStemmedReview(review):
    review=review.lower()
    review=review.replace("<br /><br />"," ")
    #Tokenize
    tokens=tokenizer.tokenize(review)
    new_tokens=[token for token in tokens if token not in  en_stopwords]
    stemmed_tokens=[ps.stem(token) for token in new_tokens]
    clean_review=' '.join(stemmed_tokens)
    return clean_review

df['review'].apply(getStemmedReview)
X_train = df.loc[:35000, 'review'].values
y_train = df.loc[:35000, 'sentiment'].values
X_test = df.loc[35000:, 'review'].values
y_test = df.loc[35000:, 'sentiment'].values


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, encoding='utf-8', decode_error='ignore')
vectorizer.fit(X_train)
X_train=vectorizer.transform(X_train)
X_test=vectorizer.transform(X_test)

from sklearn.linear_model import LogisticRegression
model=LogisticRegression(solver='liblinear')
model.fit(X_train,y_train)
print("Score on training data is: "+str(model.score(X_train,y_train)))
print("Score on testing data is: "+str(model.score(X_test,y_test)))


"If you haven't seen the gong show TV series then you won't like this movie much at all, not that knowing the series makes this a great movie. <br /><br />I give it a 5 out of 10 because a few things make it kind of amusing that help make up for its obvious problems.<br />When we perform prediction on above point we can see below that our model is performing well enough."
# Here 0 denotes a negative sentiment
model.predict(X_test[0])


"If you haven't seen the gong show TV series then you won't like this movie much at all, not that knowing the series makes this a great movie. <br /><br />I give it a 5 out of 10 because a few things make it kind of amusing that help make up for its obvious problems.<br />When we perform prediction on above point we can see below that our model is performing well enough."
# 78% probability that the given text is negative
model.predict_proba(X_test[0])

import joblib
joblib.dump(en_stopwords, 'movieclassifier/pkl_objects/stopwords.pkl')
joblib.dump(model, 'movieclassifier/pkl_objects/model.pkl')
joblib.dump(vectorizer, 'movieclassifier/pkl_objects/vectorizer.pkl')