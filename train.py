import numpy as np
import re
import pickle
import nltk
from nltk.corpus import stopwords
from sklearn.datasets import data
from sklearn.datasets import load_files
nltk.download('stopwords')


with open('X.pickle','rb') as f:
    X = pickle.load(f)

with open('y.pickle','rb') as f:
    y = pickle.load(f)

#preprocessing the data

corpus=[]

for i in range(len(X)):
    review = re.sub(r'\W', ' ', str(X[i]))
    review = review.lower()
    review = re.sub(r'\s+[a-z]+\s', ' ', review)
    review = re.sub(r'^[a-z]\s+', ' ', review)
    review = re.sub(r'\s+', ' ', review)
    corpus.append(review)


#creating tf-idf model

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer= TfidfVectorizer(max_features=50000, min_df=2, max_df=0.7, stop_words=stopwords.words('english'))
X=vectorizer.fit_transform(corpus)


#spilt test and train data
from sklearn.model_selection import train_test_split
text_train,text_test,sent_train,sent_test = train_test_split(X, y, test_size=0.1, random_state=0)



#training the model

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()

classifier.fit(text_train,sent_train)

#testing the accurancy
sent_pred = classifier.predict(text_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(sent_test, sent_pred)
print(cm)
accurancy = (cm[0,0]+cm[1,1])/len(sent_test)
print('accurancy is ', (accurancy*100), '%')

#pickling the classifier

with open('classifier.pickle','wb') as f:
    pickle.dump(classifier,f)

#pickling the tfidf vectorizer
with open('tfidfmodel.pickle','wb') as f:
    pickle.dump(vectorizer,f)