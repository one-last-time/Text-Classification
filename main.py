import numpy as np
import re
import pickle
import nltk
from nltk.corpus import stopwords
from sklearn.datasets import data
from sklearn.datasets import load_files
nltk.download('stopwords')


infile= open('X.pickle','rb')
X1=pickle.load(infile)
infile.close()

infile= open('y.pickle','rb')
y1=pickle.load(infile)
infile.close()