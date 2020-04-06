import numpy as np
import re
import pickle
import nltk
from nltk.corpus import stopwords
from sklearn.datasets import load_files
nltk.download('stopwords')

Reviews=load_files('txt_sentoken')

X,y= Reviews.data,Reviews.target