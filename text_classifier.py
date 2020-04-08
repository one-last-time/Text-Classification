import numpy as np
import pickle


#unpickling pretrained tfiidf  and classifier

with open('classifier.pickle','rb') as f:
    clf=pickle.load(f)

with open('tfidfmodel.pickle','rb') as f:
    tfidf=pickle.load(f)



sample=['it was great show.,have a good day and wash your hands']
sample2=['The network is bad']

sample=tfidf.transform(sample2).toarray()

print(clf.predict(sample))


