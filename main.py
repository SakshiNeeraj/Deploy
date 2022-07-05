import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
import pickle

df_fake=pd.read_csv("Fake.csv")
df_true=pd.read_csv("True.csv")

df_fake["class"]=0
df_true["class"]=1

df_fake_manual_testing=df_fake.tail(10)
df_fake.drop([23470,23480],axis=0,inplace=True)
df_true_manual_testing=df_true.tail(10)
df_true.drop([21406,21416],axis=0,inplace=True)
df_manual_testing=pd.concat([df_fake_manual_testing,df_true_manual_testing],axis=0)
df_manual_testing.to_csv("manual_testing.csv")

df_merge=pd.concat([df_fake,df_true],axis=0)

df=df_merge.drop(["subject","date"],axis=1)

df = df.sample(frac = 1)

def conversion(title):
 title = title.lower()
 title = re.sub('\[.*?\]', '', title)
 title = re.sub("\\W"," ",title)
 title = re.sub('https?://\S+|www\.\S+', '', title)
 title = re.sub('<.*?>+', '', title)
 title = re.sub('[%s]' % re.escape(string.punctuation), '', title)
 title = re.sub('\n', '', title)
 title = re.sub('\w*\d\w*', '', title)
 return title

df["title"] = df["title"].apply(conversion)

x = df.iloc[0:5000, 0]
y = df.iloc[0:5000, -1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

pickle.dump(vectorization, open('transform.pkl','wb'))

knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(xv_train, y_train)

LR = LogisticRegression()
LR.fit(xv_train, y_train)

svc = SVC()
svc.fit(xv_train, y_train)

models = list()

logistic_regression = Pipeline([('m', LogisticRegression())])
models.append(('logistic', logistic_regression))

svc = Pipeline([('m', SVC())])
models.append(('svc', svc))

k_n_n = Pipeline([('m', KNeighborsClassifier(n_neighbors=3))])
models.append(('knn', k_n_n))

ensemble = VotingClassifier(estimators=models, voting='hard')
ensemble.fit(xv_train, y_train)

filename='nlp_model.pkl'
pickle.dump(ensemble, open(filename, 'wb'))
