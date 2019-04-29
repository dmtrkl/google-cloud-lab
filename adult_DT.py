import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import os
from google.cloud import storage
import time

start = time.time()

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "Dmytro-Kolesnykov-Week10-66d0123e7f6b.json" 
client = storage.Client()
bucket = client.get_bucket('dmytro-kolesnykov-week10')

np.random.seed(0)
data = pd.read_csv("adult.data")
values = set([])
for x in data.income.values:
    values.add(x)
values = list(values)
onehot = pd.get_dummies(data.assign(bool_income = data.income.apply(lambda x: values.index(x))).drop(['income'], axis=1))

X_train, X_test, y_train, y_test = train_test_split(
    onehot.drop(['bool_income'], axis=1), onehot['bool_income'], test_size=0.2, random_state=42)
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
accuracy = (preds == y_test).sum()/len(y_test)

end = time.time() - start

print('accuracy : {} '.format(accuracy))
print('time taken : {} '.format(end))

blob = storage.Blob("adult-script.data", bucket)
blob.upload_from_string('accuracy : {} '.format(accuracy) + '\n' + time taken : {} '.format(end)')


timeReg1 = 0
timeReg2 = 0
timeReg3 = 0
for i in range(5):
    timeS1 = time.time()
    data = pd.read_csv("https://storage.googleapis.com/eubucket-week10/adult.data") #From Bucket in EU Region
    timeE1 = time.time()

    timeS2 = time.time()
    data = pd.read_csv("https://storage.googleapis.com/testbucket-week10/adult.data") #From Bucket in Japan Region
    timeE2 = time.time()

    timeS3 = time.time()
    data = pd.read_csv("https://storage.googleapis.com/dmytro-kolesnykov-week10/adult.data") #From Bucket in US Region
    timeE3 = time.time()
    
    timeReg1 += timeE1 - timeS1
    timeReg2 += timeE2 - timeS2
    timeReg3 += timeE3 - timeS3
    
print("TimeEU:", timeReg1 / 5, " --- TimeJ:", timeReg2 / 5 , " --- TimeUS:", timeReg3 / 5 )
