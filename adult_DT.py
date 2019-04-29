import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
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
print('accuracy = {} '.format( (preds == y_test).sum()/len(y_test)))