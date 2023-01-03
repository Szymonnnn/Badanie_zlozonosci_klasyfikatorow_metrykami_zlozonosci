import numpy as np
from rse import RandomSubspaceEnsemble
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
import time

dataset = np.genfromtxt("datasets/ionosphere_2.csv", delimiter=";")

X = dataset[:, :-1]
y = dataset[:, -1].astype(int)

X[0][0]=0

print("Total number of features", X.shape[1])

n_splits = 5
n_repeats = 10

rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1234)

# Eksperyment badający różnicę między głosowaniem miękkim i twardym klasyfikatorów

clf = RandomSubspaceEnsemble(base_estimator=GaussianNB(), n_estimators=10, n_subspace_features=5, hard_voting=True, random_state=123)
scores = []
for train, test in rskf.split(X, y):
    clf.fit(X[train], y[train])
    y_pred = clf.predict(X[test])
    scores.append(accuracy_score(y[test], y_pred))
print("Hard voting - accuracy score: %.3f (%.3f)" % (np.mean(scores), np.std(scores)))

clf = RandomSubspaceEnsemble(base_estimator=GaussianNB(), n_estimators=10, n_subspace_features=5, hard_voting=False, random_state=123)
scores = []
for train, test in rskf.split(X, y):
    clf.fit(X[train], y[train])
    y_pred = clf.predict(X[test])
    scores.append(accuracy_score(y[test], y_pred))
print("Soft voting - accuracy score: %.3f (%.3f)" % (np.mean(scores), np.std(scores)))

# Eksperyment wpływ liczby cech urzytej do stworzenia jednej podprzestrzeni
print()

clf = RandomSubspaceEnsemble(base_estimator=GaussianNB(), n_estimators=10, n_subspace_features=10, hard_voting=False, random_state=123)
scores = []
for train, test in rskf.split(X, y):
    clf.fit(X[train], y[train])
    y_pred = clf.predict(X[test])
    scores.append(accuracy_score(y[test], y_pred))
print("10/34 features - accuracy score: %.3f (%.3f)" % (np.mean(scores), np.std(scores)))

clf = RandomSubspaceEnsemble(base_estimator=GaussianNB(), n_estimators=10, n_subspace_features=15, hard_voting=False, random_state=123)
scores = []
for train, test in rskf.split(X, y):
    clf.fit(X[train], y[train])
    y_pred = clf.predict(X[test])
    scores.append(accuracy_score(y[test], y_pred))
print("15/34 features - accuracy score: %.3f (%.3f)" % (np.mean(scores), np.std(scores)))

clf = RandomSubspaceEnsemble(base_estimator=GaussianNB(), n_estimators=10, n_subspace_features=20, hard_voting=False, random_state=123)
scores = []
for train, test in rskf.split(X, y):
    clf.fit(X[train], y[train])
    y_pred = clf.predict(X[test])
    scores.append(accuracy_score(y[test], y_pred))
print("20/34 features - accuracy score: %.3f (%.3f)" % (np.mean(scores), np.std(scores)))

# Eksperyment wpływ doboru liczby klasyfikatorów
print()

clf = RandomSubspaceEnsemble(base_estimator=GaussianNB(), n_estimators=5, n_subspace_features=15, hard_voting=False, random_state=123)
scores = []
for train, test in rskf.split(X, y):
    clf.fit(X[train], y[train])
    y_pred = clf.predict(X[test])
    scores.append(accuracy_score(y[test], y_pred))
print("5 estimators - accuracy score: %.3f (%.3f)" % (np.mean(scores), np.std(scores)))

clf = RandomSubspaceEnsemble(base_estimator=GaussianNB(), n_estimators=10, n_subspace_features=15, hard_voting=False, random_state=123)
scores = []
for train, test in rskf.split(X, y):
    clf.fit(X[train], y[train])
    y_pred = clf.predict(X[test])
    scores.append(accuracy_score(y[test], y_pred))
print("10 estimators - accuracy score: %.3f (%.3f)" % (np.mean(scores), np.std(scores)))

clf = RandomSubspaceEnsemble(base_estimator=GaussianNB(), n_estimators=15, n_subspace_features=15, hard_voting=False, random_state=123)
scores = []
for train, test in rskf.split(X, y):
    clf.fit(X[train], y[train])
    y_pred = clf.predict(X[test])
    scores.append(accuracy_score(y[test], y_pred))
print("15 estimators - accuracy score: %.3f (%.3f)" % (np.mean(scores), np.std(scores)))

import problexity as px

cc = px.ComplexityCalculator()
cc.fit(X, y)

print(cc.complexity)
print(cc._metrics())
print(cc.score())

print(cc.report())

# Import matplotlib
import matplotlib.pyplot as plt

# Prepare figure
fig = plt.figure(figsize=(7,7))

# Generate plot describing the dataset
cc.plot(fig, (1,1,1))
fig.savefig('xd.png')

import pandas
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier


names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",names=names)

array = dataframe.values
x = array[:,0:8]
y = array[:,8]
max_features = 3

kfold = model_selection.KFold(n_splits=10)
rf = DecisionTreeClassifier(max_features=max_features)
num_trees = 100

model = BaggingClassifier(base_estimator=rf, n_estimators=num_trees, random_state=2020)
results = model_selection.cross_val_score(model, x, y, cv=kfold)
print("Accuracy: %0.2f (+/- %0.2f)" % (results.mean(), results.std()))
