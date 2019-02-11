# This code was written in the Atom IDE with Hydrogen as a Jupyter environment
# The # %% lines are cell breaks in Hydrogen

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import sklearn.model_selection as sklms
from sklearn.datasets import load_digits
%matplotlib inline

# Using cross-validation to evaluate a model on different subsets of the same data

# %%

# Training and testing a logistic regression model on the digits dataset

digits = load_digits()
xTrain, xTest, yTrain, yTest = sklms.train_test_split(digits.data, digits.target, test_size=0.3)

logModel = LogisticRegression()
logModel.fit(xTrain, yTrain)
logModel.score(xTest, yTest)
# %%

# Training and testing a SVM on the digits dataset

SVM_model = SVC()
SVM_model.fit(xTrain, yTrain)
SVM_model.score(xTest, yTest)

# %%

# Training and testing a random forest model on the digits dataset

RF_model = RandomForestClassifier()
RF_model.fit(xTrain, yTrain)
RF_model.score(xTest, yTest)

# %%

# Depending on the train_test_split seed, these models will perform very differently.  K-fold is another way:

kf = sklms.KFold(n_splits=3)
kf

for iTrain, iTest in kf.split(range(1, 10)):
    print(iTrain, iTest)

# the model parameter n_splits=3 is visualized here as 3 rows of indices.  Theses are the number of 'folds' of the data

# %%


def getScore(model, xTrain, xTest, yTrain, yTest):
    model.fit(xTrain, yTrain)
    return model.score(xTest, yTest)


# %%
skf = sklms.StratifiedKFold(n_splits=3)  # a little better than KFold, since it splits the data uniformly by category
scores_log = []
scores_RF = []
scores_SVM = []

for iTrain, iTest in kf.split(digits.data):
    xTrain, xTest, yTrain, yTest = digits.data[iTrain], digits.data[iTest], digits.target[iTrain], digits.target[iTest]
    scores_SVM.append(getScore(SVC(), xTrain, xTest, yTrain, yTest))
    scores_log.append(getScore(LogisticRegression(), xTrain, xTest, yTrain, yTest))
    scores_RF.append(getScore(RandomForestClassifier(n_estimators=40), xTrain, xTest, yTrain, yTest))
# %%
print("Logistic Regression scores: ", scores_log)
print("Average: ", np.average(scores_log), '\n')
print("Random Forest (40) scores: ", scores_RF)
print("Average: ", np.average(scores_RF), '\n')
print("SVC scores: ", scores_SVM)
print("Average: ", np.average(scores_SVM))
# %%

# This whole process of creating folds and testing is automated by cross_val_score:

sklms.cross_val_score(LogisticRegression(), digits.data, digits.target)
# %%
sklms.cross_val_score(SVC(), digits.data, digits.target)
# %%
sklms.cross_val_score(RandomForestClassifier(n_estimators=40), digits.data, digits.target)

# %%

# Alternatively, one can test the same method with different parameters:

temp = sklms.cross_val_score(RandomForestClassifier(n_estimators=40), digits.data, digits.target)
print(temp)
print(np.average(temp))
# %%
temp = sklms.cross_val_score(RandomForestClassifier(n_estimators=50), digits.data, digits.target)
print(temp)
print(np.average(temp))
# %%
temp = sklms.cross_val_score(RandomForestClassifier(n_estimators=60), digits.data, digits.target)
print(temp)
print(np.average(temp))
