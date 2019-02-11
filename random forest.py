# This code was written in the Atom IDE with Hydrogen as a Jupyter environment
# The # %% lines are cell breaks in Hydrogen

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection as sklms
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
%matplotlib inline

# Random forest builds multiple decision trees based on subsets of training data
# This will use the digits data set from sklearn and train a random forest model
# to identify written digits

# %%
digits = load_digits()
dir(digits)
# %%
plt.matshow(digits.images[17])
plt.show()
# %%
df = pd.DataFrame(digits.data)
df['target'] = digits.target
df.head()

# each data element is a list of 64 values, which map to cell values in the image.

# %%
# Make training and testing sets
xTrain, xTest, yTrain, yTest = sklms.train_test_split(df.drop(columns=['target']), df.target, test_size=0.2, random_state=12345)
# %%
# train the random forest
RF_model = RandomForestClassifier()
RF_model.fit(xTrain, yTrain)

# %%

# the value n_estimators is how many random trees are generated.  How does changing this change the results?

estimators = []
scores = []
for nEst in range(10,120):
    RF_model = RandomForestClassifier(n_estimators=nEst)
    RF_model.fit(xTrain, yTrain)
    #print('For n_estimators='+str(nEst)+', score=',str(RF_model.score(xTest, yTest)))
    estimators.append(nEst)
    scores.append(RF_model.score(xTest, yTest))

plt.scatter(estimators, scores)
plt.xlabel('n_estimators')
plt.ylabel('RF_model.score')
plt.show()

# The scoring doesn't seem to converge at large n_estimators

# %%

# Where does the model fail?

yPredicted = RF_model.predict(xTest)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(yTest, yPredicted)
cm # off-diagonal are incorrect predictions
