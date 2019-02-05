# This code was written in the Atom IDE with Hydrogen as a Jupyter environment
# The # %% lines are cell breaks in Hydrogen

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as skll
import sklearn.model_selection as sklms
from sklearn.datasets import load_digits
%matplotlib inline

# determining the value of a hand-written numerical digit
# since the values are a set of only 10 potential elements, multiclass classification is ideal
# sklearn has a built in dataset of such digits, which is imported above.

# %%
# load the training set:
digits = load_digits()
dir(digits)

# %%
# what does 'data' look like?
digits.data[0]
# %%
# There are also images...
plt.gray()
plt.matshow(digits.images[0])
plt.show()

# maybe a zero?

# %%
# and 'target' gives the actual values of the digits
digits.target[0:5]

# %%

# let's make training and testing sets
xTrain, xTest, yTrain, yTest = sklms.train_test_split(digits.data, digits.target, test_size=0.2, random_state=12345)

# make the linear model
logModel = skll.LogisticRegression()
logModel.fit(xTrain, yTrain)

# %%
logModel.score(xTest, yTest)
# the model looks pretty good.  Let's test it

# %%
testValue = 128
target = str(digits.target[testValue])
prediction = str(logModel.predict([digits.data[testValue]])[0])
plt.matshow(digits.images[testValue])
plt.show()
print('For a target of '+target+' the model predicted '+prediction)

# %%

# the model failed somewhere given the ~97% score...where did it fail?
yPredicted = logModel.predict(xTest)
from sklearn.metrics import confusion_matrix
confusion_matrix(yTest, yPredicted)

# diag elements are correct predictions, eg (0,0) is the model predicting 0 for a value 0
# off-diag elements are counts of incorrect predictions
