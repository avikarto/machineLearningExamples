# This code was written in the Atom IDE with Hydrogen as a Jupyter environment
# The # %% lines are cell breaks in Hydrogen

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as skll
import sklearn.model_selection as sklms
%matplotlib inline

# predicting if a person will buy an insurance policy based on their age
# this task is suited for binary logistic regression

# %%
insuranceData = {
    'age': [22, 25, 47, 52, 46, 56, 55, 60, 62, 61, 18, 28, 27, 29, 49, 55, 25, 58, 19, 18, 21, 26, 40, 45, 50, 54, 23],
    'bought policy?': [0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0]
}

df = pd.DataFrame.from_dict(insuranceData)
df.head()

# %%
# Let's visualize this data
plt.scatter(df['age'], df['bought policy?'])
plt.ylabel('bought policy? (bool)')
plt.xlabel('age')
plt.show()

# Older people seem more likely to buy insurance

# %%
# generate training and test sets, and train the model
xTrain, xTest, yTrain, yTest = sklms.train_test_split(df[['age']], df['bought policy?'], test_size=0.1, random_state=12345)
logModel = skll.LogisticRegression()
logModel.fit(xTrain, yTrain)
# %%
# test our predictions
print(xTest.values, '\n')
print(logModel.predict(xTest))
# %%
logModel.score(xTest, yTest)

# %%
# the model isn't truly perfect, the set is just very small.  We can see the actual probability by age:

xs = np.arange(18, 80)
ys = [logModel.predict_proba([[xs[i]]])[0, 1] for i in range(0, 62)]
plt.plot(xs, ys)
plt.ylabel('probability')
plt.xlabel('age')
plt.show()
