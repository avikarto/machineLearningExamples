# This code was written in the Atom IDE with Hydrogen as a Jupyter environment
# The # %% lines are cell breaks in Hydrogen

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import sklearn.model_selection as sklms
from sklearn.svm import SVC
%matplotlib inline

# SVM classification of sklearn iris data, to determine species of iris based on petal/sepal properties

# %%
# load iris data
iris = load_iris()
dir(iris)
# %%
iris.feature_names
# %%

# make a data frame of the iris data

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df.head(1)
# %%

# What are the labels for the iris types?

print(iris.target_names)
print(iris.target)
# %%

# Let's add these labels to the DF

df['target'] = iris.target
df['type'] = df.target.apply(lambda x: iris.target_names[x])
df.head(1)

# %%

# How does the data look?  How clear is the distinction betwwen types, in the data?
type0 = df[df.target == 0]
type1 = df[df.target == 1]
type2 = df[df.target == 2]
plt.scatter(type0['sepal length (cm)'], type0['sepal width (cm)'], color='blue')
plt.scatter(type1['sepal length (cm)'], type1['sepal width (cm)'], color='red')
plt.scatter(type2['sepal length (cm)'], type2['sepal width (cm)'], color='black')
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.show()

# %%
plt.scatter(type0['petal length (cm)'], type0['petal width (cm)'], color='blue')
plt.scatter(type1['petal length (cm)'], type1['petal width (cm)'], color='red')
plt.scatter(type2['petal length (cm)'], type2['petal width (cm)'], color='black')
plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')
plt.show()

# In both cases, blue (setosa) seems the easiest to classify.  The other two may be easier to classify based on petal
# properties than sepal properties.  Of course, this will all happen simultaneously in the model.

# %%

# Training the model
x = df.drop(columns=['target', 'type'])
y = df.target
xTrain, xTest, yTrain, yTest = sklms.train_test_split(x, y, test_size=0.2, random_state=12345)

SVM_model = SVC()
SVM_model.fit(xTrain, yTrain)

# %%
SVM_model.score(xTest, yTest)
# %%

# Let's say we find an iris with the following properties:
# sepal length = 10, sepal width = 6, petal length = 5, petal width = 2
# What type is it?

iris.target_names[SVM_model.predict([[10, 6, 5, 2]])[0]]
