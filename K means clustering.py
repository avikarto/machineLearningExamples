# This code was written in the Atom IDE with Hydrogen as a Jupyter environment
# The # %% lines are cell breaks in Hydrogen

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

# K-means clustring is a method for unsupervised grouping of data into K localized sets
# Ideally K is chosen by an "elbow" method, basically finding the K value at which the total RMSE
#       from data points to group centroids, summed across all groups, stops decreasing rapidly
#       (the K at which the slope of total RMSE vs K is reduced dramatically)

# %%
df = pd.read_csv('income.csv')
df.head()
# %%
plt.scatter(df['Age'], df['Income($)'])
plt.show()

# There appears to be 3 clusters of data

# %%
km = KMeans(n_clusters=3)
km
# %%
yPredicted = km.fit_predict(df.drop(columns=['Name']))
yPredicted  # This will give the cluster label for each data point
# %%
df['cluster'] = yPredicted
df.head()
# %%
# I want to plot the clusters by color, so I'll add colors to the df corresponding to the cluster
colors = []
for i in range(len(df.cluster)):
    colors.append(['r', 'b', 'k'][df.cluster[i]])
df['color'] = colors
df.head()
# %%
plt.scatter(df['Age'], df['Income($)'], color=df.color)
plt.show()

# This isn't the way I'd have chosen the clusters...
#   Possibly an issue of feature scaling (x-range=~20, y-range=~120,000)
#   This scaling mismatch really screws with the RMSE calculation
# Let's rescale y and try again

# %%
df['Income(10k$)'] = df['Income($)']/10000
yPredicted = km.fit_predict(df.drop(columns=['Name', 'color', 'cluster', 'Income($)']))
df['cluster'] = yPredicted
colors = []
for i in range(len(df.cluster)):
    colors.append(['r', 'b', 'k'][df.cluster[i]])
df['color'] = colors
plt.scatter(df['Age'], df['Income(10k$)'], color=df.color)
plt.xlabel("Age")
plt.ylabel("Income (10k$)")
plt.show()

# Much better.

# %%
# Where are the cluster centers located, though?
km.cluster_centers_


# %%

# In general, we don't know K a priori.  Let's find the "elbow" point of K

kValues = range(1,11)
sse = []
rmse = []
for k in kValues:
    km = KMeans(n_clusters=k)
    km.fit(df[['Age', 'Income(10k$)']])
    sse.append(km.inertia_)  # this gives the sum of squares error
    rmse.append(np.sqrt(km.inertia_/k))  # this gives the RMS error

plt.plot(kValues, sse, color='b')
plt.plot(kValues, np.array(rmse)*30, color='r')  # scaled up for comparison
plt.show()

# the elbow is at K=3 for both cases, as expeced
