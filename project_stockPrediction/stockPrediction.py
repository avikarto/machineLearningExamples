# Analysis of stock data.  ML is used to predict the closing price of Google stock
# based on the actions of Apple, Amazon, and Microsoft stocks for a given day.

# This code was written in the Atom IDE with Hydrogen as a Jupyter environment.
# The # %% lines are cell breaks in Hydrogen.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

# %%
# Import 4 years of training data into data frames (Jan 1 2014 - Jan 1 2018 for GOOG, AAPL, AMZN, and MSFT)
# Data gathered from finance.yahoo.com

aaplTrain = pd.read_csv('AAPL_train.csv')
amznTrain = pd.read_csv('AMZN_train.csv')
msftTrain = pd.read_csv('MSFT_train.csv')
googTrain = pd.read_csv('GOOG_train.csv')
googTrain.head()

# %%
# The test data will be from Jan 2 2018 - Jan 2 2019.  Importing this test data...

aaplTest = pd.read_csv('AAPL_test.csv')
amznTest = pd.read_csv('AMZN_test.csv')
msftTest = pd.read_csv('MSFT_test.csv')
googTest = pd.read_csv('GOOG_test.csv')

# %%

# I want to make sure that all training/testing DFs have the same sizes (no missing rows or columns)

dataframes = [aaplTrain, amznTrain, msftTrain, googTrain, aaplTest, amznTest, msftTest, googTest]
for i in dataframes:
    print(i.shape)

# looks good!

# %%

# I'll assume that when predicting Google's price at close, its trading volume, high price, low price, and
# adjusted close price will not be known a priori.  As such, these columns will be dropped from GOOG DFs.

for i in [googTrain, googTest]:
    i.drop(columns=['Volume', 'Adj Close', 'High', 'Low'], inplace=True)

# %%

# At this point, I want to find which parameters most strongly influence Google's closing price.  I'll
# explore this by checking various correlations.

corrDF = pd.DataFrame(data={
    'googOpen': googTrain.Open, 'amznOpen': amznTrain.Open, 'msftOpen': msftTrain.Open, 'aaplOpen': aaplTrain.Open,
    'googClose': googTrain.Close, 'amznClose': amznTrain.Close, 'msftClose': msftTrain.Close, 'aaplClose': aaplTrain.Close,
    'amznHigh': amznTrain.High, 'msftHigh': msftTrain.High, 'aaplHigh': aaplTrain.High,
    'amznLow': amznTrain.Low, 'msftLow': msftTrain.Low, 'aaplLow': aaplTrain.Low
})
corrDF.corr()[['googClose']].sort_values(by='googClose', ascending=False)

# Google's closing price seems to be most strongly correlated with the actions of Amazon, followed by Microsoft,
# and lastly Apple.

# to be continued...
