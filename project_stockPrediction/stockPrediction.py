# Analysis of stock data.  ML is used to predict the closing price of Google stock
# based on the actions of Apple, Amazon, and Microsoft stocks for a given day.

# This code was written in the Atom IDE with Hydrogen as a Jupyter environment.
# The # %% lines are cell breaks in Hydrogen.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as skll
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

############################
########### Part 1 #########
############################

# I'll assume that when predicting Google's price at close, its trading volume, high price, low price, and
# adjusted close price will not be known a priori.  As such, these columns will be dropped from GOOG DFs.

for i in [googTrain, googTest]:
    i.drop(columns=['Volume', 'Adj Close', 'High', 'Low'], inplace=True)
googTrain.head(1)
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

# Google's closing price seems to be most strongly correlated with the behavior of Amazon, followed by Microsoft,
# and lastly Apple.

# %%

# Let's see if a linear model is reasonable.

names = ['Google', 'Apple', 'Amazon', 'Microsoft']  # for index i
vars = ['Open', 'Close', 'High', 'Low']  # for index j
figure, plots = plt.subplots(nrows=4, ncols=4)
figure.set_figheight(20)
figure.set_figwidth(20)
for i in range(4):
    name = [googTrain, aaplTrain, amznTrain, msftTrain][i]
    for j in range(4):
        plots[i, j].set_ylabel('Google Close Price')
        if j == 0:
            plots[i, j].set_xlabel(names[i]+' open price')
            plots[i, j].scatter(name.Open, googTrain.Close)
        elif j == 1 and i != 0:
            plots[i, j].set_xlabel(names[i]+' high price')
            plots[i, j].scatter(name.High, googTrain.Close)
        elif j == 2 and i != 0:
            plots[i, j].set_xlabel(names[i]+' low price')
            plots[i, j].scatter(name.Low, googTrain.Close)
        elif j == 3 and i != 0:
            plots[i, j].set_xlabel(names[i]+' Close price')
            plots[i, j].scatter(name.Close, googTrain.Close)
        if i == 0 and j != 0:
            plots[i, j].axis('off')
plt.show()

# Everything looks pretty linear, aside from Apple.  This was already determined to have the lowest correlation with
# Google's closing price, though, so I'll just roll with a linear model and see how it does.

# %%
# Time to craft the training DFs.
x_train = corrDF.drop(columns='googClose')
y_train = corrDF[['googClose']]

# Let's test a linear model and see how it does

linModel = skll.LinearRegression()
linModel.fit(x_train, y_train)
print(linModel.score(x_train, y_train))

# That seems to have worked pretty well

# %%
# How does the testing data hold up?
x_test = pd.DataFrame(data={
    'googOpen': googTest.Open, 'amznOpen': amznTest.Open, 'msftOpen': msftTest.Open, 'aaplOpen': aaplTest.Open,
    'amznClose': amznTest.Close, 'msftClose': msftTest.Close, 'aaplClose': aaplTest.Close,
    'amznHigh': amznTest.High, 'msftHigh': msftTest.High, 'aaplHigh': aaplTest.High,
    'amznLow': amznTest.Low, 'msftLow': msftTest.Low, 'aaplLow': aaplTest.Low
})
y_test = googTest.Close
linModel.score(x_test, y_test)

# Still pretty good!

# %%

############################
########### Part 2 #########
############################

# In this part, I want to predict Tomorrow's closing price of Google based on what the 4 stocks did Today.
# In this case, its trading volume, high price, low price, and close price of the previous day becomes relevant.
# Since I dropped a lot of this data previously, I'll reimport the DFs for google

googTrain2 = pd.read_csv('GOOG_train.csv')
googTest2 = pd.read_csv('GOOG_test.csv')
for g in [googTrain2, googTest2]:
    g['nextClose'] = g.Close.copy()

# %%

# The primary changes that need to happen in this part is that the target values (nextClose) need to be redefined by
#   shifting Google's nextClose columns back one day across all data, and then dropping the last element:

lenTrain = len(googTrain2)
lenTest = len(googTest2)
print('Train tail...', googTrain2.nextClose.tail(2).values)
print('Test head...', googTest2.nextClose.head(2).values)
print('Test tail...', googTest2.nextClose.tail(2).values)

for i in range(lenTrain):
    if i != lenTrain-1:
        googTrain2.loc[i, 'nextClose'] = googTrain2.loc[i+1, 'nextClose']
    else:
        googTrain2.loc[i, 'nextClose'] = googTest2.loc[0, 'nextClose']

print('New train tail...', googTrain2.nextClose.tail(2).values)

for i in range(lenTest-1):
        googTest2.loc[i, 'nextClose'] = googTest2.loc[i+1, 'nextClose']

print('New test head...', googTest2.nextClose.head(2).values)

googTest2.drop([lenTest-1], inplace=True)

print('New test tail...', googTest2.nextClose.tail(2).values)
print('old test length: ', lenTest, '.... new test length: ', len(googTest2))

# %%

# Now, the parameters need to be appended with additional information regarding Google.

xTrain2 = pd.DataFrame(data={  # the now-previous day info for all compaines
    'googOpen': googTrain2.Open, 'amznOpen': amznTrain.Open, 'msftOpen': msftTrain.Open, 'aaplOpen': aaplTrain.Open,
    'googClose': googTrain2.Close, 'amznClose': amznTrain.Close, 'msftClose': msftTrain.Close, 'aaplClose': aaplTrain.Close,
    'googHigh': googTrain2.High, 'amznHigh': amznTrain.High, 'msftHigh': msftTrain.High, 'aaplHigh': aaplTrain.High,
    'googLow': googTrain2.Low, 'amznLow': amznTrain.Low, 'msftLow': msftTrain.Low, 'aaplLow': aaplTrain.Low
})
xTest2 = pd.DataFrame(data={  # the now-previous day info for all compaines
    'googOpen': googTest2.Open, 'amznOpen': amznTest.Open, 'msftOpen': msftTest.Open, 'aaplOpen': aaplTest.Open,
    'googClose': googTest2.Close, 'amznClose': amznTest.Close, 'msftClose': msftTest.Close, 'aaplClose': aaplTest.Close,
    'googHigh': googTest2.High, 'amznHigh': amznTest.High, 'msftHigh': msftTest.High, 'aaplHigh': aaplTest.High,
    'googLow': googTest2.Low, 'amznLow': amznTest.Low, 'msftLow': msftTest.Low, 'aaplLow': aaplTest.Low
})
xTest2.drop([lenTest-1], inplace=True)
yTrain2 = googTrain2.nextClose
yTest2 = googTest2.nextClose

linModel2 = skll.LinearRegression()
linModel2.fit(xTrain2, yTrain2)
linModel2.score(xTest2, yTest2)

# The model predicts the following day's closing value of Google stock with 89.625% accuracy, based on previous day data
# Not horribly bad, but not particularly useful either.  Maybe predictions within 5% would be worth acting on.

############################
########### Part 3 #########
############################

# Let's see if we can get a better prediction with cross-validation and different models.
# We can start by combining all of the data.

df3X = xTrain2.append(xTest2)
df3Y = googTrain2['nextClose'].append(googTest2['nextClose'])


# %%
# Train and test various models with k-fold cross-validation

linModel3 = skll.LinearRegression()
scoresLinear = []
TheilSen3 = skll.TheilSenRegressor()  # a good multivariate regressor
scoresTheilSen = []


def getScore(model, xTrain, xTest, yTrain, yTest):
    model.fit(xTrain, yTrain)
    return model.score(xTest, yTest)


# %%

import sklearn.model_selection as sklms
splits=5
kf = sklms.KFold(n_splits=splits, shuffle=False)

savedPredictionsL = []
savedPredictionsT = []
savedActual = []
for iTrain, iTest in kf.split(X=df3X, y=df3Y):
    xTrain3 = []
    xTest3 = []
    yTrain3 = []
    yTest3 = []
    for i in iTrain:
        xTrain3.append(df3X.iloc[i])
        yTrain3.append(df3Y.iloc[i])
    for i in iTest:
        xTest3.append(df3X.iloc[i])
        yTest3.append(df3Y.iloc[i])
    scoresLinear.append(getScore(linModel3, xTrain3, xTest3, yTrain3, yTest3))
    scoresTheilSen.append(getScore(TheilSen3, xTrain3, xTest3, yTrain3, yTest3))
    # for analysis below...
    savedPredictionsL.append(linModel3.predict(xTest3))
    savedPredictionsT.append(TheilSen3.predict(xTest3))
    savedActual.append(yTest3)

# %%

print('Linear model scores: \n', scoresLinear)
print('Average Linear model score: \n', np.average(scoresLinear), '\n')
print('TheilSen model scores: \n', scoresTheilSen)
print('Average TheilSen model score: \n', np.average(scoresTheilSen))

# about between about 90-99% accuracy on both linear and theil-sen models with k-folds.
# This is a wider range than I'd have liked, but almost certainly due to small sample size.

# %%
# Visualizing the predictions at each fold

for i in range(splits):
    print('For split number ', i+1, '...')
    plt.scatter(range(len(savedPredictions[i])), savedActual[i], color='blue', label='actual', alpha=0.3)
    plt.scatter(range(len(savedPredictionsL[i])), savedPredictionsL[i], color='black', marker='_', label='Linear Prediction')
    plt.scatter(range(len(savedPredictionsT[i])), savedPredictionsT[i], color='red', marker='|', label='TheilSen Prediction')
    plt.xlabel('index')
    plt.ylabel('Google Closing Price')
    plt.legend()
    plt.show()

# This appears to be predicting pretty accurately, in both cases.

# %%
# re-training the part 2 data set with the Theil-Sen model:

TheilSenModel2 = skll.TheilSenRegressor()
TheilSenModel2.fit(xTrain2, yTrain2)
TheilSenModel2.score(xTest2, yTest2)

# The new model is 90.02% accurate, compared to the 89.63% accuracy of the linear model.
