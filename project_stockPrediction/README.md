# stockPrediction

This project seeks to predict the closing price of Google stock based on the behavior of Apple, Amazon, and Microsoft for a given day.

4 years of training data (Jan 1 2014 - Jan 1 2018) was impored from fininace.yahoo.com to train the linear regression, which was tested against one year of test data (Jan 2 2018 - Jan 2 2019), for each of the 4 stocks.


--------------------------------------------------------------------
--------------------------------------------------------------------
--------------------------------------------------------------------

Part 1:

Predicting closing price of Google, based on same-day behavior of Apple, Amazon, and Microsoft

A linear relationship is found between Google's closing price and the following variables - opening price, closing price, high price, and low price - for Amazon, Microsoft, and Apple (Apple's same-day behavior is fairly uncorrelated with Google's closing price, but still behaves somewhat linearly and is included in the model).

sklearn.linear_model.score returned a 96.82% model accuracy when the trained model was run on the test data.

--------------------------------------------------------------------
--------------------------------------------------------------------
--------------------------------------------------------------------

Part 2:

Predicting closing price of Google, based on previous-day behavior of Apple, Amazon, and Microsoft.  Effectively, this predicts "tomorrow"'s stock price based on "today"'s information - much more useful in real application than the result of part 1.

sklearn.linear_model.score returned a 89.63% model accuracy when the trained model was run on the test data.


--------------------------------------------------------------------
--------------------------------------------------------------------
--------------------------------------------------------------------

Part 3:

Updating the results of Part 2 by implementing k-fold cross-validation testing with sklearn.linear_model.LinearRegression and sklearn.linear_model.TheilSenRegressor

Theil-Sen performs slightly better than the linear regression (94.4% vs 94.1% average score) across 5 folds.

Re-fitting the data set from Part 2 with the Theil-Sen regressor returns a 90.02% predictor accuracy, an improvement over the linear regression model used in Part 2.
