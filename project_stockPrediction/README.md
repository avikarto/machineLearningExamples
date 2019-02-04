# stockPrediction

This project seeks to predict the closing price of Google stock based on the behavior of Apple, Amazon, and Microsoft for a given day.  A linear relationship is found between Google's closing price and the following variables - opening price, closing price, high price, and low price - for Amazon, Microsoft, and Apple.

4 years of training data (Jan 1 2014 - Jan 1 2018) was impored from fininace.yahoo.com to train the linear regression, which was tested against one year of test data (Jan 2 2018 - Jan 2 2019), for each of the 4 stocks.

sklearn.linear_model.score returned a 96.82% model accuracy when the trained model was run on the test data.

Note:
- Apple's behavior is fairly uncorrelated with Google's closing price, but still behaves somewhat linearly and is included in the model.
