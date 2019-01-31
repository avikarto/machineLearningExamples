# This code was written in the Atom IDE with Hydrogen as a Jupyter environment
# The # %% lines are cell breaks in Hydrogen

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as skll
%matplotlib inline

# predicting a home price based on square footage, number of bedrooms, and age of house
# given input data, find the price of homes with the following properties -
# 1: 3000 sq ft, 3 bedrooms, 40 years old
# 2: 2500 sq ft, 4 bedrooms, 5 years old

# %%

data = {
    'Area': [2600, 3000, 3200, 3600, 4000],
    'Bedrooms': [3, 4, np.nan, 3, 5],  # a missing piece of data!
    'Age': [20, 15, 18, 30, 8],
    'Price': [550000, 565000, 610000, 595000, 760000]
}

df = pd.DataFrame.from_dict(data)
df

# %%

# I'll start by filling the NaN value by assuming the average of the column (rounded to the nearest integer,
# because who has a house with fractions of bedrooms?)

df.Bedrooms.fillna(np.round(df.Bedrooms.median()), inplace=True)
df
# %%

# How does each variable influence the price?

df.corr()['Price']

# Area and Bedrooms appear to be fairly well-correlated with price, suggesting a strong influence
# Age seems to influence the price less than the other factors

# %%
cols = list(df.columns)
cols.remove('Price')
for i in cols:
    plt.xlabel(i)
    plt.ylabel('Price')
    plt.scatter(df[[i]], df[['Price']])
    plt.show()

# The data is too sparse to see a real pattern, but I'll assume a linear relation for the sake of this example

# %%
linReg = skll.LinearRegression()
# Area, Bedrooms, and Age are the independent variables, and Price is the dependent
linReg.fit(df[['Area', 'Bedrooms', 'Age']], df.Price)
# %%

# Finding numerical coefficients for the model: Price = c1*Area + c2*Bedrooms + c2*Age + intercept

print(linReg.coef_)  # 3 outputs are c1, c2, c3
print(linReg.intercept_)
# %%

# Finally, let's predict the price of the following homes:
# 1: 3000 sq ft, 3 bedrooms, 40 years old
# 2: 2500 sq ft, 4 bedrooms, 5 years old

linReg.predict([[3000, 3, 40]])
# %%
linReg.predict([[2500, 4, 5]])
