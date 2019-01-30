# This code was written in the Atom IDE with Hydrogen as a Jupyter environment
# The # %% lines are cell breaks in Hydrogen

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as skll

# predicting a home price based on square footage

prices = {'Area':[2600, 3000, 3200, 3600, 4000], 'Price':[550000, 565000, 610000, 680000, 725000]}
df = pd.DataFrame.from_dict(prices)
df

# %%

# how are these data corrolated?

%matplotlib inline
plt.scatter(df.Area,df.Price)
plt.xlabel('Area(sq ft)')
plt.ylabel('Price (USD)')
plt.show()

# the relation looks fairly linear.  Let's try linear regression.

# %%
line = skll.LinearRegression()
line.fit(df[['Area']], df[['Price']])
# %%

# Let's see how this model matches the data, visually.

m = float(line.coef_)  # the slope
b = float(line.intercept_)  # the y-intercept
plt.scatter(df.Area,df.Price)
xs = [i for i in range(2400,4500,100)]
ys = [m*x+b for x in xs]
plt.plot(df.Area, line.predict(df[['Area']]), color='blue')  # automatically from predict
plt.plot(xs, ys, color='red', linestyle='dashed')  # from manual linear calculation
plt.xlabel('Area(sq ft)')
plt.ylabel('Price (USD)')
plt.show()

# %%
print(line.predict([[3300]]))  # how much would a 3300 square foot house cost?
print()
print(m*3300+b)  # the linear model should predict this explicitly through the linear equation
# %%

# What if I wanted to predict a set of prices all at once?

line.predict(np.array([3000, 3500, 5000]).reshape(-1, 1))

# %%
pass
# %%
pass
# %%
pass
# %%
pass
# %%
pass
