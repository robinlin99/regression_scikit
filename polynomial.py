# This is an example of the use of Polynomial Features
import matplotlib
import numpy as np
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
from sklearn import datasets, linear_model
from sklearn.preprocessing import PolynomialFeatures



# Training set
x_train = [[2], [9], [12], [16], [20]]
y_train = [[9], [30], [10], [10], [100]]
xx = np.linspace(0, 20, 100)

# This is the linear model
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# This is the polynomial model with degree n
poly = PolynomialFeatures(degree=9)
poly1 = PolynomialFeatures(degree=4)
x_train_poly = poly.fit_transform(x_train)
x_train_poly1 = poly1.fit_transform(x_train)
regressor_n = LinearRegression()
regressor_n_1 = LinearRegression()
regressor_n.fit(x_train_poly, y_train)
regressor_n_1.fit(x_train_poly1, y_train)

# Loading the regression coefficients
a = regressor_n.coef_[0][0]
b = regressor_n.coef_[0][1]
c = regressor_n.coef_[0][2]
xx_n = poly.transform(xx.reshape(xx.shape[0], 1))
xx_n1 = poly1.transform(xx.reshape(xx.shape[0], 1))


matplotlib.pyplot.scatter(x_train,y_train,color='black')
plt.plot(xx, regressor_n.predict(xx_n), c='r',linestyle='--')
plt.plot(xx, regressor_n_1.predict(xx_n1), c='b',linestyle='--')
plt.axis([0, 25, 0, 210])
plt.grid(True)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Figure 1: Polynomial regression tutorial')
plt.savefig('Figure 1: Polynomial regression tutorial')
