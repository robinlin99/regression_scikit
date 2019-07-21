from sklearn import datasets, linear_model
import matplotlib
import numpy as np
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os


x = np.array([[0],[2],[6],[8],[15],[19],[30],[33]])
y = np.array([[0],[3],[4],[7],[13],[15],[18],[21]])
x_0 = np.array(range(40))


regr = linear_model.LinearRegression()
regr.fit(x,y)
x=x.flatten()
y=y.flatten()

m = regr.coef_[0][0]
b = regr.intercept_[0]

def graph(formula, x_range):
    x = np.array(x_range)
    y = eval(formula)
    plt.plot(x, y)


matplotlib.pyplot.scatter(x,y,color='green')
graph('m*x+b',[0,40])
plt.savefig('figure')







