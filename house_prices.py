import numpy as np
import matplotlib.pyplot as plt
import machine_learning_algorithms as mla


# 2d example data set from Coursera course Machine Learning
# linear regression is used to estimate the prices of houses in dependence 
# on the size of the house in square feet and on the number of bedrooms
data = np.loadtxt("data/house_prices.txt", delimiter=",")
X    = data[:, 0:2]
y    = data[:, 2]


def plotCostHistory (X, y, standardization=False, iterations=50, 
                     learningRate=1e-7, regularization=0.0):
    
    lr = mla.LinearRegression(X, y, standardization)
    
    costHistory = lr.gradientDescent(iterations=iterations,
                                     learningRate=learningRate, 
                                     regularization=regularization)
    
    plt.figure()
    ax = plt.gca()
    plt.subplots_adjust(top=0.95, bottom=0.13, left=0.13, right=0.98)
    plt.rcParams.update({"font.size": 18})
    plt.plot(np.arange(1, iterations+1), costHistory, color="black")
    plt.text(0.4, 0.9, "learning rate = " + str(learningRate), 
              transform=ax.transAxes)
    plt.text(0.4, 0.8, "regularization = " + str(regularization), 
              transform=ax.transAxes)
    plt.xlabel("number of iterations")
    plt.ylabel(r"cost function $J$")
    plt.grid(color="lightgray")
    plt.show()


def plotData (X, y, size=True, bedrooms=True, standardization=False, 
              iterations=50, learningRate=1e-7, regularization=0.0):
    
    lr = mla.LinearRegression(X, y, standardization)
    
    lr.gradientDescent(iterations=iterations, learningRate=learningRate, 
                       regularization=regularization)
    
    if size == True:
        plt.figure()
        ax = plt.gca()
        plt.subplots_adjust(top=0.98, bottom=0.14, left=0.23, right=0.97)
        plt.rcParams.update({"font.size": 18})
        plt.scatter(X[:, 0], y, color="black", marker="x", zorder=2)
        plt.scatter(X[:, 0], lr.predict(X), color="red", zorder=3)
        plt.text(0.60, 0.25, r"$\theta_0$ = " + str(np.around(lr.theta[0], 1)), 
                  transform=ax.transAxes)
        plt.text(0.60, 0.15, r"$\theta_1$ = " + str(np.around(lr.theta[1], 1)), 
                  transform=ax.transAxes)
        plt.text(0.60, 0.05, r"$\theta_2$ = " + str(np.around(lr.theta[2], 1)), 
                  transform=ax.transAxes)
        plt.xlabel(r"house size in ft$^2$")
        plt.ylabel("house price in $")
        plt.grid(color="lightgray")
        plt.show()
    
    if bedrooms ==True:
        plt.figure()
        ax = plt.gca()
        plt.subplots_adjust(top=0.98, bottom=0.14, left=0.23, right=0.97)
        plt.rcParams.update({"font.size": 18})
        plt.scatter(X[:, 1], y, color="black", marker="x", zorder=2)
        plt.scatter(X[:, 1], lr.predict(X), color="red", zorder=3)
        plt.text(0.05, 0.90, r"$\theta_0$ = " + str(np.around(lr.theta[0], 1)), 
                  transform=ax.transAxes)
        plt.text(0.05, 0.80, r"$\theta_1$ = " + str(np.around(lr.theta[1], 1)), 
                  transform=ax.transAxes)
        plt.text(0.05, 0.70, r"$\theta_2$ = " + str(np.around(lr.theta[2], 1)), 
                  transform=ax.transAxes)
        plt.xlabel(r"number of bedrooms")
        plt.ylabel("house price in $")
        plt.grid(color="lightgray")
        plt.show()


plotCostHistory(X, y, standardization=True, learningRate=1)
plotData(X, y, size=True, bedrooms=True, standardization=True, learningRate=1)
