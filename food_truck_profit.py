import numpy as np
import matplotlib.pyplot as plt
import machine_learning_algorithms as mla


# 1d example data set from Coursera course Machine Learning
# linear regression is used to estimate the profit of a food truck in $10,000s 
# in dependence on the population of the city in 10,000s
data = np.loadtxt("data/food_truck_profit.txt", delimiter=",")
X    = data[:, 0]
y    = data[:, 1]


def plotCostHistory (X, y, standardization=False, iterations=1500, 
                     learningRate=0.01, regularization=0.0):
    
    lr = mla.LinearRegression(X, y, standardization)
    
    costHistory = lr.gradientDescent(iterations=iterations,
                                     learningRate=learningRate, 
                                     regularization=regularization)    
    
    plt.figure()
    ax = plt.gca()
    plt.subplots_adjust(top=0.98, bottom=0.13, left=0.13, right=0.98)
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


def plotData (X, y, standardization=False, iterations=1500, learningRate=0.01, 
              regularization=0.0):
    
    lr = mla.LinearRegression(X, y, standardization=standardization)
    
    lr.gradientDescent(iterations=iterations, learningRate=learningRate, 
                       regularization=regularization)
    
    plt.figure()
    ax = plt.gca()
    plt.subplots_adjust(top=0.98, bottom=0.14, left=0.13, right=0.97)
    plt.rcParams.update({"font.size": 18})
    plt.scatter(X, y, color="black", marker="x", zorder=2)
    plt.plot(X, lr.predict(X), color="red")
    plt.text(0.65, 0.15, r"$\theta_0$ = " + str(np.around(lr.theta[0], 4)), 
              transform=ax.transAxes)
    plt.text(0.65, 0.05, r"$\theta_1$ = " + str(np.around(lr.theta[1], 4)), 
              transform=ax.transAxes)
    plt.xlabel("population of city in 10,000s")
    plt.ylabel("profit in $10,000s")
    plt.xlim(0.0, 25.0)
    plt.grid(color="lightgray")
    plt.show()


def plotCostFunction (X, y, standardization=False):
    
    lr = mla.LinearRegression(X, y, standardization=standardization)
    
    theta0 = np.arange(-5.0, 15.0 + 1e-4, 0.1)
    theta1 = np.arange(-5.0, 15.0 + 1e-4, 0.1)
    
    Theta0, Theta1, Cost = lr.getCost(theta0, theta1)
    
    plt.figure()
    ax = plt.axes(projection="3d")
    plt.rcParams.update({"font.size": 18})
    plt.subplots_adjust(top=1.05, bottom=0.05, left=0.01, right=0.99)
    ax.plot_surface(Theta0, Theta1, Cost, cmap="viridis")
    ax.set_xlabel(r"$\theta_0$", fontsize=18, labelpad=15.0)
    ax.set_ylabel(r"$\theta_1$", fontsize=18, labelpad=15.0)
    ax.set_zlabel(r"cost $J$", fontsize=18, labelpad=15.0)
    plt.show()
    
    plt.figure()
    plt.subplots_adjust(top=0.97, bottom=0.14, left=0.13, right=0.96)
    plt.rcParams.update({'font.size': 18})
    plt.contour(Theta0, Theta1, Cost, 100)
    plt.xlabel(r"$\theta_0$")
    plt.ylabel(r"$\theta_1$")
    plt.show()


plotCostHistory(X, y, standardization=True)
plotData(X, y, standardization=True)
plotCostFunction(X, y, standardization=True)
