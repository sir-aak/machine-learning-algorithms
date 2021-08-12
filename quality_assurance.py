import numpy as np
import matplotlib.pyplot as plt
import machine_learning_algorithms as mla


# 2d example data set from Coursera course Machine Learning
# polynomial features of fifth degree are added and logistic regression is used 
# to estimate whether the microchips from a fabrication plant pass quality 
# assurance in dependence on two tests
data = np.loadtxt("data/microchip_tests.txt", delimiter=",")
X    = data[:, 0:2]
y    = data[:, 2]


def plotCostHistory (X, y, standardization=False, addPolyFeats=False, degree=5, 
                     iterations=400000, learningRate=0.1, regularization=0.0):
    
    lr = mla.LogisticRegression(X, y, standardization=standardization, 
                                addPolyFeats=addPolyFeats, degree=degree)
    
    costHistory = lr.gradientDescent(iterations=iterations, 
                                     learningRate=learningRate, 
                                     regularization=regularization)
    
    plt.figure()
    ax = plt.gca()
    plt.subplots_adjust(top=0.98, bottom=0.13, left=0.14, right=0.98)
    plt.rcParams.update({"font.size": 18})
    plt.plot(np.arange(1, iterations+1), costHistory, color="black")
    plt.text(0.4, 0.9, "learning rate = " + str(learningRate), 
             transform=ax.transAxes)
    plt.text(0.4, 0.8, "regularization = " + str(regularization), 
             transform=ax.transAxes)
    plt.xlabel("number of iterations")
    plt.ylabel(r"cost function $J$")
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    plt.grid(color="lightgray")
    plt.show()


def plotData (X, y, standardization=False, addPolyFeats=False, degree=5, 
              iterations=400000, learningRate=0.1, regularization=0.0):
    
    Xfailed = X[y == False]
    Xpassed = X[y == True]
    
    lr = mla.LogisticRegression(X, y, standardization=standardization, 
                                addPolyFeats=addPolyFeats, degree=degree)
    
    lr.gradientDescent(iterations=iterations, learningRate=learningRate, 
                       regularization=regularization)
    
    lower  = -1.2
    upper  = +1.2
    step   = 0.01
    test1  = np.arange(lower, upper + 1e-2, step)
    test2  = np.arange(lower, upper + 1e-2, step)
    extent = np.array([lower, upper + 1e-2, lower, upper + 1e-2])
    scores = np.array(np.meshgrid(test1, test2)).T.reshape(-1, 2)
    
    if addPolyFeats == True:
        scores = lr.addPolynomialFeatures(scores, degree=degree)
    
    prediction   = lr.predict(scores).reshape(241, 241)
    boundaryBool = np.logical_and(prediction > 0.48, prediction < 0.52)
    boundary     = np.ma.masked_where(boundaryBool == True, prediction)
    
    plt.figure()
    plt.subplots_adjust(top=0.98, bottom=0.14, left=0.08, right=0.97)
    plt.rcParams.update({"font.size": 18})
    plt.scatter(Xpassed[:, 0], Xpassed[:, 1], color="darkblue", marker="x", 
                label="passed", zorder=2)
    plt.scatter(Xfailed[:, 0], Xfailed[:, 1], color="red", marker="x", 
                label="failed", zorder=2)
    plt.scatter(2, 2, color="white", marker="o", label="boundary")
    admission = plt.imshow(boundary, extent=extent, origin="lower", 
                           cmap="coolwarm_r", vmin=0.0, vmax=1.0)
    cb = plt.colorbar(admission)
    cb.set_label("probability to pass the test", fontsize=18)
    plt.xlim(-1.2, 1.2)
    plt.ylim(-1.2, 1.2)
    plt.xlabel("microchip test 1")
    plt.ylabel("microchip test 2")
    plt.legend(loc="lower left", fontsize=15)
    plt.grid(color="darkgray")
    plt.show()


plotCostHistory(X, y, standardization=True, addPolyFeats=True, degree=5, 
                iterations=400000, learningRate=0.1, regularization=1.0)

plotData(X, y, standardization=True, addPolyFeats=True, degree=5, 
         iterations=400000, learningRate=0.1, regularization=1.0)
