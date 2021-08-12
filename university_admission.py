import numpy as np
import matplotlib.pyplot as plt
import machine_learning_algorithms as mla


# 2d example data set from Coursera course Machine Learning
# in one variant, polynomial features of fourth degree are added and 
# logistic regression is used to estimate the applicant's university admission 
# chance in dependence on the results of two exams
data = np.loadtxt("data/exam_scores.txt", delimiter=",")
X    = data[:, 0:2]
y    = data[:, 2]


def plotCostHistory (X, y, standardization=False, addPolyFeats=False, degree=5, 
                     iterations=400000, learningRate=1e-3, regularization=0.0):
    
    lr = mla.LogisticRegression(X, y, standardization=standardization, 
                                addPolyFeats=addPolyFeats, degree=degree)
    
    costHistory = lr.gradientDescent(iterations=iterations, 
                                     learningRate=learningRate, 
                                     regularization=regularization)
    
    # trained parameters without polynomial features:
    
    # non standardized for 3.000.000 iterations:
    # lr.theta = np.array([-21.06746245, 0.17350979, 0.16833432])
    
    # standardized for 400.000 iterations:
    # lr.theta = np.array([1.65840542, 3.86476728, 3.60126676])
    
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
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    plt.grid(color="lightgray")
    plt.show()


def plotData (X, y, standardization=False, addPolyFeats=False, degree=5, 
              iterations=400000, learningRate=1e-3, regularization=0.0):
    
    Xfailed = X[y == False]
    Xpassed = X[y == True]
    
    lr = mla.LogisticRegression(X, y, standardization=standardization, 
                                addPolyFeats=addPolyFeats, degree=degree)
    
    lr.gradientDescent(iterations=iterations, learningRate=learningRate, 
                       regularization=regularization)
    
    # trained parameters without polynomial features:
    
    # non standardized for 3.000.000 iterations:
    # lr.theta = np.array([-21.06746245, 0.17350979, 0.16833432])
    
    # standardized for 400.000 iterations:
    # lr.theta = np.array([1.65840542, 3.86476728, 3.60126676])
    
    lower  = 0
    upper  = 101
    step   = 0.5
    exam1  = np.arange(lower, upper, step)
    exam2  = np.arange(lower, upper, step)
    extent = np.array([lower, upper + 1e-2, lower, upper + 1e-2])
    scores = np.array(np.meshgrid(exam1, exam2)).T.reshape(-1, 2)
    
    if addPolyFeats == True:
        scores = lr.addPolynomialFeatures(scores, degree=degree)
    
    prediction   = lr.predict(scores).reshape(202, 202)
    boundaryBool = np.logical_and(prediction > 0.47, prediction < 0.53)
    boundary     = np.ma.masked_where(boundaryBool == True, prediction)
    
    if addPolyFeats == False:
        
        x_bounds = np.array([np.min(X[:, 0]), np.max(X[:, 0])])
        
        if standardization == True:
            y_bounds = -(lr.theta[1] * (x_bounds - lr.mu) / lr.sigma \
                         + lr.theta[0]) / lr.theta[2]
            y_bounds = y_bounds * lr.sigma + lr.mu
        
        else:
            y_bounds = -(lr.theta[1] * x_bounds + lr.theta[0]) / lr.theta[2]
    
    plt.figure()
    plt.subplots_adjust(top=0.98, bottom=0.14, left=0.05, right=0.97)
    plt.rcParams.update({"font.size": 18})
    passed = plt.scatter(Xpassed[:, 0], Xpassed[:, 1], color="darkblue", 
                         marker="x", zorder=2)
    failed = plt.scatter(Xfailed[:, 0], Xfailed[:, 1], color="red", 
                         marker="x", zorder=2)
    
    if addPolyFeats == False:
        bounds,   = plt.plot(x_bounds, y_bounds, color="black")
        admission = plt.imshow(prediction, extent=extent, origin="lower", 
                               cmap="coolwarm_r", vmin=0.0, vmax=1.0)
    
    else:
        bounds    = plt.scatter(-1, -1, marker="o", color="white")
        admission = plt.imshow(boundary, extent=extent, origin="lower", 
                               cmap="coolwarm_r", vmin=0.0, vmax=1.0)
    
    cb = plt.colorbar(admission)
    cb.set_label("admission probability", fontsize=18)
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.xlabel("exam 1 score")
    plt.ylabel("exam 2 score")
    plt.legend([passed, failed, bounds], 
               ["passed", "failed", "decision boundary"], loc="lower left")
    plt.grid(color="lightgray")
    plt.show()


plotCostHistory(X, y, standardization=True, addPolyFeats=True, degree=4, 
                iterations=400000, learningRate=1e-3, regularization=1.0)

plotData(X, y, standardization=True, addPolyFeats=True, degree=4, 
         iterations=400000, learningRate=1e-3, regularization=1.0)
