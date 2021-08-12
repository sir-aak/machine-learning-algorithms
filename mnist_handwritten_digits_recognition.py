import numpy as np
import matplotlib.pyplot as plt
import gzip
import machine_learning_algorithms as mla


# Loads MNIST data set for handwritten digits consisting of a training set
# and a test set of image data. Each image represents a handwritten number 
# between 0 and 9 and has a size of 28 x 28 = 784 pixels, each of them 
# corresponding a value between 0 and 255. The parameter n_train_images 
# determines how many training samples are loaded for the training, up to a 
# maximum of 60 k training images.
# Returns normalized feature matrices for the training set of n_train_images 
# images and the test set of 10 k test images as well as their corresponding 
# labels.
def loadData (n_train_images=5000):
    
    image_size     = 28
    n_test_images  = 10000
    
    # read data and labels from file
    file_train_images = gzip.open("data/train-images-idx3-ubyte.gz", "r")
    file_test_images  = gzip.open("data/t10k-images-idx3-ubyte.gz", "r")
    file_train_labels = gzip.open("data/train-labels-idx1-ubyte.gz", "r")
    file_test_labels  = gzip.open("data/t10k-labels-idx1-ubyte.gz", "r")
    
    # ignoring the first bytes
    file_train_images.read(16)
    file_test_images.read(16)
    file_train_labels.read(8)
    file_test_labels.read(8)
    
    # create normalized feature matrix from train data
    buffer  = file_train_images.read(image_size * image_size * n_train_images)
    X_train = np.frombuffer(buffer, dtype=np.uint8).astype(np.float32) / 255.0
    X_train = X_train.reshape(n_train_images, image_size * image_size)
    
    # create normalized feature matrix from test data
    buffer  = file_test_images.read(image_size * image_size * n_test_images)
    X_test  = np.frombuffer(buffer, dtype=np.uint8).astype(np.float32) / 255.0
    X_test  = X_test.reshape(n_test_images, image_size * image_size)
    
    # create label vector from train data
    buffer  = file_train_labels.read(n_train_images)
    y_train = np.frombuffer(buffer, dtype=np.uint8).astype(np.int8)
    
    # create label vector from test data
    buffer  = file_test_labels.read(n_test_images)
    y_test  = np.frombuffer(buffer, dtype=np.uint8).astype(np.int8)
    
    return (X_train, y_train, X_test, y_test)


# plots the first nrow_digits * nrow_digits digits of the MNIST training set
def viewData (nrow_digits=10):
    
    n_train_images = np.power(nrow_digits, 2)
    (X_train, y_train, X_test, y_test) = loadData(n_train_images)
    
    X_train = X_train.reshape(n_train_images, 28, 28)
    
    image = np.empty((28 * nrow_digits, 28 * nrow_digits))
    
    for i in range (nrow_digits):
        
        X = np.empty((28, 28))
        
        for j in range (nrow_digits):
            
            if j == 0:
                X = X_train[j + nrow_digits * i]
            
            else:
                X = np.append(X, X_train[j + nrow_digits * i], axis=1)
        
        image[i * 28:(i + 1) * 28, :] = X
    
    plt.imshow(image, cmap="binary")
    plt.xticks(ticks=[])
    plt.yticks(ticks=[])
    plt.show()


# Trains an artificial neural network of sigmoid neurons to recognize 
# handwritten digits for a given network structure and prints the prediction 
# precisions. network_structure should be a list of the number of units in each
# layer. Since the data set consists of images with size 28 x 28 = 728, the 
# first layer must have 784 units. The output layer should have 10 units in 
# order to recognize handwritten digits between 0 and 9.
def trainNeuralNetwork (n_train_images=5000, network_structure=[784, 25, 10], 
              iterations=400, learning_rate=1.0, regularization=0.0):
    
    (X_train, y_train, X_test, y_test) = loadData(n_train_images)
    
    NN = mla.NeuralNetwork(X_train, y_train, network_structure)
    
    J_hist = NN.gradientDescent(iterations=iterations, 
                                learning_rate=learning_rate, 
                                regularization=regularization)
    
    p = NN.predict(X_train)
    print("train precision: ", np.mean(p == y_train) * 100, "%")
    
    p = NN.predict(X_test)
    print("test precision: ", np.mean(p == y_test) * 100, "%")
    
    plt.plot(J_hist)


trainNeuralNetwork(n_train_images=5000, network_structure=[784, 25, 10], 
                   iterations=400, learning_rate=1.0, regularization=0.0)

viewData(nrow_digits=10)
