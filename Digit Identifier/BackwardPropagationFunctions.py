# -*- coding: utf-8 -*-

import numpy as np
# function that takes our labels and turns them into an array
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1)) # tuple of size, classes are zero through nine, so add 1 to get 10 (correct number of output classes) 
    one_hot_Y[np.arange(Y.size), Y] = 1 # For each column go the row specified by Y (our label) and set to 1
    one_hot_Y = one_hot_Y.T # Transposed
    return one_hot_Y

# Backward propagation function
# Inserting our errors for each stage as explained above
def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    m = Y.size
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2