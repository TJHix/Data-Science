# -*- coding: utf-8 -*-

import numpy as np
#Our initial weights and biases
def init_params():
    W1 = np.random.rand(10, 784) - 0.5 # returns random value between -0.5 and +0.5
    b1 = np.random.rand(10, 1) - 0.5 # randn(n,m) is size of array
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

# ReLU is our linear function as explained above
def ReLU(Z):
    return np.maximum(Z, 0) # returns z if element greater than zero or return zero if less than zero

#Softmax activation to return a probability between 0 and 1
def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

#Our forward propagation as explained above
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1 #always np.dot as we are dealing with matrices
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def ReLU_deriv(Z):
    return Z > 0
# This works as when booleans are converted to numbers True goes to 1 and False goes to 0.