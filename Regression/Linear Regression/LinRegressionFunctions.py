# -*- coding: utf-8 -*-


def init_params(predictors):
    #initialize model parameters
    #k is a scaling factor that we use to reduce the weights and biases initially
    k = math.sqrt(1 / predictors)
    #we set a random seed so if we re-run this code, we get the same results
    np.random.seed(0)
    weights = np.random.rand(predictors, 1) * 2 * k - k
    biases = np.ones((1, 1)) * 2 * k - k
    return  [weights, biases]

def forward(params, x):
    weights, biases = params
    #multiply x values by w values with matrix multiplication, then add b
    prediction = x @ weights + biases
    return prediction


def mse(actual, predicted):
    #calculate mean squared error
    return np.mean((actual - predicted) ** 2)

def mse_grad(actual, predicted):
    #the derivative of mean squared error
    return predicted - actual

def backward(params, x, lr, grad):
    #multiply the gradient by the x values
    #divide x by the number of rows in x to avoid updates that are too large
    w_grad = (x.T / x.shape[0]) @ grad
    b_grad = np.mean(grad, axis=0)

    params[0] -= w_grad * lr
    params[1] -= b_grad * lr

    return params