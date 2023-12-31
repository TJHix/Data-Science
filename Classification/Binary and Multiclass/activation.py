# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 11:04:36 2023

@author: Tom Hicks
"""

'activation taken from dataquest, authour: Vik Paruchi'

from network import Module
import numpy as np

class Relu(Module):
    def __init__(self):
        pass

    def forward(self, x):
        return np.maximum(x, 0)

    def backward(self, grad, lr, prev_hidden):
        return np.multiply(grad, np.heaviside(prev_hidden, 0))