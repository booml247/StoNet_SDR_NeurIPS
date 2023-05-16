import numpy as np

def relu(x):
    x[x < 0] = 0
    return x