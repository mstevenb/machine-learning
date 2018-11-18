# Lesson 2. Neural Networks
# 16. Softmax

import numpy as np
from math import exp


# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.
def softmax_math(L):
    denom = sum(exp(x) for x in L)
    return [exp(x)/denom for x in L]


def softmax_np(L):
    expL = np.exp(L)
    denom = sum(expL)
    return np.divide(expL, denom)


my_list = [2, 1, 0]
print(softmax_math(my_list))
print(softmax_np(my_list))
