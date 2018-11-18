# Lesson 2. Neural Networks
# 21. Cross-Entropy 2
import numpy as np


# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.
def cross_entropy(Y, P):
    y = np.float64(Y)
    p = np.float64(P)
    return -np.sum(y*np.log(p) + (1-y)*np.log(1-p))


test_Y = [1, 0, 1, 1]
test_P = [0.4, 0.6, 0.1, 0.5]
print(cross_entropy(test_Y, test_P))
