"""
Linear Algebra with Numpy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

a = np.array([1, 3, 2, 4])
A = np.array([[3, 1, 2],
              [2, 3, 4]])
B = np.array([[0, 1],
              [2, 3],
              [4,5]])
C = np.array([[0, 1],
              [2, 3],
              [4, 5],
              [0, 1],
              [2, 3],
              [4, 5]])

print("A is a {} matrix".format(A.shape))
print("B is a {} matrix".format(B.shape))
print("C is a {} matrix".format(C.shape))
