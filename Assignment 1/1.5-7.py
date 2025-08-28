import numpy as np
from scipy.linalg import circulant

n_f = 4
exponents = np.outer(np.arange(n_f), np.arange(n_f))

F4 = np.power(1j, exponents)

first_row_c4 = np.array([2, -1, 0, -1])
C4_scipy = circulant(first_row_c4)

_, Q = np.linalg.eigh(C4_scipy)

print(Q)